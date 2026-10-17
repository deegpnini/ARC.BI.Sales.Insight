"""
ARC.BI.Sales.Insight — Validacao v3 (alvo em razao).
"""
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "vendas_complexo_limpo.csv"
OUTPUT_DIR = ROOT / "outputs"

N_LAGS = 6
RANDOM_STATE = 42
N_SPLITS = 5


def load_monthly(path):
    print("Carregando dataset...")
    df = pd.read_csv(path, parse_dates=["data_venda"])
    print(f"   {len(df):,} registros")
    monthly = (
        df.set_index("data_venda")
        .resample("MS")["receita_total"].sum()
        .rename("receita").to_frame().reset_index()
        .rename(columns={"data_venda": "mes"})
    )
    return monthly.sort_values("mes").reset_index(drop=True)


def add_features(df, n_lags=N_LAGS):
    out = df.copy()
    out["mes_num"] = out["mes"].dt.month
    for lag in range(1, n_lags + 1):
        out[f"lag_{lag}"] = out["receita"].shift(lag)
    out["roll_mean_3"] = out["receita"].shift(1).rolling(3).mean()
    out["roll_mean_6"] = out["receita"].shift(1).rolling(6).mean()
    out["roll_std_3"] = out["receita"].shift(1).rolling(3).std()
    return out.dropna().reset_index(drop=True)


def seasonal_naive_forecast(history, test_dates):
    hist_idx = history.set_index("mes")["receita"]
    preds = []
    for test_date in test_dates:
        target = test_date - pd.DateOffset(years=1)
        if target in hist_idx.index:
            preds.append(hist_idx.loc[target])
        elif len(hist_idx) >= 12:
            preds.append(hist_idx.iloc[-12])
        else:
            preds.append(hist_idx.mean())
    return np.array(preds)


def naive_forecast(history, n_steps):
    return np.array([history["receita"].iloc[-1]] * n_steps)


def baseline_naive_seasonal(history, y_lag1_test, test_months):
    y_train = history["receita"].reset_index(drop=True)
    ratios = (y_train / y_train.shift(1)).dropna()
    months_train = history["mes"].dt.month.iloc[1:].reset_index(drop=True)
    factor_by_month = pd.Series(ratios.values, index=months_train).groupby(level=0).mean()
    factors_test = factor_by_month.reindex(test_months).fillna(1.0).values
    return y_lag1_test * factors_test


def evaluate_fold(train, test, feature_cols):
    train = train.copy()
    test = test.copy()
    train["ratio"] = train["receita"] / train["lag_1"]
    test["ratio"] = test["receita"] / test["lag_1"]

    X_train, y_train_ratio = train[feature_cols], train["ratio"]
    X_test = test[feature_cols]

    model = lgb.LGBMRegressor(
        n_estimators=200, learning_rate=0.08,
        num_leaves=15, max_depth=4,
        min_child_samples=3, subsample=1.0, colsample_bytree=1.0,
        random_state=RANDOM_STATE, verbosity=-1,
    )
    model.fit(X_train, y_train_ratio)
    ratio_pred = model.predict(X_test)
    y_pred = test["lag_1"].values * ratio_pred

    y_seasonal = seasonal_naive_forecast(train, test["mes"])
    y_naive = naive_forecast(train, len(test))
    y_det = baseline_naive_seasonal(train, test["lag_1"].values, test["mes"].dt.month.values)

    y_true = test["receita"].values

    def mape(yt, yp):
        return np.mean(np.abs((yt - yp) / yt)) * 100
    def mae(yt, yp):
        return np.mean(np.abs(yt - yp))

    results = {
        "n_train": len(train), "n_test": len(test),
        "mape_model": mape(y_true, y_pred),
        "mape_naive": mape(y_true, y_naive),
        "mape_seasonal": mape(y_true, y_seasonal),
        "mape_det": mape(y_true, y_det),
        "mae_model": mae(y_true, y_pred),
        "mae_naive": mae(y_true, y_naive),
        "mae_seasonal": mae(y_true, y_seasonal),
        "mae_det": mae(y_true, y_det),
        "bias_model": (y_pred - y_true).mean() / y_true.mean(),
    }
    best_mape = min(results["mape_naive"], results["mape_seasonal"], results["mape_det"])
    results["best_baseline_mape"] = best_mape
    results["venceu_best"] = results["mape_model"] < best_mape
    return results


def main():
    print("=" * 60)
    print("  ARC.BI.Sales.Insight - VALIDACAO v3 (alvo em razao)")
    print("=" * 60)

    monthly = load_monthly(DATA_PATH)
    featured = add_features(monthly)
    feature_cols = [c for c in featured.columns if c not in ("mes", "receita")]

    print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")
    print("Alvo: y_t / lag_1 (razao, nao nivel)")

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    results = []
    for fold, (tri, tei) in enumerate(tscv.split(featured), 1):
        train = featured.iloc[tri]
        test = featured.iloc[tei]
        r = evaluate_fold(train, test, feature_cols)
        r["fold"] = fold
        results.append(r)

        print(f"\nDobra {fold}: train={r['n_train']}, test={r['n_test']}")
        print(f"   MAPE — LGBM: {r['mape_model']:.2f}%  |  Naive: {r['mape_naive']:.2f}%  |  Seasonal: {r['mape_seasonal']:.2f}%  |  Deterministico: {r['mape_det']:.2f}%")
        print(f"   Melhor baseline: {min(r['mape_naive'], r['mape_seasonal'], r['mape_det']):.2f}%")
        print(f"   Venceu? {'SIM' if r['venceu_best'] else 'NAO'}  |  Vies: {r['bias_model']:+.3f}")

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "validate_v3_results.csv", index=False)

    print("\n" + "=" * 60)
    print("  RESUMO FINAL (v3)")
    print("=" * 60)
    print(f"   MAPE LightGBM (razao):    {df['mape_model'].mean():.2f}%")
    print(f"   MAPE Naive simples:       {df['mape_naive'].mean():.2f}%")
    print(f"   MAPE Seasonal Naive:      {df['mape_seasonal'].mean():.2f}%")
    print(f"   MAPE Baseline determin.:  {df['mape_det'].mean():.2f}%")

    venceu = df["venceu_best"].sum()
    print(f"\n   Vitorias vs MELHOR baseline: {venceu}/{N_SPLITS}")
    print(f"\n   Vies por dobra: {[round(b, 3) for b in df['bias_model'].tolist()]}")

    print("\n" + "=" * 60)
    print("  VEREDITO (pre-registrado)")
    print("=" * 60)
    if venceu >= 3:
        print("   RESULTADO: LightGBM venceu em >=3/5 dobras.")
        print("   -> Mudanca de alvo FUNCIONOU.")
    else:
        print("   RESULTADO: LightGBM NAO venceu o melhor baseline.")
        print("   -> Modelo simples basta.")
        print("   -> Resultado honesto e publicavel.")

    print(f"\n   Resultados salvos: {OUTPUT_DIR / 'validate_v3_results.csv'}")


if __name__ == "__main__":
    main()
