"""
ARC.BI.Sales.Insight — Validacao v4 (dados reais Olist, semanal).
"""
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "olist_semanal.csv"
OUTPUT_DIR = ROOT / "outputs"

N_LAGS = 4
RANDOM_STATE = 42
N_SPLITS = 5


def load_weekly(path):
    print("Carregando Olist semanal...")
    df = pd.read_csv(path, parse_dates=["semana"])
    n_original = len(df)
    # Remover semanas truncadas (menos de 30% da mediana de pedidos)
    mediana = df["n_pedidos"].median()
    df = df[df["n_pedidos"] >= mediana * 0.3].copy()
    print(f"   {len(df)} semanas uteis (removidas {n_original - len(df)} truncadas)")
    print(f"   Periodo: {df['semana'].min().date()} a {df['semana'].max().date()}")
    return df.sort_values("semana").reset_index(drop=True)


def add_features(df, n_lags=N_LAGS):
    out = df.copy()
    out["mes_num"] = out["semana"].dt.month
    out["semana_do_ano"] = out["semana"].dt.isocalendar().week.astype(int)
    for lag in range(1, n_lags + 1):
        out[f"lag_{lag}"] = out["receita_total"].shift(lag)
    out["roll_mean_4"] = out["receita_total"].shift(1).rolling(4).mean()
    out["roll_mean_8"] = out["receita_total"].shift(1).rolling(8).mean()
    out["roll_std_4"] = out["receita_total"].shift(1).rolling(4).std()
    return out.dropna().reset_index(drop=True)


def naive_forecast(history, n_steps):
    return np.array([history["receita_total"].iloc[-1]] * n_steps)


def baseline_deterministico(history, y_lag1_test, test_meses):
    """Naive × fator sazonal medio do treino (por mes)."""
    y_train = history["receita_total"].reset_index(drop=True)
    ratios = (y_train / y_train.shift(1)).dropna()
    months_train = history["semana"].dt.month.iloc[1:].reset_index(drop=True)
    factor = pd.Series(ratios.values, index=months_train).groupby(level=0).mean()
    factors = factor.reindex(test_meses).fillna(1.0).values
    return y_lag1_test * factors


def evaluate_fold(train, test, feature_cols):
    X_train, y_train = train[feature_cols], train["receita_total"]
    X_test, y_test = test[feature_cols], test["receita_total"]

    model = lgb.LGBMRegressor(
        n_estimators=200, learning_rate=0.08,
        num_leaves=15, max_depth=4,
        min_child_samples=3, subsample=1.0, colsample_bytree=1.0,
        random_state=RANDOM_STATE, verbosity=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_naive = naive_forecast(train, len(test))
    y_det = baseline_deterministico(train, test["lag_1"].values, test["semana"].dt.month.values)

    y_true = test["receita_total"].values

    def mape(yt, yp): return np.mean(np.abs((yt - yp) / np.clip(yt, 1, None))) * 100
    def mae(yt, yp): return np.mean(np.abs(yt - yp))

    results = {
        "n_train": len(train), "n_test": len(test),
        "mape_model": mape(y_true, y_pred),
        "mape_naive": mape(y_true, y_naive),
        "mape_det": mape(y_true, y_det),
        "mae_model": mae(y_true, y_pred),
        "mae_naive": mae(y_true, y_naive),
        "mae_det": mae(y_true, y_det),
        "bias_model": (y_pred - y_true).mean() / y_true.mean(),
    }
    best_mape = min(results["mape_naive"], results["mape_det"])
    results["best_baseline_mape"] = best_mape
    results["venceu_best"] = results["mape_model"] < best_mape
    return results


def main():
    print("=" * 60)
    print("  ARC.BI.Sales.Insight - VALIDACAO v4 (Olist semanal)")
    print("=" * 60)

    df = load_weekly(DATA_PATH)
    df_feat = add_features(df)
    feature_cols = [c for c in df_feat.columns if c not in ("semana", "receita_total", "n_pedidos", "n_itens", "ano", "mes")]

    print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    results = []
    for fold, (tri, tei) in enumerate(tscv.split(df_feat), 1):
        train = df_feat.iloc[tri]
        test = df_feat.iloc[tei]
        r = evaluate_fold(train, test, feature_cols)
        r["fold"] = fold
        results.append(r)

        print(f"\nDobra {fold}: train={r['n_train']}, test={r['n_test']}")
        print(f"   MAPE - LGBM: {r['mape_model']:.2f}%  |  Naive: {r['mape_naive']:.2f}%  |  Deterministico: {r['mape_det']:.2f}%")
        print(f"   Melhor baseline: {r['best_baseline_mape']:.2f}%  |  Venceu? {'SIM' if r['venceu_best'] else 'NAO'}")

    df_r = pd.DataFrame(results)
    df_r.to_csv(OUTPUT_DIR / "validate_v4_olist_results.csv", index=False)

    print("\n" + "=" * 60)
    print("  RESUMO FINAL (v4 - Olist semanal)")
    print("=" * 60)
    print(f"   MAPE LightGBM:   {df_r['mape_model'].mean():.2f}%")
    print(f"   MAPE Naive:      {df_r['mape_naive'].mean():.2f}%")
    print(f"   MAPE Determ.:    {df_r['mape_det'].mean():.2f}%")
    print(f"\n   Vitorias vs MELHOR baseline: {df_r['venceu_best'].sum()}/{N_SPLITS}")

    print("\n" + "=" * 60)
    print("  VEREDITO")
    print("=" * 60)
    v = df_r["venceu_best"].sum()
    if v >= 3:
        print("   LightGBM venceu o baseline em dados REAIS. Pronto para publicar.")
    else:
        print("   LightGBM NAO venceu o baseline em dados reais.")
        print("   Resultado honesto: modelo simples e mais robusto.")
        print("   Publicavel como achado cientifico.")


if __name__ == "__main__":
    main()
