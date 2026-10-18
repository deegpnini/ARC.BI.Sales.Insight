"""
ARC.BI.Sales.Insight — Validacao v5 (EXPLORATORIA).

Mudancas em relacao a v4:
- REMOVIDAS: semana_do_ano e mes_num (features de calendario)
  Justificativa: series curtas (~1.6 anos) nao tem dados suficientes
  para o modelo aprender sazonalidade anual. A arvore memoriza em vez de aprender.

- MANTIDAS: lags (1-4), roll_mean (4, 8), roll_std (4).

Esta e uma VARIANTE EXPLORATORIA. O resultado deve ser reportado como tal.
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
    mediana = df["n_pedidos"].median()
    df = df[df["n_pedidos"] >= mediana * 0.3].copy()
    print(f"   {len(df)} semanas uteis (removidas {n_original - len(df)} truncadas)")
    print(f"   Periodo: {df['semana'].min().date()} a {df['semana'].max().date()}")
    return df.sort_values("semana").reset_index(drop=True)


def add_features(df, n_lags=N_LAGS):
    out = df.copy()
    # REMOVIDAS: mes_num e semana_do_ano (features de calendario)
    for lag in range(1, n_lags + 1):
        out[f"lag_{lag}"] = out["receita_total"].shift(lag)
    out["roll_mean_4"] = out["receita_total"].shift(1).rolling(4).mean()
    out["roll_mean_8"] = out["receita_total"].shift(1).rolling(8).mean()
    out["roll_std_4"] = out["receita_total"].shift(1).rolling(4).std()
    return out.dropna().reset_index(drop=True)


def naive_forecast(history, n_steps):
    return np.array([history["receita_total"].iloc[-1]] * n_steps)


def baseline_deterministico(history, y_lag1_test, test_meses):
    """Naive x fator sazonal medio do treino (por mes).
    
    DEFINICAO: fator sazonal = media da razao (y_t / y_{t-1}) agrupada por mes.
    Previsao = lag_1 * fator do mes correspondente.
    """
    y_train = history["receita_total"].reset_index(drop=True)
    ratios = (y_train / y_train.shift(1)).dropna()
    months_train = history["semana"].dt.month.iloc[1:].reset_index(drop=True)
    factor = pd.Series(ratios.values, index=months_train).groupby(level=0).mean()
    factors = factor.reindex(test_meses).fillna(1.0).values
    return y_lag1_test * factors


def wape(y_true, y_pred):
    """WAPE = sum(|y_pred - y_true|) / sum(|y_true|) * 100."""
    return np.sum(np.abs(y_pred - y_true)) / np.sum(np.abs(y_true)) * 100


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

    def mape(yt, yp):
        return np.mean(np.abs((yt - yp) / np.clip(yt, 1, None))) * 100

    results = {
        "n_train": len(train), "n_test": len(test),
        "mape_model": mape(y_true, y_pred),
        "mape_naive": mape(y_true, y_naive),
        "mape_det": mape(y_true, y_det),
        "wape_model": wape(y_true, y_pred),
        "wape_naive": wape(y_true, y_naive),
        "wape_det": wape(y_true, y_det),
        "bias_model": (y_pred - y_true).mean() / y_true.mean(),
    }
    best_mape = min(results["mape_naive"], results["mape_det"])
    results["best_baseline_mape"] = best_mape
    results["venceu_best"] = results["mape_model"] < best_mape
    return results


def main():
    print("=" * 60)
    print("  ARC.BI.Sales.Insight - VALIDACAO v5 (EXPLORATORIA)")
    print("=" * 60)
    print("  Mudanca: removidas semana_do_ano e mes_num")

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
        print(f"   MAPE - LGBM: {r['mape_model']:.2f}%  |  Naive: {r['mape_naive']:.2f}%  |  Det: {r['mape_det']:.2f}%")
        print(f"   WAPE - LGBM: {r['wape_model']:.2f}%  |  Naive: {r['wape_naive']:.2f}%  |  Det: {r['wape_det']:.2f}%")
        print(f"   Venceu? {'SIM' if r['venceu_best'] else 'NAO'}")

    df_r = pd.DataFrame(results)
    df_r.to_csv(OUTPUT_DIR / "validate_v5_exploratoria_results.csv", index=False)

    print("\n" + "=" * 60)
    print("  RESUMO FINAL (v5 - EXPLORATORIA)")
    print("=" * 60)
    print(f"   MAPE LightGBM:   {df_r['mape_model'].mean():.2f}%")
    print(f"   MAPE Naive:      {df_r['mape_naive'].mean():.2f}%")
    print(f"   MAPE Det.:       {df_r['mape_det'].mean():.2f}%")
    print(f"\n   WAPE LightGBM:   {df_r['wape_model'].mean():.2f}%")
    print(f"   WAPE Naive:      {df_r['wape_naive'].mean():.2f}%")
    print(f"   WAPE Det.:       {df_r['wape_det'].mean():.2f}%")
    print(f"\n   Vitorias vs MELHOR baseline: {df_r['venceu_best'].sum()}/{N_SPLITS}")

    print("\n" + "=" * 60)
    print("  RESULTADO (sem interpretacao)")
    print("=" * 60)
    v = df_r["venceu_best"].sum()
    print(f"   LightGBM venceu em {v}/{N_SPLITS} dobras.")
    print(f"   Variante: v5 EXPLORATORIA.")


if __name__ == "__main__":
    main()
