"""
ARC.BI.Sales.Insight — Validacao v5c (CORRIGIDA).

Correcoes em relacao a v5b:
1. Adicionado naive1 (baseline justo: test["lag_1"])
2. Melhor baseline compara com min(naive12, naive1, determinístico)
3. Adicionado assert que TRAVA features proibidas
4. Adicionado contador n_fallback (quantas semanas caem no fillna(1.0))
5. Hiperparmetros documentados no docstring
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

# LISTA EXPLICITA E TRAVADA
FEATURES = ["lag_1", "lag_2", "lag_3", "lag_4",
            "roll_mean_4", "roll_mean_8", "roll_std_4"]

# TRAVA: features de calendario NAO podem estar na lista
assert not {"semana_do_ano", "mes_num"} & set(FEATURES), "Features de calendario proibidas!"

RANDOM_STATE = 42
N_SPLITS = 5

# Hiperparametros (documentados)
LGB_PARAMS = dict(
    n_estimators=200,
    learning_rate=0.08,
    num_leaves=15,
    max_depth=4,
    min_child_samples=3,
    subsample=1.0,
    colsample_bytree=1.0,
    random_state=RANDOM_STATE,
    verbosity=-1,
)


def load_weekly(path):
    print("Carregando Olist semanal...")
    df = pd.read_csv(path, parse_dates=["semana"])
    n_original = len(df)
    mediana = df["n_pedidos"].median()
    df = df[df["n_pedidos"] >= mediana * 0.3].copy()
    print(f"   {len(df)} semanas uteis (removidas {n_original - len(df)} truncadas)")
    print(f"   Periodo: {df['semana'].min().date()} a {df['semana'].max().date()}")
    return df.sort_values("semana").reset_index(drop=True)


def add_lag_features(df):
    out = df.copy()
    for lag in [1, 2, 3, 4]:
        out[f"lag_{lag}"] = out["receita_total"].shift(lag)
    out["roll_mean_4"] = out["receita_total"].shift(1).rolling(4).mean()
    out["roll_mean_8"] = out["receita_total"].shift(1).rolling(8).mean()
    out["roll_std_4"] = out["receita_total"].shift(1).rolling(4).std()
    return out.dropna().reset_index(drop=True)


def naive_forecast_12(history, n_steps):
    """Naive de 12 passos: ultimo valor do treino, congelado."""
    return np.array([history["receita_total"].iloc[-1]] * n_steps)


def baseline_deterministico(history, y_lag1_test, test_meses):
    """Naive x fator sazonal medio do treino (por mes).
    
    DEFINICAO:
      fator_mes = media( y_t / y_{t-1} ) para cada mes
      previsao = lag_1 * fator_mes
    Se o mes nao aparece no treino, usa fator = 1.0 (ignora sazonalidade).
    """
    y_train = history["receita_total"].reset_index(drop=True)
    ratios = (y_train / y_train.shift(1)).dropna()
    months_train = history["semana"].dt.month.iloc[1:].reset_index(drop=True)
    factor = pd.Series(ratios.values, index=months_train).groupby(level=0).mean()
    factors = factor.reindex(test_meses).fillna(1.0).values
    return y_lag1_test * factors


def wape(y_true, y_pred):
    return np.sum(np.abs(y_pred - y_true)) / np.sum(np.abs(y_true)) * 100


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100


def evaluate_fold(train, test, feature_cols):
    X_train, y_train = train[feature_cols], train["receita_total"]
    X_test, y_test = test[feature_cols], test["receita_total"]

    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_naive12 = naive_forecast_12(train, len(test))
    y_naive1 = test["lag_1"].values
    meses_teste = test["semana"].dt.month.values
    y_det = baseline_deterministico(train, test["lag_1"].values, meses_teste)

    # Contar semanas que caem no fallback (mes nao visto no treino)
    n_fallback = int((~pd.Series(meses_teste).isin(set(train["semana"].dt.month))).sum())

    y_true = test["receita_total"].values

    results = {
        "n_train": len(train), "n_test": len(test),
        "mape_model": mape(y_true, y_pred),
        "mape_naive12": mape(y_true, y_naive12),
        "mape_naive1": mape(y_true, y_naive1),
        "mape_det": mape(y_true, y_det),
        "wape_model": wape(y_true, y_pred),
        "wape_naive12": wape(y_true, y_naive12),
        "wape_naive1": wape(y_true, y_naive1),
        "wape_det": wape(y_true, y_det),
        "n_fallback": n_fallback,
    }

    # MELHOR BASELINE da dobra: menor MAPE entre os 3
    best_mape = min(results["mape_naive12"], results["mape_naive1"], results["mape_det"])
    results["best_baseline_mape"] = best_mape
    results["venceu_best"] = results["mape_model"] < best_mape

    return results


def main():
    print("=" * 70)
    print("  ARC.BI.Sales.Insight - VALIDACAO v5c (CORRIGIDA)")
    print("=" * 70)
    print(f"  Features: {FEATURES}")
    print(f"  Assert: SEM semana_do_ano, SEM mes_num")
    print(f"  Naive12: 12 passos congelado | Naive1: test['lag_1'] (JUSTO)")
    print(f"  Hiperparametros: {LGB_PARAMS}")

    df = load_weekly(DATA_PATH)
    df_feat = add_lag_features(df)

    # TRAVA: todas as FEATURES existem?
    faltando = [f for f in FEATURES if f not in df_feat.columns]
    assert not faltando, f"Features ausentes: {faltando}"
    print(f"\nFeatures ({len(FEATURES)}): {FEATURES}")

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    results = []
    for fold, (tri, tei) in enumerate(tscv.split(df_feat), 1):
        train = df_feat.iloc[tri]
        test = df_feat.iloc[tei]
        r = evaluate_fold(train, test, FEATURES)
        r["fold"] = fold
        results.append(r)

        print(f"\nDobra {fold}: train={r['n_train']}, test={r['n_test']}")
        print(f"   MAPE LGBM: {r['mape_model']:.2f}% | Naive12: {r['mape_naive12']:.2f}% | Naive1: {r['mape_naive1']:.2f}% | Det: {r['mape_det']:.2f}%")
        print(f"   Melhor baseline: {r['best_baseline_mape']:.2f}% | Venceu? {'SIM' if r['venceu_best'] else 'NAO'}")
        print(f"   Fallback (mes nao visto): {r['n_fallback']}/{r['n_test']} semanas")

    df_r = pd.DataFrame(results)
    df_r.to_csv(OUTPUT_DIR / "validate_v5c_corrigida_results.csv", index=False)

    print("\n" + "=" * 70)
    print("  RESUMO FINAL (v5c)")
    print("=" * 70)
    print(f"   MAPE LGBM:        {df_r['mape_model'].mean():.2f}%")
    print(f"   MAPE Naive12:     {df_r['mape_naive12'].mean():.2f}%")
    print(f"   MAPE Naive1:      {df_r['mape_naive1'].mean():.2f}%")
    print(f"   MAPE Det:         {df_r['mape_det'].mean():.2f}%")
    print(f"\n   WAPE LGBM:        {df_r['wape_model'].mean():.2f}%")
    print(f"   WAPE Naive12:     {df_r['wape_naive12'].mean():.2f}%")
    print(f"   WAPE Naive1:      {df_r['wape_naive1'].mean():.2f}%")
    print(f"   WAPE Det:         {df_r['wape_det'].mean():.2f}%")
    print(f"\n   Vitorias vs MELHOR baseline: {df_r['venceu_best'].sum()}/{N_SPLITS}")
    print(f"   Fallback total: {df_r['n_fallback'].sum()} semanas caem em fator 1.0")


if __name__ == "__main__":
    main()
