"""
ARC.BI.Sales.Insight — Validacao v6 (alvo em razao, Olist).

Variante da v3 aplicada ao Olist.
Alvo: y_t / lag_1
Previsao: lag_1 * ratio_prevista
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

FEATURES = ["lag_1", "lag_2", "lag_3", "lag_4",
            "roll_mean_4", "roll_mean_8", "roll_std_4"]
assert not {"semana_do_ano", "mes_num"} & set(FEATURES)

RANDOM_STATE = 42
N_SPLITS = 5

LGB_PARAMS = dict(
    n_estimators=200, learning_rate=0.08, num_leaves=15, max_depth=4,
    min_child_samples=3, subsample=1.0, colsample_bytree=1.0,
    random_state=RANDOM_STATE, verbosity=-1,
)


def load_weekly(path):
    df = pd.read_csv(path, parse_dates=["semana"])
    mediana = df["n_pedidos"].median()
    df = df[df["n_pedidos"] >= mediana * 0.3].copy()
    return df.sort_values("semana").reset_index(drop=True)


def add_lag_features(df):
    out = df.copy()
    for lag in [1, 2, 3, 4]:
        out[f"lag_{lag}"] = out["receita_total"].shift(lag)
    out["roll_mean_4"] = out["receita_total"].shift(1).rolling(4).mean()
    out["roll_mean_8"] = out["receita_total"].shift(1).rolling(8).mean()
    out["roll_std_4"] = out["receita_total"].shift(1).rolling(4).std()
    return out.dropna().reset_index(drop=True)


def wape(y_true, y_pred):
    return np.sum(np.abs(y_pred - y_true)) / np.sum(np.abs(y_true)) * 100


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100


def evaluate_fold(train, test, feature_cols):
    train = train.copy()
    test = test.copy()
    
    # ALVO EM RAZAO
    train["ratio"] = train["receita_total"] / train["lag_1"]
    X_train = train[feature_cols]
    y_train_ratio = train["ratio"]

    X_test = test[feature_cols]
    y_test = test["receita_total"]

    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X_train, y_train_ratio)
    
    # Reconstruir nivel
    ratio_pred = model.predict(X_test)
    y_pred = test["lag_1"].values * ratio_pred

    y_naive1 = test["lag_1"].values
    y_true = y_test.values

    return {
        "n_train": len(train), "n_test": len(test),
        "mape_model": mape(y_true, y_pred),
        "mape_naive1": mape(y_true, y_naive1),
        "wape_model": wape(y_true, y_pred),
        "wape_naive1": wape(y_true, y_naive1),
    }


def main():
    print("=" * 70)
    print("  ARC.BI.Sales.Insight - VALIDACAO v6 (RAZAO, Olist)")
    print("=" * 70)
    print(f"  Alvo: y_t / lag_1 (razao)")
    print(f"  Features: {FEATURES}")

    df = load_weekly(DATA_PATH)
    df_feat = add_lag_features(df)

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    results = []
    for fold, (tri, tei) in enumerate(tscv.split(df_feat), 1):
        train = df_feat.iloc[tri]
        test = df_feat.iloc[tei]
        r = evaluate_fold(train, test, FEATURES)
        r["fold"] = fold
        results.append(r)

        print(f"\nDobra {fold}: train={r['n_train']}, test={r['n_test']}")
        print(f"   MAPE - LGBM: {r['mape_model']:.2f}%  |  Naive1: {r['mape_naive1']:.2f}%")
        print(f"   WAPE - LGBM: {r['wape_model']:.2f}%  |  Naive1: {r['wape_naive1']:.2f}%")

    df_r = pd.DataFrame(results)
    df_r.to_csv(OUTPUT_DIR / "validate_v6_ratio_results.csv", index=False)

    print("\n" + "=" * 70)
    print("  RESUMO FINAL (v6)")
    print("=" * 70)
    print(f"   MAPE LGBM:        {df_r['mape_model'].mean():.2f}%")
    print(f"   MAPE Naive1:      {df_r['mape_naive1'].mean():.2f}%")
    print(f"   WAPE LGBM:        {df_r['wape_model'].mean():.2f}%")
    print(f"   WAPE Naive1:      {df_r['wape_naive1'].mean():.2f}%")


if __name__ == "__main__":
    main()
