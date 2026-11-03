"""
ARC.BI.Sales.Insight — Validacao rigorosa (v2, corrigida).

Correcoes aplicadas (auditoria Claude):
1. Veredito compara com o MELHOR baseline de cada dobra (naive OU seasonal)
2. Metrica renomeada para "MAE relativo ao melhor baseline" (nao mais MASE)
3. MAPE em vez de MAE medio (evita escala misturada)
4. Remover out/2026 (mes incompleto) do dataset
5. Remover features 'ano' e 'trimestre' (arvore nao extrapola tendencia)
6. Confirmar shift(1) em roll_* (documentado)
7. Horizonte 1 passo a frente (nao recursivo)
8. Dobras 3-5 como principais; 1-2 como exploratorias
9. Oráculo (piso de erro) sera adicionado em generate_oracle.py

Features anti-vazamento:
- lag_1 a lag_6: shift(1..6) — passado apenas
- roll_mean_3/6, roll_std_3: shift(1) + rolling — passado apenas
- mes_num: calendario (publico)
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
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_LAGS = 6
RANDOM_STATE = 42
N_SPLITS = 5
REMOVER_ULTIMO_MES = True  # out/2026 (incompleto)


def load_monthly(path: Path) -> pd.DataFrame:
    print("Carregando dataset...")
    df = pd.read_csv(path, parse_dates=["data_venda"])
    print(f"   {len(df):,} registros brutos")
    print(f"   {df['data_venda'].min().date()} a {df['data_venda'].max().date()}")

    if REMOVER_ULTIMO_MES:
        ultimo_mes = df["data_venda"].dt.to_period("M").max()
        n_ultimo_mes = (df["data_venda"].dt.to_period("M") == ultimo_mes).sum()
        n_medio = df.groupby(df["data_venda"].dt.to_period("M")).size().median()
        if n_ultimo_mes < n_medio * 0.7:
            print(f"   Removendo {ultimo_mes} (incompleto: {n_ultimo_mes} registros vs mediana {n_medio:.0f})")
            df = df[df["data_venda"].dt.to_period("M") < ultimo_mes]

    monthly = (
        df.set_index("data_venda")
        .resample("MS")["receita_total"].sum()
        .rename("receita").to_frame().reset_index()
        .rename(columns={"data_venda": "mes"})
    )
    monthly = monthly.sort_values("mes").reset_index(drop=True)
    print(f"   {len(monthly)} meses agregados")
    return monthly


def add_features(df: pd.DataFrame, n_lags: int = N_LAGS) -> pd.DataFrame:
    """
    Features SEM vazamento temporal.
    Todas usam shift(1) ou mais atras.
    REMOVIDO: 'ano' (arvore nao extrapola) e 'trimestre' (redundante com mes_num).
    """
    out = df.copy()
    out["mes_num"] = out["mes"].dt.month

    # Lags (todos shift >= 1)
    for lag in range(1, n_lags + 1):
        out[f"lag_{lag}"] = out["receita"].shift(lag)

    # Rolling com shift(1) ANTES (evita vazamento)
    out["roll_mean_3"] = out["receita"].shift(1).rolling(3).mean()
    out["roll_mean_6"] = out["receita"].shift(1).rolling(6).mean()
    out["roll_std_3"] = out["receita"].shift(1).rolling(3).std()

    return out.dropna().reset_index(drop=True)


def seasonal_naive_forecast(history: pd.DataFrame, test_dates: pd.Series) -> np.ndarray:
    """Seasonal naive: previsao do mes M = valor do mesmo mes M ano anterior."""
    history_indexed = history.set_index("mes")["receita"]
    preds = []
    for test_date in test_dates:
        target_date = test_date - pd.DateOffset(years=1)
        if target_date in history_indexed.index:
            preds.append(history_indexed.loc[target_date])
        elif len(history_indexed) >= 12:
            preds.append(history_indexed.iloc[-12])
        else:
            preds.append(history_indexed.mean())
    return np.array(preds)


def naive_forecast(history: pd.DataFrame, n_steps: int) -> np.ndarray:
    """Naive simples: repete o ultimo valor conhecido."""
    return np.array([history["receita"].iloc[-1]] * n_steps)


def mae_relative(y_true, y_pred, y_baseline) -> float:
    """
    MAE relativo ao baseline: MAE(model) / MAE(baseline).
    < 1: modelo venceu. = 1: empate. > 1: perdeu.
    (NAO e MASE — o MASE verdadeiro escala pelo naive no TREINO.)
    """
    mae_model = np.mean(np.abs(y_true - y_pred))
    mae_base = np.mean(np.abs(y_true - y_baseline))
    if mae_base == 0:
        return float("inf")
    return mae_model / mae_base


def evaluate_fold(train, test, feature_cols) -> dict:
    X_train, y_train = train[feature_cols], train["receita"]
    X_test, y_test = test[feature_cols], test["receita"]

    model = lgb.LGBMRegressor(
        n_estimators=200, learning_rate=0.08,
        num_leaves=15, max_depth=4,
        min_child_samples=3, subsample=1.0, colsample_bytree=1.0,
        random_state=RANDOM_STATE, verbosity=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_seasonal = seasonal_naive_forecast(train, test["mes"])
    y_naive = naive_forecast(train, len(test))

    mae_model = np.mean(np.abs(y_test.values - y_pred))
    mae_seasonal = np.mean(np.abs(y_test.values - y_seasonal))
    mae_naive = np.mean(np.abs(y_test.values - y_naive))

    # Melhor baseline de cada dobra
    best_baseline_mae = min(mae_naive, mae_seasonal)
    best_baseline_name = "naive" if mae_naive <= mae_seasonal else "seasonal"

    rel_seasonal = mae_relative(y_test.values, y_pred, y_seasonal)
    rel_naive = mae_relative(y_test.values, y_pred, y_naive)
    rel_best = mae_relative(y_test.values, y_pred,
                            y_naive if best_baseline_name == "naive" else y_seasonal)

    return {
        "n_train": len(train), "n_test": len(test),
        "mae_model": mae_model,
        "mae_seasonal": mae_seasonal,
        "mae_naive": mae_naive,
        "best_baseline": best_baseline_name,
        "best_baseline_mae": best_baseline_mae,
        "rel_seasonal": rel_seasonal,
        "rel_naive": rel_naive,
        "rel_best": rel_best,
        "mape_model": np.mean(np.abs((y_test.values - y_pred) / y_test.values)) * 100,
        "mape_seasonal": np.mean(np.abs((y_test.values - y_seasonal) / y_test.values)) * 100,
        "mape_naive": np.mean(np.abs((y_test.values - y_naive) / y_test.values)) * 100,
    }


def main():
    print("=" * 60)
    print("  ARC.BI.Sales.Insight - VALIDACAO (v2)")
    print("=" * 60)

    monthly = load_monthly(DATA_PATH)
    featured = add_features(monthly)
    feature_cols = [c for c in featured.columns if c not in ("mes", "receita")]

    print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")
    print("Features removidas: ano, trimestre (arvore nao extrapola)")

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    results = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(featured), 1):
        train = featured.iloc[train_idx]
        test = featured.iloc[test_idx]
        r = evaluate_fold(train, test, feature_cols)
        r["fold"] = fold
        results.append(r)

        marcador = "principal" if fold >= 3 else "exploratoria"
        print(f"\nDobra {fold} ({marcador}): train={r['n_train']}, test={r['n_test']}")
        print(f"   MAE: LGBM R$ {r['mae_model']:>12,.0f}  |  Seasonal R$ {r['mae_seasonal']:>12,.0f}  |  Naive R$ {r['mae_naive']:>12,.0f}")
        print(f"   Melhor baseline: {r['best_baseline']} (R$ {r['best_baseline_mae']:,.0f})")
        print(f"   rel_seasonal={r['rel_seasonal']:.3f}  rel_naive={r['rel_naive']:.3f}  rel_best={r['rel_best']:.3f}")

    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_DIR / "validate_results_v2.csv", index=False)

    print("\n" + "=" * 60)
    print("  RESUMO HONESTO (v2)")
    print("=" * 60)
    print(f"   MAPE LightGBM:        {df_results['mape_model'].mean():.2f}%")
    print(f"   MAPE Seasonal Naive:  {df_results['mape_seasonal'].mean():.2f}%")
    print(f"   MAPE Naive:           {df_results['mape_naive'].mean():.2f}%")
    print(f"\n   MAE relativo ao seasonal naive (mediana): {df_results['rel_seasonal'].median():.3f}")
    print(f"   MAE relativo ao naive simples (mediana):  {df_results['rel_naive'].median():.3f}")
    print(f"   MAE relativo ao MELHOR baseline (mediana):{df_results['rel_best'].median():.3f}")

    venceu_seasonal = (df_results["rel_seasonal"] < 1).sum()
    venceu_naive = (df_results["rel_naive"] < 1).sum()
    venceu_best = (df_results["rel_best"] < 1).sum()

    print(f"\n   Vitorias vs Seasonal Naive: {venceu_seasonal}/{N_SPLITS}")
    print(f"   Vitorias vs Naive simples:  {venceu_naive}/{N_SPLITS}")
    print(f"   Vitorias vs MELHOR baseline:{venceu_best}/{N_SPLITS}")

    # Veredito honesto
    print("\n" + "=" * 60)
    print("  VEREDITO HONESTO")
    print("=" * 60)
    if venceu_best >= 4:
        print("   LightGBM vence o MELHOR baseline na maioria das dobras.")
        print("   -> Pipeline validado. Modelo aprendeu padrao nao-trivial.")
    elif venceu_best >= 2:
        print("   Resultado MISTO: venceu em algumas, perdeu em outras.")
        print("   -> Nao ha evidencia forte de superioridade.")
        print("   -> Recomendado: testar em dados reais (Olist) antes de publicar.")
    else:
        print("   LightGBM NAO superou o melhor baseline.")
        print("   -> Resultado honesto e publicavel.")
        print("   -> Baseline forte (naive) e mais robusto em dados sinteticos.")
        print("   -> Proximo passo: Olist (dados reais).")

    print(f"\n   Resultados salvos: {OUTPUT_DIR / 'validate_results_v2.csv'}")


if __name__ == "__main__":
    main()
