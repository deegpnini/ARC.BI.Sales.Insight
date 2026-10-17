"""
ARC.BI.Sales.Insight — Forecast com LightGBM.
"""
import warnings
from pathlib import Path
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "vendas_limpo.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_LAGS = 3
HORIZON = 3
TEST_MONTHS = 6
RANDOM_STATE = 42


def load_monthly(path: Path) -> pd.DataFrame:
    print("Carregando dataset limpo...")
    df = pd.read_csv(path, parse_dates=["data_venda"])
    print(f"   {len(df):,} registros carregados")
    monthly = (
        df.set_index("data_venda")
        .resample("MS")["receita_total"].sum()
        .rename("receita").to_frame().reset_index()
        .rename(columns={"data_venda": "mes"})
    )
    monthly = monthly.sort_values("mes").reset_index(drop=True)
    print(f"   {len(monthly)} meses agregados "
          f"({monthly['mes'].min().date()} a {monthly['mes'].max().date()})")
    return monthly


def add_features(df: pd.DataFrame, n_lags: int = N_LAGS) -> pd.DataFrame:
    out = df.copy()
    out["mes_num"] = out["mes"].dt.month
    out["ano"] = out["mes"].dt.year
    out["trimestre"] = out["mes"].dt.quarter
    for lag in range(1, n_lags + 1):
        out[f"lag_{lag}"] = out["receita"].shift(lag)
    out["roll_mean_3"] = out["receita"].shift(1).rolling(3).mean()
    out["roll_mean_6"] = out["receita"].shift(1).rolling(6).mean()
    out["roll_std_3"] = out["receita"].shift(1).rolling(3).std()
    return out


def train_test_split_temporal(df: pd.DataFrame, test_months: int = TEST_MONTHS):
    df = df.dropna().reset_index(drop=True)
    if len(df) <= test_months + 3:
        raise ValueError(f"Serie mensal curta demais ({len(df)} meses).")
    split = len(df) - test_months
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()
    feature_cols = [c for c in df.columns if c not in ("mes", "receita")]
    X_train, y_train = train[feature_cols], train["receita"]
    X_test, y_test = test[feature_cols], test["receita"]
    return train, test, X_train, y_train, X_test, y_test, feature_cols


def train_model(X_train, y_train, X_test, y_test):
    model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.08,
        num_leaves=15,
        max_depth=4,
        min_child_samples=3,
        min_split_gain=0.0,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=42,
        verbosity=-1,
    )
    model.fit(X_train, y_train)
    return model


def metrics(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE_%": mape}


def plot_forecast(train, test, y_pred, path: Path):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(train["mes"], train["receita"], label="Treino (real)", color="#94A3B8", linewidth=1.8)
    ax.plot(test["mes"], test["receita"], label="Teste (real)", color="#C084FC", linewidth=2.2)
    ax.plot(test["mes"], y_pred, label="Previsao (LightGBM)", color="#22D3EE", linewidth=2.2, linestyle="--")
    ax.set_title("ARC.BI Sales Insight - Previsao de Receita Mensal", fontsize=13, pad=12)
    ax.set_xlabel("Mes")
    ax.set_ylabel("Receita (R$)")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"   Grafico salvo: {path.name}")


def recursive_forecast(model, history: pd.DataFrame, feature_cols, horizon: int = HORIZON):
    hist = history.copy()
    preds = []
    for _ in range(horizon):
        row = hist.iloc[[-1]].copy()
        next_mes = (row["mes"].iloc[0] + pd.offsets.MonthBegin(1))
        row["mes"] = next_mes
        row["mes_num"] = next_mes.month
        row["ano"] = next_mes.year
        row["trimestre"] = next_mes.quarter
        serie = hist["receita"].tolist()
        for lag in range(1, N_LAGS + 1):
            row[f"lag_{lag}"] = serie[-lag] if len(serie) >= lag else np.nan
        recent = pd.Series(serie)
        row["roll_mean_3"] = recent.iloc[-3:].mean() if len(recent) >= 3 else recent.mean()
        row["roll_mean_6"] = recent.iloc[-6:].mean() if len(recent) >= 6 else recent.mean()
        row["roll_std_3"] = recent.iloc[-3:].std() if len(recent) >= 3 else 0.0
        X_next = row[feature_cols]
        y_hat = float(model.predict(X_next)[0])
        y_hat = max(y_hat, 0)
        new = row.copy()
        new["receita"] = y_hat
        hist = pd.concat([hist, new], ignore_index=True)
        preds.append({"mes": next_mes, "receita_prevista": y_hat})
    return pd.DataFrame(preds)


def main():
    print("=" * 50)
    print("  ARC.BI.Sales.Insight - Forecast (LightGBM)")
    print("=" * 50)
    monthly = load_monthly(DATA_PATH)
    featured = add_features(monthly)
    train, test, X_train, y_train, X_test, y_test, feature_cols = train_test_split_temporal(featured)
    print(f"\nTreino: {train['mes'].min().date()} a {train['mes'].max().date()} ({len(train)} meses)")
    print(f"Teste:  {test['mes'].min().date()} a {test['mes'].max().date()} ({len(test)} meses)")
    print(f"\nTreinando LightGBM com {len(feature_cols)} features...")
    model = train_model(X_train, y_train, X_test, y_test)
    y_pred = model.predict(X_test)
    m = metrics(y_test.values, y_pred)
    print("\nMetricas LightGBM (hold-out temporal):")
    print(f"   MAE : R$ {m['MAE']:,.2f}")
    print(f"   RMSE: R$ {m['RMSE']:,.2f}")
    print(f"   MAPE: {m['MAPE_%']:.2f}%")
    
    # Baseline naive (último valor)
    y_naive = np.repeat(train["receita"].iloc[-1], len(test))
    m_naive = metrics(y_test.values, y_naive)
    print("\nMetricas Baseline Naive (ultimo valor repetido):")
    print(f"   MAE : R$ {m_naive['MAE']:,.2f}")
    print(f"   RMSE: R$ {m_naive['RMSE']:,.2f}")
    print(f"   MAPE: {m_naive['MAPE_%']:.2f}%")
    
    # Comparação
    ganho = (m_naive['MAPE_%'] - m['MAPE_%']) / m_naive['MAPE_%'] * 100
    print(f"\nGanho vs Naive: {ganho:+.1f}%")
    print(f"   MAE : R$ {m['MAE']:,.2f}")
    print(f"   RMSE: R$ {m['RMSE']:,.2f}")
    print(f"   MAPE: {m['MAPE_%']:.2f}%")
    imp = (pd.DataFrame({"feature": feature_cols, "importance": model.booster_.feature_importance(importance_type="gain")})
           .sort_values("importance", ascending=False).head(10))
    print("\nTop features:")
    for _, r in imp.iterrows():
        print(f"   {r['feature']:<15} {r['importance']:.0f}")
    plot_path = OUTPUT_DIR / "07_forecast_lightgbm.png"
    plot_forecast(train, test, y_pred, plot_path)
    full = featured.dropna().reset_index(drop=True)
    future = recursive_forecast(model, full, feature_cols, horizon=HORIZON)
    future_path = OUTPUT_DIR / "forecast_proximos_meses.csv"
    future.to_csv(future_path, index=False)
    print(f"\nPrevisao {HORIZON} meses a frente:")
    for _, r in future.iterrows():
        print(f"   {r['mes'].strftime('%Y-%m')}: R$ {r['receita_prevista']:,.2f}")
    metrics_path = OUTPUT_DIR / "forecast_metrics.json"
    pd.Series(m).to_json(metrics_path, indent=2)
    print(f"   Metricas: {metrics_path.name}")
    print("\nOK Forecast concluido.")


if __name__ == "__main__":
    main()
