"""
ARC.BI.Sales.Insight — Processamento Olist (dados reais).
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OLIST_DIR = ROOT / "data" / "raw" / "olist"
OUTPUT = ROOT / "data" / "processed" / "olist_semanal.csv"


def main():
    print("=" * 60)
    print("  ARC.BI.Sales.Insight — Processamento Olist")
    print("=" * 60)

    print("\nCarregando orders...")
    orders = pd.read_csv(OLIST_DIR / "olist_orders_dataset.csv")
    print(f"   {len(orders):,} pedidos")

    print("Carregando order_items...")
    items = pd.read_csv(OLIST_DIR / "olist_order_items_dataset.csv")
    print(f"   {len(items):,} itens")

    print("\nJuntando orders + order_items...")
    df = items.merge(orders, on="order_id", how="inner")
    print(f"   {len(df):,} linhas")

    print("\nFiltrando apenas pedidos entregues...")
    df = df[df["order_status"] == "delivered"].copy()
    print(f"   {len(df):,} linhas")

    print("\nConvertendo data...")
    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"], errors="coerce"
    )
    df = df.dropna(subset=["order_purchase_timestamp"])
    print(f"   Periodo: {df['order_purchase_timestamp'].min().date()} a {df['order_purchase_timestamp'].max().date()}")

    print("\nAgregando por semana...")
    df["semana"] = df["order_purchase_timestamp"].dt.to_period("W").dt.start_time

    receita_semanal = (
        df.groupby("semana")
        .agg(
            receita_total=("price", "sum"),
            n_pedidos=("order_id", "nunique"),
            n_itens=("order_item_id", "count"),
        )
        .reset_index()
    )

    print("\nFiltrando semanas incompletas (< 10 pedidos)...")
    receita_semanal = receita_semanal[receita_semanal["n_pedidos"] >= 10]
    print(f"   {len(receita_semanal)} semanas uteis")

    receita_semanal["ano"] = receita_semanal["semana"].dt.year
    receita_semanal["mes"] = receita_semanal["semana"].dt.month
    receita_semanal["semana_do_ano"] = receita_semanal["semana"].dt.isocalendar().week

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    receita_semanal.to_csv(OUTPUT, index=False)
    print(f"\nOK Arquivo salvo: {OUTPUT}")
    print(f"   {len(receita_semanal)} semanas")


if __name__ == "__main__":
    main()
