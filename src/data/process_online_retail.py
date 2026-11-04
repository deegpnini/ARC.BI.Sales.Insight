"""ETL do Online Retail II (UCI) para o dashboard ARC.BI."""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "real" / "online_retail_II.xlsx"
OUT = ROOT / "data" / "processed" / "vendas_limpo.csv"

def main():
    print(f"Carregando {RAW}...")
    df1 = pd.read_excel(RAW, sheet_name="Year 2009-2010")
    df2 = pd.read_excel(RAW, sheet_name="Year 2010-2011")
    df = pd.concat([df1, df2], ignore_index=True)
    print(f"   {len(df):,} transacoes carregadas")

    df = df.rename(columns={
        "Invoice": "order_id",
        "StockCode": "product_id",
        "Description": "produto",
        "Quantity": "quantidade",
        "InvoiceDate": "data_venda",
        "Price": "unit_price",
        "Customer ID": "cliente",
        "Country": "country",
    })

    df["receita_total"] = df["quantidade"] * df["unit_price"]
    df = df[~df["order_id"].astype(str).str.startswith("C")]
    df = df.dropna(subset=["cliente", "receita_total"])
    df = df[df["receita_total"] > 0]
    
    # Mapeamento de categorias reais baseado em palavras-chave
    def mapear_categoria(desc):
        if pd.isna(desc): return "Outros"
        desc = str(desc).upper()
        if "LIGHT" in desc or "LAMP" in desc or "CANDLE" in desc: return "Iluminação"
        if "BAG" in desc or "PURSE" in desc: return "Bolsas"
        if "MUG" in desc or "CUP" in desc or "TEA" in desc: return "Cozinha/Mesa"
        if "CHRISTMAS" in desc or "XMAS" in desc: return "Natal"
        if "HEART" in desc or "LOVE" in desc: return "Decoração"
        if "TOY" in desc or "GAME" in desc or "PLAY" in desc: return "Brinquedos"
        if "CARD" in desc or "PAPER" in desc or "WRAP" in desc: return "Papelaria"
        if "JEWEL" in desc or "NECKLACE" in desc or "BRACELET" in desc: return "Joias"
        return "Outros"

    df["categoria"] = df["produto"].apply(mapear_categoria)

    df["regiao"] = df["country"]

    df.to_csv(OUT, index=False)
    print(f"OK: {OUT}")
    print(f"   Registros finais: {len(df):,}")

if __name__ == "__main__":
    main()
