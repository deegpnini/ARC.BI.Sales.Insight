"""ETL do Online Retail II (UCI) para o dashboard ARC.BI.

Dataset: transações de uma empresa de gifts UK (2009-2011).
Fonte: https://archive.ics.uci.edu/dataset/502/online+retail+ii
Licença: CC BY 4.0

Gera data/processed/vendas_limpo.csv no formato esperado pelo dashboard.
"""
import pandas as pd
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "real" / "online_retail_II.xlsx"
OUT = ROOT / "data" / "processed" / "vendas_limpo.csv"

# ==========================================
# DICIONÁRIO DE TRADUÇÃO (match parcial)
# ==========================================
TRADUCOES = {
    "POSTAGE": "Frete",
    "REGENCY CAKESTAND": "Suporte para Bolo (3 Andares)",
    "WHITE HANGING HEART": "Porta-Vela Coração Branco",
    "PAPER CRAFT": "Artesanato em Papel",
    "JUMBO BAG RED": "Sacola Jumbo Vermelha",
    "ASSORTED COLOUR BIRD": "Ornamento de Pássaro Colorido",
    "PARTY BUNTING": "Bandeirola de Festa",
    "MEDIUM CERAMIC TOP": "Pote de Cerâmica Médio",
    "PAPER CHAIN KIT": "Kit Corrente de Papel",
    "MANUAL": "Manual",
    "RABBIT NIGHT LIGHT": "Luz Noturna de Coelho",
    "DOLLY GIRL LUNCH BOX": "Marmita Menina",
    "SPACEBOY LUNCH BOX": "Marmita Menino Espacial",
    "RED TOADSTOOL": "Luz Noturna Cogumelo Vermelho",
    "ROUND SNACK BOXES": "Conjunto 4 Potes Lanche",
    "RED RETROSPOT ROUND": "Latas de Bolo Redondas",
    "SET OF 6 SPICE": "Conjunto 6 Latas de Tempero",
    "LUNCH BAG PINK": "Bolsa Térmica Rosa",
    "LUNCH BAG RED": "Bolsa Térmica Vermelha",
    "LUNCH BAG BLACK": "Bolsa Térmica Caveira Preta",
    "LUNCH BAG WOODLAND": "Bolsa Térmica Floresta",
    "LUNCH BAG": "Bolsa Térmica",
    "HEART T-LIGHT": "Vela Coração",
    "CHRISTMAS": "Natal",
    "LIGHT": "Luz",
}

def traduzir_produto(nome):
    """Traduz o produto usando match parcial (contém)."""
    if not isinstance(nome, str):
        return nome
    nome_upper = nome.upper().strip()
    
    # Match exato
    if nome_upper in TRADUCOES:
        return TRADUCOES[nome_upper]
    
    # Match parcial (contains)
    for chave, valor in TRADUCOES.items():
        if chave in nome_upper:
            return valor
    
    # Fallback: capitalizar
    return nome.strip().title()

def mapear_categoria(desc):
    """Deriva categoria da Description por prioridade."""
    if not isinstance(desc, str) or not desc.strip():
        return "Outros"
    d = desc.upper()
    if any(k in d for k in ["CHRISTMAS", "XMAS", "ADVENT", "SANTA", "REINDEER"]): return "Natal"
    if any(k in d for k in ["BAG", "TOTE", "PURSE", "BACKPACK", "LUNCH BAG", "SHOPPER"]): return "Bolsas"
    if any(k in d for k in ["MUG", "CUP", "TEA", "BOWL", "PLATE", "JUG", "CAKESTAND", "CUTLERY", "KITCHEN"]): return "Cozinha/Mesa"
    if any(k in d for k in ["LIGHT", "LAMP", "CANDLE", "HOLDER", "LANTERN", "T-LIGHT"]): return "Iluminacao"
    if any(k in d for k in ["CARD", "PAPER", "WRAP", "NOTEBOOK", "PEN", "PENCIL", "ENVELOPE"]): return "Papelaria"
    if any(k in d for k in ["BOX", "TIN", "STORAGE", "JAR", "BASKET", "CONTAINER"]): return "Armazenamento"
    if any(k in d for k in ["JEWEL", "NECKLACE", "BRACELET", "RING", "EARRING", "BROOCH"]): return "Joias"
    if any(k in d for k in ["TOY", "GAME", "PLAY", "DOLL", "TEDDY", "PUZZLE"]): return "Brinquedos"
    if any(k in d for k in ["HEART", "HANGING", "FRAME", "SIGN", "ORNAMENT", "DOORSTOP", "WALL ART"]): return "Decoracao"
    return "Outros"

def main():
    print(f"📂 Carregando {RAW}...")
    df1 = pd.read_excel(RAW, sheet_name="Year 2009-2010")
    df2 = pd.read_excel(RAW, sheet_name="Year 2010-2011")
    df = pd.concat([df1, df2], ignore_index=True)
    print(f"   {len(df):,} transacoes carregadas")

    # Renomear colunas
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

    # Receita
    df["receita_total"] = df["quantidade"] * df["unit_price"]

    # Limpar
    df = df[~df["order_id"].astype(str).str.startswith("C")]
    df = df.dropna(subset=["cliente", "receita_total"])
    df = df[df["receita_total"] > 0]

    # Traduzir produto (mantém original e cria a coluna traduzida)
    df["produto_traduzido"] = df["produto"].apply(traduzir_produto)
    df["categoria"] = df["produto"].apply(mapear_categoria)
    df["regiao"] = df["country"]

    # Salvar
    df.to_csv(OUT, index=False)
    print(f"✅ Dataset limpo: {OUT}")
    print(f"   Registros finais: {len(df):,}")
    print(f"   Colunas: {list(df.columns)}")

if __name__ == "__main__":
    main()
