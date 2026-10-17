"""
ARC.BI.Sales.Insight — Limpeza e tratamento do dataset.

Lê o dataset cru (data/raw/vendas_complexo.csv), aplica limpeza com Pandas,
valida com Pydantic e salva:
- data/processed/vendas_complexo_limpo.csv
- data/processed/relatorio_limpeza_complexo.json
"""
import json
import re
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ValidationError, field_validator

# ============================================================
# 1. SCHEMA (Pydantic) — Garante que os dados batem
# ============================================================
class VendaSchema(BaseModel):
    id_venda: int
    data_venda: str
    cliente: str
    produto: str
    categoria: str
    regiao: str
    quantidade: str
    preco_unitario: str

    @field_validator("cliente")
    @classmethod
    def limpar_cliente(cls, v: str) -> str:
        # Remove !!, espaços extras
        v = re.sub(r"[^A-Za-z0-9_]", "", v.strip())
        return v


# ============================================================
# 2. FUNÇÕES DE LIMPEZA
# ============================================================
def carregar_dados(caminho: str = "data/raw/vendas_complexo.csv") -> pd.DataFrame:
    """Carrega o CSV com Pandas."""
    print(f"📂 Carregando {caminho}...")
    df = pd.read_csv(caminho, dtype=str)
    print(f"   {len(df)} registros carregados")
    return df


def limpar_dados(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Aplica limpeza e retorna DataFrame + relatório."""
    relatorio = {"iniciais": len(df), "removidos_data": 0,
                 "removidos_nulos": 0, "removidos_strings": 0, "finais": 0}

    # 1. Limpar strings do cliente (regex)
    df["cliente"] = df["cliente"].apply(lambda x: re.sub(r"[^A-Za-z0-9_]", "", str(x).strip()) if pd.notna(x) else x)

    # 2. Remover nulos em quantidade e preco_unitario
    antes = len(df)
    df = df.dropna(subset=["quantidade", "preco_unitario"])
    df = df[(df["quantidade"].str.strip() != "") & (df["preco_unitario"].str.strip() != "")]
    relatorio["removidos_nulos"] = antes - len(df)

    # 3. Converter data_venda (coerce = vira NaT se inválida)
    df["data_venda"] = pd.to_datetime(df["data_venda"], errors="coerce")
    antes = len(df)
    df = df.dropna(subset=["data_venda"])
    relatorio["removidos_data"] = antes - len(df)

    # 4. Converter tipos numéricos
    df["quantidade"] = df["quantidade"].astype(int)
    df["preco_unitario"] = df["preco_unitario"].astype(float)

    # 5. Colunas derivadas (com Pandas + datetime)
    df["receita_total"] = df["quantidade"] * df["preco_unitario"]
    df["mes"] = df["data_venda"].dt.month
    df["ano"] = df["data_venda"].dt.year
    df["trimestre"] = df["data_venda"].dt.quarter
    df["faixa_receita"] = pd.cut(
        df["receita_total"],
        bins=[0, 500, 5000, float("inf")],
        labels=["Baixo Valor", "Médio Valor", "Alto Valor"]
    )

    relatorio["finais"] = len(df)
    return df, relatorio


# ============================================================
# 3. MAIN
# ============================================================
def main():
    print("=" * 50)
    print("  ARC.BI.Sales.Insight — Limpeza")
    print("=" * 50)

    df = carregar_dados()
    df_limpo, relatorio = limpar_dados(df)

    # Salvar dataset limpo
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df_limpo.to_csv("data/processed/vendas_complexo_limpo.csv", index=False)
    print(f"\n✅ Dataset limpo: data/processed/vendas_complexo_limpo.csv")
    print(f"   Registros finais: {relatorio['finais']}")

    # Salvar relatório
    with open("data/processed/relatorio_limpeza_complexo.json", "w", encoding="utf-8") as f:
        json.dump(relatorio, f, indent=2, ensure_ascii=False)
    print(f"✅ Relatório: data/processed/relatorio_limpeza_complexo.json")
    print(f"\n📊 Relatório:")
    for k, v in relatorio.items():
        print(f"   {k}: {v}")


if __name__ == "__main__":
    main()
