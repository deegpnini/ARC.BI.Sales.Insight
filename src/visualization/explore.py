"""
ARC.BI.Sales.Insight — Análise Exploratória (EDA).

Gera 6 gráficos profissionais a partir do dataset limpo.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="darkgrid", palette="viridis")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["figure.dpi"] = 100


def carregar() -> pd.DataFrame:
    print("Carregando dataset limpo...")
    df = pd.read_csv("data/processed/vendas_limpo.csv", parse_dates=["data_venda"])
    print(f"   {len(df)} registros carregados")
    return df


def grafico_sazonalidade(df, out):
    mensal = df.groupby("mes")["receita_total"].sum() / 1000
    fig, ax = plt.subplots()
    mensal.plot(kind="bar", ax=ax, color="#a855f7", edgecolor="white")
    ax.set_title("Receita Total por Mes (Sazonalidade)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Mes")
    ax.set_ylabel("Receita (R$ milhares)")
    ax.set_xticklabels(["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
                        "Jul", "Ago", "Set", "Out", "Nov", "Dez"])
    plt.tight_layout()
    plt.savefig(out / "01_sazonalidade.png")
    plt.close()
    print("   01_sazonalidade.png")


def grafico_top_produtos(df, out):
    top = df.groupby("produto")["receita_total"].sum().nlargest(10) / 1000
    fig, ax = plt.subplots()
    top.sort_values().plot(kind="barh", ax=ax, color="#c084fc", edgecolor="white")
    ax.set_title("Top 10 Produtos por Receita", fontsize=14, fontweight="bold")
    ax.set_xlabel("Receita (R$ milhares)")
    plt.tight_layout()
    plt.savefig(out / "02_top_produtos.png")
    plt.close()
    print("   02_top_produtos.png")


def grafico_regioes(df, out):
    reg = df.groupby("regiao")["receita_total"].sum() / 1000
    fig, ax = plt.subplots()
    ax.pie(reg.values, labels=reg.index, autopct="%1.1f%%",
           colors=sns.color_palette("viridis", len(reg)), startangle=90)
    ax.set_title("Receita por Regiao", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out / "03_regioes.png")
    plt.close()
    print("   03_regioes.png")


def grafico_correlacao(df, out):
    cols = ["quantidade", "preco_unitario", "receita_total", "mes", "ano"]
    corr = df[cols].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, ax=ax)
    ax.set_title("Matriz de Correlacao", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out / "04_correlacao.png")
    plt.close()
    print("   04_correlacao.png")


def grafico_segmentacao(df, out):
    clientes = df.groupby("cliente")["receita_total"].sum()
    faixas = pd.cut(clientes, bins=[0, 5000, 15000, float("inf")],
                    labels=["Bronze", "Prata", "Ouro"])
    contagem = faixas.value_counts()
    fig, ax = plt.subplots()
    contagem.plot(kind="bar", ax=ax,
                  color=["#60a5fa", "#4ade80", "#a855f7"], edgecolor="white")
    ax.set_title("Distribuicao de Clientes por Segmento", fontsize=14, fontweight="bold")
    ax.set_xlabel("Segmento")
    ax.set_ylabel("Numero de clientes")
    ax.set_xticklabels(contagem.index, rotation=0)
    plt.tight_layout()
    plt.savefig(out / "05_segmentacao.png")
    plt.close()
    print("   05_segmentacao.png")


def grafico_evolucao(df, out):
    df["periodo"] = df["data_venda"].dt.to_period("M").astype(str)
    serie = df.groupby("periodo")["receita_total"].sum() / 1000
    fig, ax = plt.subplots()
    serie.plot(kind="line", ax=ax, color="#a855f7", linewidth=2, marker="o")
    ax.set_title("Evolucao Mensal da Receita (2023-2025)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Mes/Ano")
    ax.set_ylabel("Receita (R$ milhares)")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(out / "06_evolucao.png")
    plt.close()
    print("   06_evolucao.png")


def main():
    print("=" * 50)
    print("  ARC.BI.Sales.Insight — EDA")
    print("=" * 50)

    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)

    df = carregar()

    grafico_sazonalidade(df, out)
    grafico_top_produtos(df, out)
    grafico_regioes(df, out)
    grafico_correlacao(df, out)
    grafico_segmentacao(df, out)
    grafico_evolucao(df, out)

    print("")
    print("6 graficos salvos em outputs/")


if __name__ == "__main__":
    main()
