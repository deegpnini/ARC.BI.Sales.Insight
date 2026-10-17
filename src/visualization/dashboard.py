"""
ARC.BI.Sales.Insight — Dashboard Streamlit.

Roda com: streamlit run src/visualization/dashboard.py
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# Configuração da página
# ============================================================
st.set_page_config(
    page_title="ARC.BI Sales Insight",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS customizado (dark mode roxo)
st.markdown("""
<style>
    .main { background-color: #0a0a0f; }
    .stApp { background-color: #0a0a0f; }
    [data-testid="stMetricValue"] {
        color: #a855f7;
        font-size: 2rem;
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] {
        color: #c084fc;
        font-size: 0.9rem;
    }
    h1, h2, h3 { color: #e9d5ff !important; }
    .stSidebar { background-color: #12121a; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Carregamento com cache (evita reprocessar)
# ============================================================
@st.cache_data
def carregar_dados() -> pd.DataFrame:
    caminho = Path("data/processed/vendas_limpo.csv")
    if not caminho.exists():
        st.error("❌ Dataset não encontrado. Rode `python src/data/clean.py` primeiro.")
        st.stop()
    df = pd.read_csv(caminho, parse_dates=["data_venda"])
    return df


df = carregar_dados()

# ============================================================
# Título + Subtítulo
# ============================================================
st.title("📊 ARC.BI Sales Insight")
st.caption("Análise interativa do dataset sintético de vendas (2023-2025)")

# ============================================================
# Barra Lateral (Filtros)
# ============================================================
st.sidebar.header("🎛️ Filtros")

# Período
data_min = df["data_venda"].min().date()
data_max = df["data_venda"].max().date()
data_inicio, data_fim = st.sidebar.date_input(
    "Período",
    value=[data_min, data_max],
    min_value=data_min,
    max_value=data_max,
)

# Região
regioes_disponiveis = sorted(df["regiao"].unique())
regioes_sel = st.sidebar.multiselect(
    "Regiões", regioes_disponiveis, default=regioes_disponiveis
)

# Categoria
cats_disponiveis = sorted(df["categoria"].unique())
cats_sel = st.sidebar.multiselect(
    "Categorias", cats_disponiveis, default=cats_disponiveis
)

# ============================================================
# Aplicar filtros
# ============================================================
mask = (
    (df["data_venda"].dt.date >= data_inicio)
    & (df["data_venda"].dt.date <= data_fim)
    & (df["regiao"].isin(regioes_sel))
    & (df["categoria"].isin(cats_sel))
)
df_filt = df[mask]

# Se filtro esvaziar, avisa
if df_filt.empty:
    st.warning("Nenhum registro para os filtros selecionados.")
    st.stop()

# ============================================================
# KPIs (4 cards no topo)
# ============================================================
col1, col2, col3, col4 = st.columns(4)

receita_total = df_filt["receita_total"].sum()
ticket_medio = df_filt["receita_total"].mean()
n_pedidos = len(df_filt)

# Crescimento YoY (2024 vs 2023)
try:
    r2023 = df_filt[df_filt["ano"] == 2023]["receita_total"].sum()
    r2024 = df_filt[df_filt["ano"] == 2024]["receita_total"].sum()
    if r2023 > 0:
        cresc = ((r2024 - r2023) / r2023) * 100
        delta_str = f"{cresc:+.1f}% YoY"
    else:
        delta_str = "N/D"
except Exception:
    delta_str = "N/D"

with col1:
    st.metric("💰 Receita Total", f"R$ {receita_total:,.0f}".replace(",", "."))
with col2:
    st.metric("🎟️ Ticket Médio", f"R$ {ticket_medio:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
with col3:
    st.metric("📦 Total Pedidos", f"{n_pedidos:,}".replace(",", "."))
with col4:
    st.metric("📈 Crescimento", delta_str)

st.divider()

# ============================================================
# Gráfico 1: Evolução Temporal
# ============================================================
st.subheader("📈 Evolução da Receita no Tempo")

df_filt["periodo"] = df_filt["data_venda"].dt.to_period("M").astype(str)
serie = df_filt.groupby("periodo")["receita_total"].sum().reset_index()

fig1 = px.line(
    serie,
    x="periodo",
    y="receita_total",
    markers=True,
    color_discrete_sequence=["#a855f7"],
)
fig1.update_layout(
    template="plotly_dark",
    xaxis_title="Período",
    yaxis_title="Receita (R$)",
    paper_bgcolor="#0a0a0f",
    plot_bgcolor="#12121a",
    height=400,
)
fig1.update_traces(line=dict(width=3))
st.plotly_chart(fig1, use_container_width=True)

# ============================================================
# Gráficos 2 e 3: Categoria + Top Produtos
# ============================================================
col_esq, col_dir = st.columns(2)

with col_esq:
    st.subheader("📊 Receita por Categoria")
    cat_df = df_filt.groupby("categoria")["receita_total"].sum().reset_index()
    cat_df = cat_df.sort_values("receita_total", ascending=True)
    fig2 = px.bar(
        cat_df,
        x="receita_total",
        y="categoria",
        orientation="h",
        color="receita_total",
        color_continuous_scale="Purples",
    )
    fig2.update_layout(
        template="plotly_dark",
        xaxis_title="Receita (R$)",
        yaxis_title="",
        paper_bgcolor="#0a0a0f",
        plot_bgcolor="#12121a",
        showlegend=False,
        height=400,
    )
    st.plotly_chart(fig2, use_container_width=True)

with col_dir:
    st.subheader("🏆 Top 10 Produtos")
    top_df = (
        df_filt.groupby("produto")["receita_total"]
        .sum()
        .nlargest(10)
        .reset_index()
        .sort_values("receita_total", ascending=True)
    )
    fig3 = px.bar(
        top_df,
        x="receita_total",
        y="produto",
        orientation="h",
        color="receita_total",
        color_continuous_scale="Viridis",
    )
    fig3.update_layout(
        template="plotly_dark",
        xaxis_title="Receita (R$)",
        yaxis_title="",
        paper_bgcolor="#0a0a0f",
        plot_bgcolor="#12121a",
        showlegend=False,
        height=400,
    )
    st.plotly_chart(fig3, use_container_width=True)

# ============================================================
# Gráfico 4: Regiões (Donut)
# ============================================================
st.subheader("🗺️ Distribuição por Região")

reg_df = df_filt.groupby("regiao")["receita_total"].sum().reset_index()
fig4 = px.pie(
    reg_df,
    values="receita_total",
    names="regiao",
    hole=0.4,
    color_discrete_sequence=px.colors.sequential.Purples_r,
)
fig4.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0a0a0f",
    plot_bgcolor="#12121a",
    height=400,
)
st.plotly_chart(fig4, use_container_width=True)

# ============================================================
# Segmentação de Clientes (Bronze/Prata/Ouro)
# ============================================================
st.subheader("👑 Top 10 Clientes (por gasto total)")

clientes = df_filt.groupby("cliente")["receita_total"].sum().reset_index()
clientes["segmento"] = pd.cut(
    clientes["receita_total"],
    bins=[0, 5000, 15000, float("inf")],
    labels=["🥉 Bronze", "🥈 Prata", "🥇 Ouro"],
)
clientes = clientes.sort_values("receita_total", ascending=False).head(10)

st.dataframe(
    clientes,
    use_container_width=True,
    column_config={
        "cliente": "Cliente",
        "receita_total": st.column_config.NumberColumn("Gasto Total", format="R$ %.2f"),
        "segmento": "Segmento",
    },
    hide_index=True,
)

st.caption("📌 ARC.BI Sales Insight — Data de hoje: " + pd.Timestamp.now().strftime("%d/%m/%Y %H:%M"))
