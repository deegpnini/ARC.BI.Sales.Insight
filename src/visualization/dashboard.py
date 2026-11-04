import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="ARC.BI Sales Insight",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tema Neutro Escuro (Padrão 2026)
st.markdown("""
    <style>
    .stApp { background-color: #0F172A; color: #F8FAFC; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; color: #38BDF8; font-weight: 700; }
    .src-banner {
        background-color: #312E81; border: 1px solid #6366F1;
        color: #E0E7FF; padding: 1rem; border-radius: 0.5rem;
        margin-bottom: 1.5rem; font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "vendas_limpo.csv"

@st.cache_data
def load_data(path):
    if not Path(path).exists(): return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["data_venda"])
    # Normalização básica
    df["receita_total"] = pd.to_numeric(df["receita_total"], errors="coerce").fillna(0)
    df["quantidade"] = pd.to_numeric(df.get("quantidade", 1), errors="coerce").fillna(0)
    return df.sort_values("data_venda")

df = load_data(DATA_PATH)

st.title("📈 ARC.BI Sales Insight")
st.caption("Dashboard interativo de vendas — pipeline de BI")

st.markdown("""
<div class="src-banner">
    <strong>Fonte dos dados:</strong> <b>Online Retail II</b> (UCI, dados reais de e-commerce UK, 2009-2011, CC BY 4.0).
    <br><strong>Nota Metodológica:</strong> Categorias derivadas por regras de negócio sobre a coluna <code>Description</code>. Top 10 produtos traduzidos manualmente.
</div>
""", unsafe_allow_html=True)

if df.empty:
    st.error("Dataset não encontrado. Rode `python src/data/process_online_retail.py` primeiro.")
    st.stop()

# Filtros
st.sidebar.header("🎯 Filtros")
paises = st.sidebar.multiselect("Regiões", sorted(df["regiao"].unique()), default=sorted(df["regiao"].unique())[:5])
data_min, data_max = df["data_venda"].min().date(), df["data_venda"].max().date()
periodo = st.sidebar.date_input("Período", (data_min, data_max), min_value=data_min, max_value=data_max)

if len(periodo) == 2:
    f = df[(df["regiao"].isin(paises)) & (df["data_venda"].dt.date >= periodo[0]) & (df["data_venda"].dt.date <= periodo[1])]
else:
    f = df[df["regiao"].isin(paises)]

# KPIs
total_receita = f["receita_total"].sum()
total_pedidos = f["order_id"].nunique()
aov = total_receita / total_pedidos if total_pedidos > 0 else 0
qtd_itens = f["quantidade"].sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Receita Total", f"R$ {total_receita:,.2f}")
col2.metric("Total de Pedidos", f"{total_pedidos:,}")
col3.metric("Ticket Médio (AOV)", f"R$ {aov:,.2f}")
col4.metric("Unidades Vendidas", f"{qtd_itens:,.0f}")

# Narrativa
top_cat = f.groupby("categoria")["receita_total"].sum().sort_values(ascending=False)
top_cat_nome = top_cat.index[0] if not top_cat.empty else "N/A"
top_cat_pct = (top_cat.iloc[0] / total_receita * 100) if total_receita > 0 else 0
top_pais = f.groupby("regiao")["receita_total"].sum().sort_values(ascending=False)
top_pais_nome = top_pais.index[0] if not top_pais.empty else "N/A"
top_pais_pct = (top_pais.iloc[0] / total_receita * 100) if total_receita > 0 else 0

st.markdown(f"""
<div style="background-color: #1E293B; border-left: 4px solid #38BDF8; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1.5rem;">
    <b>🧠 Síntese Executiva:</b><br/>
    No período selecionado, a receita total foi de <b>R$ {total_receita:,.2f}</b>.
    A categoria líder foi <b>{top_cat_nome}</b> ({top_cat_pct:.1f}% do total).
    O mercado <b>{top_pais_nome}</b> concentra {top_pais_pct:.1f}% das vendas.
</div>
""", unsafe_allow_html=True)

# Gráficos
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Receita por País (Top 10)")
    df_pais = f.groupby("regiao", as_index=False)["receita_total"].sum().sort_values("receita_total", ascending=True).tail(10)
    fig_pais = px.bar(df_pais, x="receita_total", y="regiao", orientation="h", color_discrete_sequence=["#38BDF8"])
    fig_pais.update_layout(template="plotly_dark", height=400, margin=dict(t=30, b=20, l=20, r=20), xaxis_title="Receita (R$)", yaxis_title="")
    st.plotly_chart(fig_pais, width="stretch")

with col_b:
    st.subheader("Receita por Categoria (Top 8 + Outros)")
    df_cat = f.groupby("categoria", as_index=False)["receita_total"].sum().sort_values("receita_total", ascending=False)
    if len(df_cat) > 8:
        top8 = df_cat.iloc[:8]
        outros_val = df_cat.iloc[8:]["receita_total"].sum()
        df_cat_final = pd.concat([top8, pd.DataFrame([{"categoria": "Outros", "receita_total": outros_val}])], ignore_index=True)
    else:
        df_cat_final = df_cat
    df_cat_final = df_cat_final.sort_values("receita_total", ascending=True)
    fig_cat = px.bar(df_cat_final, x="receita_total", y="categoria", orientation="h", color_discrete_sequence=["#818CF8"])
    fig_cat.update_layout(template="plotly_dark", height=400, margin=dict(t=30, b=20, l=20, r=20), xaxis_title="Receita (R$)", yaxis_title="")
    st.plotly_chart(fig_cat, width="stretch")

# Evolução Mensal
st.subheader("Evolução Mensal da Receita")
df_mensal = f.set_index("data_venda").resample("ME")["receita_total"].sum().reset_index()
fig_linha = px.line(df_mensal, x="data_venda", y="receita_total", markers=True, color_discrete_sequence=["#34D399"])
fig_linha.update_layout(template="plotly_dark", height=350, margin=dict(t=30, b=20, l=20, r=20), xaxis_title="Mês", yaxis_title="Receita (R$)")
st.plotly_chart(fig_linha, width="stretch")

# Rankings
col_c, col_d = st.columns(2)

with col_c:
    st.subheader("Top 10 Produtos")
    # Agrupar por produto e produto_traduzido
    top_prod = f.groupby(["produto", "produto_traduzido"], as_index=False)["receita_total"].sum().sort_values("receita_total", ascending=False).head(10)
    top_prod["Receita"] = top_prod["receita_total"].map(lambda x: f"R$ {x:,.2f}")
    
    # Renomear colunas para exibição
    top_prod_exibicao = top_prod[["produto", "produto_traduzido", "Receita"]].copy()
    top_prod_exibicao.columns = ["Produto Original (EN)", "Tradução (PT-BR)", "Receita"]
    
    st.dataframe(top_prod_exibicao, width="stretch", hide_index=True)
    st.caption("*Nota: os nomes dos produtos estão no idioma original do dataset (inglês), com tradução ao lado.*")

with col_d:
    st.subheader("Top 10 Clientes")
    top_cli = f.groupby("cliente", as_index=False)["receita_total"].sum().sort_values("receita_total", ascending=False).head(10)
    top_cli["cliente"] = top_cli["cliente"].astype(str)
    if len(top_cli) >= 3:
        q = top_cli["receita_total"].quantile([0.33, 0.66])
        def seg(v):
            if v >= q.iloc[1]: return "Ouro"
            if v >= q.iloc[0]: return "Prata"
            return "Bronze"
        top_cli["Segmento"] = top_cli["receita_total"].map(seg)
    else:
        top_cli["Segmento"] = "—"
    top_cli["Gasto total"] = top_cli["receita_total"].map(lambda x: f"R$ {x:,.2f}")
    st.dataframe(top_cli[["cliente", "Gasto total", "Segmento"]].rename(columns={"cliente": "Cliente"}), width="stretch", hide_index=True)

st.caption("ARC.BI.Sales.Insight · portfólio · filtros reativos · KPIs · Plotly · não usar estes números como performance de modelo ML.")
