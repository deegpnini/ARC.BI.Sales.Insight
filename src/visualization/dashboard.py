"""
ARC.BI Sales Insight — Dashboard Streamlit
Dataset sintético de demonstração (EDA / produto visual).
A validação ML com Olist é outro módulo (validate_v5c / v6).
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# -----------------------------------------------------------------------------
# Config da página
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ARC.BI Sales Insight",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "vendas_limpo.csv"

# Tema visual (CSS leve)
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    .src-banner {
        background: #1e1b4b;
        border: 1px solid #7c3aed55;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        color: #c4b5fd;
        margin-bottom: 1rem;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Carregando dados...")
def load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()

    df = pd.read_csv(p, parse_dates=["data_venda"])
    df = df.dropna(subset=["data_venda", "receita_total"])
    df["receita_total"] = pd.to_numeric(df["receita_total"], errors="coerce").fillna(0)
    df["quantidade"] = pd.to_numeric(df.get("quantidade", 1), errors="coerce").fillna(0)
    return df.sort_values("data_venda")


def fmt_brl(x: float) -> str:
    return f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def main():
    st.title("ARC.BI Sales Insight")
    st.caption("Dashboard interativo de vendas — pipeline de BI (portfólio)")

    st.markdown(
        """
        <div class="src-banner">
        <strong>Fonte dos dados:</strong> dataset <em>sintético</em> gerado neste repositório
        (<code>vendas_limpo.csv</code>), para demonstração de ETL + EDA + produto visual.
        A validação temporal com dados reais (Olist) está em
        <code>src/models/validate_v5c.py</code> — não misturar os dois discursos.
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = load_data(str(DATA_PATH))
    if df.empty:
        st.error(
            f"Arquivo não encontrado: `{DATA_PATH}`\n\n"
            "Rode antes:\n"
            "```bash\n"
            "python src/data/generate.py   # ou generate_complex.py\n"
            "python src/data/clean.py      # ou clean_complex.py\n"
            "```"
        )
        st.stop()

    # ----- Sidebar / filtros -----
    st.sidebar.header("Filtros")

    min_d, max_d = df["data_venda"].min().date(), df["data_venda"].max().date()
    periodo = st.sidebar.date_input(
        "Período",
        value=(min_d, max_d),
        min_value=min_d,
        max_value=max_d,
    )
    if isinstance(periodo, tuple) and len(periodo) == 2:
        d0, d1 = periodo
    else:
        d0, d1 = min_d, max_d

    regioes = sorted(df["regiao"].dropna().unique())
    cats = sorted(df["categoria"].dropna().unique())

    sel_reg = st.sidebar.multiselect("Regiões", regioes, default=regioes)
    sel_cat = st.sidebar.multiselect("Categorias", cats, default=cats)

    mask = (
        (df["data_venda"].dt.date >= d0)
        & (df["data_venda"].dt.date <= d1)
        & (df["regiao"].isin(sel_reg))
        & (df["categoria"].isin(sel_cat))
    )
    f = df.loc[mask].copy()

    if f.empty:
        st.warning("Nenhum registro com os filtros atuais. Afrouxa região/categoria/período.")
        st.stop()

    st.sidebar.metric("Registros filtrados", f"{len(f):,}".replace(",", "."))
    st.sidebar.caption(f"Total na base: {len(df):,}".replace(",", "."))

    # ----- KPIs -----
    receita = f["receita_total"].sum()
    pedidos = len(f)
    ticket = receita / pedidos if pedidos else 0

    # Crescimento YoY simples (mesmo período do ano anterior dentro do filtro)
    f["ano"] = f["data_venda"].dt.year
    anos = sorted(f["ano"].unique())
    if len(anos) >= 2:
        y_last, y_prev = anos[-1], anos[-2]
        r_last = f.loc[f["ano"] == y_last, "receita_total"].sum()
        r_prev = f.loc[f["ano"] == y_prev, "receita_total"].sum()
        yoy = ((r_last / r_prev) - 1) * 100 if r_prev else 0.0
    else:
        yoy = 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Receita total", fmt_brl(receita))
    c2.metric("Ticket médio", fmt_brl(ticket))
    c3.metric("Pedidos (linhas)", f"{pedidos:,}".replace(",", "."))
    c4.metric("Crescimento YoY", f"{yoy:+.1f}%")

    st.divider()

    # ----- Série temporal -----
    mensal = (
        f.set_index("data_venda")
        .resample("MS")["receita_total"]
        .sum()
        .rename("Receita")
        .reset_index()
    )
    fig_line = px.line(
        mensal,
        x="data_venda",
        y="Receita",
        title="Evolução da receita (mensal)",
        markers=True,
    )
    fig_line.update_layout(
        template="plotly_dark",
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title="Mês",
        yaxis_title="Receita (R$)",
    )
    fig_line.update_traces(line_color="#a855f7")
    st.plotly_chart(fig_line, use_container_width=True)

    # ----- Mix: região + categoria -----
    col_a, col_b = st.columns(2)

    with col_a:
        por_reg = f.groupby("regiao", as_index=False)["receita_total"].sum()
        fig_pie = px.pie(
            por_reg,
            names="regiao",
            values="receita_total",
            title="Receita por região",
            hole=0.45,
        )
        fig_pie.update_layout(template="plotly_dark", height=380, margin=dict(t=50, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        por_cat = (
            f.groupby("categoria", as_index=False)["receita_total"]
            .sum()
            .sort_values("receita_total", ascending=True)
        )
        fig_bar = px.bar(
            por_cat,
            x="receita_total",
            y="categoria",
            orientation="h",
            title="Receita por categoria",
        )
        fig_bar.update_layout(
            template="plotly_dark",
            height=380,
            margin=dict(t=50, b=20),
            xaxis_title="Receita (R$)",
            yaxis_title="",
        )
        fig_bar.update_traces(marker_color="#8b5cf6")
        st.plotly_chart(fig_bar, use_container_width=True)

    # ----- Rankings -----
    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Top 10 produtos")
        top_prod = (
            f.groupby("produto", as_index=False)["receita_total"]
            .sum()
            .sort_values("receita_total", ascending=False)
            .head(10)
        )
        top_prod["Receita"] = top_prod["receita_total"].map(fmt_brl)
        st.dataframe(
            top_prod[["produto", "Receita"]].rename(columns={"produto": "Produto"}),
            use_container_width=True,
            hide_index=True,
        )

    with col_d:
        st.subheader("Top 10 clientes")
        top_cli = (
            f.groupby("cliente", as_index=False)["receita_total"]
            .sum()
            .sort_values("receita_total", ascending=False)
            .head(10)
        )
        # Segmento simples por tercil no top (só visual)
        if len(top_cli) >= 3:
            q = top_cli["receita_total"].quantile([0.33, 0.66])
            def seg(v):
                if v >= q.iloc[1]:
                    return "Ouro"
                if v >= q.iloc[0]:
                    return "Prata"
                return "Bronze"
            top_cli["Segmento"] = top_cli["receita_total"].map(seg)
        else:
            top_cli["Segmento"] = "—"
        top_cli["Gasto total"] = top_cli["receita_total"].map(fmt_brl)
        st.dataframe(
            top_cli[["cliente", "Gasto total", "Segmento"]].rename(
                columns={"cliente": "Cliente"}
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.caption(
        "ARC.BI.Sales.Insight · portfólio · "
        "filtros reativos · KPIs · Plotly · "
        "não usar estes números como performance de modelo ML"
    )


if __name__ == "__main__":
    main()
