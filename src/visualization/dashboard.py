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




def resolve_col(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    """Retorna o primeiro nome de coluna que existir (case-insensitive)."""
    lower_map = {c.lower().strip(): c for c in df.columns}
    for name in candidates:
        key = name.lower().strip()
        if key in lower_map:
            return lower_map[key]
    raise KeyError(
        f"Coluna '{label}' não encontrada. "
        f"Tentou: {candidates}. Colunas disponíveis: {list(df.columns)}"
    )


def agrupar_outros(
    df: pd.DataFrame,
    name_col: str,
    value_col: str,
    limiar: float = 0.05,
    max_fatias: int = 6,
) -> pd.DataFrame:
    """Mantém categorias com participação >= limiar (e no máximo max_fatias). O restante vira 'Outros'."""
    g = (
        df.groupby(name_col, as_index=False)[value_col]
        .sum()
        .sort_values(value_col, ascending=False)
    )
    total = g[value_col].sum()
    if total <= 0 or g.empty:
        return g
    g["pct"] = g[value_col] / total
    principais = g.loc[g["pct"] >= limiar].head(max_fatias).copy()
    mask_outros = ~g[name_col].isin(principais[name_col])
    outros_val = g.loc[mask_outros, value_col].sum()
    if outros_val > 0:
        principais = pd.concat(
            [
                principais[[name_col, value_col]],
                pd.DataFrame({name_col: ["Outros"], value_col: [outros_val]}),
            ],
            ignore_index=True,
        )
    else:
        principais = principais[[name_col, value_col]]
    return principais.sort_values(value_col, ascending=True)

@st.cache_data(show_spinner="Carregando dados...")
def load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()

    df = pd.read_csv(p, parse_dates=["data_venda"])

    # Normaliza schema (sintético OU Online Retail II)
    rename_map = {}
    cols_lower = {c.lower().strip(): c for c in df.columns}
    aliases = {
        "cliente": ["cliente", "customer id", "customer_id", "customerid", "id_cliente"],
        "produto": ["produto", "description", "stockcode", "stock_code"],
        "categoria": ["categoria", "category", "stockcode", "stock_code"],
        "regiao": ["regiao", "região", "country", "pais", "país"],
        "receita_total": ["receita_total", "receita", "sales", "amount"],
        "data_venda": ["data_venda", "invoicedate", "invoice_date", "date"],
    }
    for canon, opts in aliases.items():
        if canon in df.columns:
            continue
        for opt in opts:
            if opt in cols_lower:
                rename_map[cols_lower[opt]] = canon
                break
    if rename_map:
        df = df.rename(columns=rename_map)

    if "data_venda" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["data_venda"]):
        df["data_venda"] = pd.to_datetime(df["data_venda"], errors="coerce")
    df = df.dropna(subset=["data_venda", "receita_total"])
    df["receita_total"] = pd.to_numeric(df["receita_total"], errors="coerce").fillna(0)
    df["quantidade"] = pd.to_numeric(df.get("quantidade", 1), errors="coerce").fillna(0)

    if "cliente" in df.columns:
        df["cliente"] = df["cliente"].astype(str).str.strip()
        df = df[
            df["cliente"].notna()
            & (df["cliente"] != "")
            & (df["cliente"].str.lower() != "nan")
        ]
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
    st.plotly_chart(fig_line, width='stretch')

    # ----- Mix: região + categoria -----
    col_a, col_b = st.columns(2)

    with col_a:
        fig_pais = px.bar(
            f.groupby("regiao", as_index=False)["receita_total"].sum().sort_values("receita_total", ascending=True).tail(10),
            x="receita_total", y="regiao", orientation="h",
            title="Receita por país (Top 10)",
            color_discrete_sequence=["#38BDF8"]
        )
        fig_pais.update_layout(
            template="plotly_dark",
            height=400,
            margin=dict(t=50, b=20, l=20, r=20),
            xaxis_title="Receita (R$)", yaxis_title=""
        )
        st.plotly_chart(fig_pais, width="stretch")

    with col_b:
        # Top 8 categorias + Outros
        df_cat = f.groupby("categoria", as_index=False)["receita_total"].sum().sort_values("receita_total", ascending=False)
        if len(df_cat) > 8:
            top8 = df_cat.iloc[:8]
            outros_val = df_cat.iloc[8:]["receita_total"].sum()
            df_cat_final = pd.concat([top8, pd.DataFrame([{"categoria": "Outros", "receita_total": outros_val}])], ignore_index=True)
        else:
            df_cat_final = df_cat
        df_cat_final = df_cat_final.sort_values("receita_total", ascending=True)
        fig_cat = px.bar(
            df_cat_final, x="receita_total", y="categoria", orientation="h",
            title="Receita por categoria (Top 8 + Outros)",
            color_discrete_sequence=["#818CF8"]
        )
        fig_cat.update_layout(
            template="plotly_dark", height=400,
            margin=dict(t=50, b=20, l=20, r=20),
            xaxis_title="Receita (R$)", yaxis_title=""
        )
        st.plotly_chart(fig_cat, width="stretch")



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
            width='stretch',
            hide_index=True,
        )

    with col_d:
        st.subheader("Top 10 clientes")
        try:
            col_cli = resolve_col(
                f,
                ["cliente", "Customer ID", "customer_id", "CustomerID", "id_cliente"],
                "cliente",
            )
        except KeyError as e:
            st.warning(str(e))
            st.stop()

        top_cli = (
            f.groupby(col_cli, as_index=False)["receita_total"]
            .sum()
            .sort_values("receita_total", ascending=False)
            .head(10)
        )
        top_cli = top_cli.rename(columns={col_cli: "cliente"})

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
            width="stretch",
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
