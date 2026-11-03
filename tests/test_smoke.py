"""Smoke tests do ARC.BI.Sales.Insight.

Filosofia: Honestidade Radical em forma de código.
Estes testes validam o estado real dos artefatos e DOCUMENTAM
a derrota do LightGBM para o Naive1 (isso é intencional).
"""
import csv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
V6_CSV = OUTPUTS / "validate_v6_ratio_results.csv"
README = ROOT / "README.md"


# ---------- Sanidade do repositório ----------

def test_outputs_principais_existem():
    """Os 7 PNGs e os CSVs de validação essenciais existem."""
    esperados = [
        "01_sazonalidade.png",
        "02_top_produtos.png",
        "03_regioes.png",
        "04_correlacao.png",
        "05_segmentacao.png",
        "06_evolucao.png",
        "07_forecast_lightgbm.png",
        "validate_v6_ratio_results.csv",
    ]
    for nome in esperados:
        assert (OUTPUTS / nome).exists(), f"Faltando: {nome}"


# ---------- Contrato do CSV v6 ----------

def _ler_v6():
    with V6_CSV.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_v6_tem_5_dobras():
    linhas = _ler_v6()
    assert len(linhas) == 5, f"Esperava 5 dobras, achei {len(linhas)}"


def test_v6_colunas_esperadas():
    linhas = _ler_v6()
    colunas = set(linhas[0].keys())
    esperadas = {"n_train", "n_test", "mape_model", "mape_naive1",
                 "wape_model", "wape_naive1", "fold"}
    assert esperadas.issubset(colunas), f"Faltando colunas: {esperadas - colunas}"


# ---------- Honestidade Radical automatizada ----------

def test_naive1_vence_todas_as_dobras():
    """Em TODAS as 5 dobras, o Naive1 tem MAPE menor que o LightGBM.

    Este teste transforma a derrota do ML em prova automatizada.
    Se algum dia o LightGBM vencer, este teste falha e nos obriga
    a atualizar o README. Honestidade Radical.
    """
    linhas = _ler_v6()
    for linha in linhas:
        fold = linha["fold"]
        mape_model = float(linha["mape_model"])
        mape_naive = float(linha["mape_naive1"])
        assert mape_naive < mape_model, (
            f"Dobra {fold}: Naive1 ({mape_naive:.2f}) >= LightGBM ({mape_model:.2f})"
        )


def test_dobra3_e_anomala():
    """A dobra 3 tem MAPE do modelo > 30 (anomalia documentada)."""
    linhas = _ler_v6()
    dobra3 = next(l for l in linhas if l["fold"] == "3")
    assert float(dobra3["mape_model"]) > 30, (
        "Dobra 3 deixou de ser anômala — README precisa ser revisto."
    )


def test_mape_medio_v6_proximo_de_25():
    """Média do MAPE do modelo na v6 ~ 25,07% (tolerância 0,5)."""
    linhas = _ler_v6()
    media = sum(float(l["mape_model"]) for l in linhas) / len(linhas)
    assert abs(media - 25.07) < 0.5, f"Média do MAPE mudou: {media:.2f}"


# ---------- Pendência conhecida: README vs CSV ----------

@pytest.mark.xfail(
    reason="README diz 52,79% na dobra 3, CSV real diz 39,92%. "
           "Pendência a reconciliar com a outra Trinity.",
    strict=False,
)
def test_readme_dobra3_consistente_com_csv():
    """Detecta a discrepância entre README e CSV na dobra 3."""
    texto = README.read_text(encoding="utf-8")
    assert "52,79" in texto, "README não menciona 52,79 (número já corrigido?)"
    linhas = _ler_v6()
    dobra3 = next(l for l in linhas if l["fold"] == "3")
    real = float(dobra3["mape_model"])
    assert abs(real - 52.79) < 0.1, (
        f"README diz 52,79 mas CSV diz {real:.2f}. Discrepância documentada."
    )
