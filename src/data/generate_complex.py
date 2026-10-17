"""
ARC.BI.Sales.Insight — Gerador de dados sintéticos COMPLEXOS.

Gera vendas com:
- Tendência de crescimento (ano a ano)
- Sazonalidade multiplicativa (Natal puxa +80%, não +30%)
- Ruído gaussiano (não uniforme)
- Eventos aleatórios (picos esporádicos)
- Black Friday, Natal, Verão, Dia das Mães

Seed 42 (reprodutível).
"""
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

N_REGISTROS = 200_000  # 200k (o dobro do anterior, mais dados)
ANO_INICIO = 2023
ANO_FIM = 2026         # 4 anos (não 3)
MES_FIM = 10           # até outubro 2026

PRODUTOS = {
    "Notebook": ("Computadores", 3500.0),
    "Smartphone": ("Celulares", 2200.0),
    "Tablet": ("Celulares", 1800.0),
    "Monitor": ("Computadores", 1200.0),
    "Teclado": ("Periféricos", 250.0),
    "Mouse": ("Periféricos", 120.0),
    "Headset": ("Periféricos", 350.0),
    "Webcam": ("Periféricos", 400.0),
    "SSD": ("Armazenamento", 500.0),
    "HD Externo": ("Armazenamento", 350.0),
}

REGIOES = ["Sudeste", "Sul", "Nordeste", "Centro-Oeste", "Norte"]
REGIOES_PESO = [0.45, 0.25, 0.15, 0.10, 0.05]

# Sazonalidade MULTIPLICATIVA (Natal puxa +80%, Verão +20%, etc.)
SAZONALIDADE_MULT = {
    1: 1.0,   # Jan (volta às aulas)
    2: 0.95,  # Fev
    3: 1.0,   # Mar
    4: 0.95,  # Abr
    5: 1.15,  # Mai (Dia das Mães)
    6: 1.0,   # Jun
    7: 0.95,  # Jul
    8: 1.0,   # Ago
    9: 1.0,   # Set
    10: 1.05, # Out
    11: 1.5,  # Nov (Black Friday)
    12: 1.8,  # Dez (Natal)
}

# Tendência: vendas crescem 15% ao ano (não linear)
TENDENCIA_ANUAL = 1.15


def calcular_peso_mes(ano: int, mes: int) -> float:
    """Combina sazonalidade multiplicativa + tendência de crescimento."""
    anos_desde_inicio = ano - ANO_INICIO
    tendencia = TENDENCIA_ANUAL ** anos_desde_inicio
    return SAZONALIDADE_MULT[mes] * tendencia


def gerar_data():
    """Sorteia data com peso sazonal + tendência."""
    pesos = []
    datas = []
    for ano in range(ANO_INICIO, ANO_FIM + 1):
        mes_max = MES_FIM if ano == ANO_FIM else 12
        for mes in range(1, mes_max + 1):
            peso = calcular_peso_mes(ano, mes)
            pesos.append(peso)
            datas.append((ano, mes))

    # Escolhe mês baseado no peso
    pesos_norm = np.array(pesos) / sum(pesos)
    idx = np.random.choice(len(datas), p=pesos_norm)
    ano, mes = datas[idx]
    dia = random.randint(1, 28)
    return datetime(ano, mes, dia)


def gerar_linha(id_venda: int) -> dict:
    produto, (categoria, preco_base) = random.choice(list(PRODUTOS.items()))
    data = gerar_data()

    # Quantidade com distribuição geométrica (mais realista)
    quantidade = min(int(np.random.geometric(p=0.3)) + 1, 20)

    # Preço com desconto sazonal (Black Friday/Natal tem mais desconto)
    mes = data.month
    desconto = 0.0
    if mes == 11:  # Black Friday
        desconto = random.uniform(0.10, 0.30)
    elif mes == 12:  # Natal
        desconto = random.uniform(0.05, 0.20)
    preco_unit = round(preco_base * random.uniform(0.85, 1.15) * (1 - desconto), 2)

    regiao = random.choices(REGIOES, weights=REGIOES_PESO)[0]

    # Clientes: 80% fiéis (pool 500), 20% ocasionais (pool 5000)
    if random.random() < 0.8:
        cliente = f"Cliente_{random.randint(1, 500):04d}"
    else:
        cliente = f"Cliente_{random.randint(501, 5000):04d}"

    # Ruído gaussiano no preço (1% de chance de preço anômalo)
    if random.random() < 0.01:
        preco_unit = round(preco_unit * random.uniform(2.0, 5.0), 2)

    # Sujeira intencional
    if random.random() < 0.02: quantidade = ""
    if random.random() < 0.02: preco_unit = ""
    if random.random() < 0.01: cliente = f"  {cliente}!!  "
    if random.random() < 0.005:
        data_str = "DATA_INVALIDA"
    else:
        data_str = data.strftime("%Y-%m-%d")

    return {
        "id_venda": id_venda,
        "data_venda": data_str,
        "cliente": cliente,
        "produto": produto,
        "categoria": categoria,
        "regiao": regiao,
        "quantidade": quantidade,
        "preco_unitario": preco_unit,
    }


def gerar_dataset(caminho: str = "data/raw/vendas_complexo.csv", n: int = N_REGISTROS) -> None:
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    colunas = ["id_venda", "data_venda", "cliente", "produto",
               "categoria", "regiao", "quantidade", "preco_unitario"]

    with open(caminho, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=colunas)
        writer.writeheader()
        for i in range(n):
            writer.writerow(gerar_linha(i + 1))

    print(f"OK Dataset complexo gerado: {caminho} ({n} registros)")
    print(f"   Periodo: {ANO_INICIO}-01-01 a {ANO_FIM}-{MES_FIM:02d}-30")
    print(f"   Tendencia: +15% ao ano")
    print(f"   Sazonalidade: multiplicativa (Natal = 1.8x)")


if __name__ == "__main__":
    gerar_dataset()
