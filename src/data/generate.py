"""
ARC.BI.Sales.Insight — Gerador de dados sintéticos.

Gera um dataset realista de vendas de e-commerce com:
- 100.000 transações ao longo de 3 anos
- Sazonalidade (Natal, Black Friday)
- Clientes fiéis vs clientes ocasionais
- Produtos por categoria com preços variados
- Ruído intencional (nulos, outliers, datas inválidas)
"""
import csv
import random
from datetime import datetime
from pathlib import Path

import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

N_REGISTROS = 100_000
ANO_INICIO = 2023
ANO_FIM = 2025

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

# Sazonalidade mensal (índice 0-11)
SAZONALIDADE = [0.8, 0.85, 0.9, 1.0, 1.05, 1.0,
                0.95, 1.0, 1.05, 1.1, 1.3, 1.6]  # Dez = Natal


def gerar_data_inicio() -> datetime:
    """Sorteia data baseada em sazonalidade + peso dos meses."""
    pesos_mes = np.array(SAZONALIDADE) / sum(SAZONALIDADE)
    mes = np.random.choice(range(12), p=pesos_mes)
    ano = random.randint(ANO_INICIO, ANO_FIM)
    dia = random.randint(1, 28)
    return datetime(ano, mes + 1, dia)


def gerar_linha(id_venda: int) -> dict:
    """Gera uma linha sintética do dataset."""
    produto, (categoria, preco_base) = random.choice(list(PRODUTOS.items()))
    quantidade = random.choices([1, 2, 3, 5, 10], weights=[50, 25, 15, 8, 2])[0]
    preco_unit = round(preco_base * random.uniform(0.85, 1.15), 2)
    data = gerar_data_inicio()
    regiao = random.choices(REGIOES, weights=REGIOES_PESO)[0]
    
    # 80% clientes fiéis (pool de 500), 20% ocasionais (pool de 5000)
    if random.random() < 0.8:
        cliente = f"Cliente_{random.randint(1, 500):04d}"
    else:
        cliente = f"Cliente_{random.randint(501, 5000):04d}"
    
    # Sujeira intencional
    if random.random() < 0.02:
        quantidade = ""  # nulo
    if random.random() < 0.02:
        preco_unit = ""  # nulo
    if random.random() < 0.01:
        cliente = f"  {cliente}!!  "  # ruído
    if random.random() < 0.005:
        data = "DATA_INVALIDA"  # outlier
    else:
        data = data.strftime("%Y-%m-%d")
    
    return {
        "id_venda": id_venda,
        "data_venda": data,
        "cliente": cliente,
        "produto": produto,
        "categoria": categoria,
        "regiao": regiao,
        "quantidade": quantidade,
        "preco_unitario": preco_unit,
    }


def gerar_dataset(caminho: str = "data/raw/vendas.csv", n: int = N_REGISTROS) -> None:
    """Gera o dataset completo e salva em CSV."""
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    colunas = ["id_venda", "data_venda", "cliente", "produto",
               "categoria", "regiao", "quantidade", "preco_unitario"]
    
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=colunas)
        writer.writeheader()
        for i in range(n):
            writer.writerow(gerar_linha(i + 1))
    
    print(f"✅ Dataset gerado: {caminho} ({n} registros)")


if __name__ == "__main__":
    gerar_dataset()
