# ARC.BI.Sales.Insight

**Pipeline de Business Intelligence e Machine Learning para vendas de e-commerce.**

![Python](https://img.shields.io/badge/Python-3.14-blue)
![Pandas](https://img.shields.io/badge/Pandas-3.0-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-4.7-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.66-red)

---

## Sobre o projeto

ARC.BI.Sales.Insight é a evolução do mini-projeto SalesInsight PY (SCTEC/SENAI). Ele transforma um script de 400 registros em um **pipeline ETL completo** com dashboard interativo e validação rigorosa de Machine Learning.

**Este repositório preserva a versão original do curso** (script puro em Python) na tag v0.1-senai.

---

## O resultado científico (honesto)

O objetivo do projeto era testar se um modelo de Gradient Boosting (LightGBM) conseguiria superar baselines simples na previsão de vendas. Testamos em dados sintéticos e reais.

### Em dados reais (Olist semanal, 83 semanas)

Implementamos uma validação temporal rigorosa (TimeSeriesSplit, 5 dobras) comparando o LightGBM com um **Naive de 1 passo** (o baseline mais justo: prever que a próxima semana será igual à semana atual).

| Modelo | MAPE Médio | WAPE Médio | Vitórias |
|:---|:---:|:---:|:---:|
| **LightGBM (alvo nivel)** | 26,46% | 27,00% | **0/5** |
| **LightGBM (alvo razão)** | 25,07% | 24,94% | **0/5** |
| **Naive1 (baseline justo)** | **16,91%** | **16,93%** | — |

**Conclusão:** O LightGBM **não superou** o baseline determinístico simples em nenhuma configuração. O fator sazonal (baseline determinístico) também não se mostrou consistente (em 32 de 60 semanas de teste, ele caiu no fallback de fator 1.0).

**Por que isso aconteceu?ʂ�
- **Fome de dados:** O Olist tem apenas 83 semanas úteis. O LightGBM precisa de mais dados para encontrar padrões robustos, e acaba sofrendo overfitting.
- **A força do passado recente:** Em sérias curtas de e-commerce, o valor da semana anterior é o preditor mais forte. O modelo de ML adicionou ruído em vez de sinal.

**Isso é um resultado honesto e publicável.** O valor do projeto está na **arquitetura de validação** e na **maturidade de documentar o resultado negativo**, não em forçar o modelo a "ganhar".

---

## Como executar o projeto

### 1. Ambiente
```bash
git clone https://github.com/deegpnini/ARC.BI.Sales.Insight.git
cd ARC.BI.Sales.Insight
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Dados Sintéticos (EDA, Dashboard)
```bash
python3 src/data/generate_complex.py
python3 src/data/clean_complex.py
python3 src/visualization/explore.py
streamlit run src/visualization/dashboard.py
```

### 3. Dados Reais (Olist)
*Requer download do dataset no Kaggle.*
```bash
python3 src/data/process_olist.py
python3 src/models/validate_v5c.py
python3 src/models/validate_v6.py
```

---

## Estrutura do projeto

```
ARC.BI.Sales.Insight/
├── data/
    ├── raw/             # Dados brutos (ignorados no Git)
    └── processed/       # Dados limpos (ignorados no Git)
├── outputs/             # Gráficos, métricas e resultados
├── src/
│   ✔── data/            # Scripts de ETL e geração
ₔ   └── models/          # Validações de ML (v3 a v6)
└── visualization/   # Streamlit e EDA
├── requirements.txt
└── README
```

---

## Variantes testadas (transparência total)

| Versão | Descrção | Dataset | Resultado Principal |
|:---:|:---|:---|:---|
| v1 | Features de calendário e nivel | Sintético | MAPE 9,99% |
| v2 | Sem calendário | Sintético | Não superou o determinístico |
| v3 | Alvo razão (y_t/lag_1) | Sintético | MAPE 10,64% |
| v4/v5 | Primeiras rodadas no Olist | Olist | MAPE ~26% |
| v5c | Validação com Naive1 e assert | olist | MAPE 26,46% |
| **v6** | **Alvo razão no Olist** | **Olist** | **MAPE 25,07%** |

Todas as variantes foram rodadas após ver os resultados anteriores.
...

**Autor:** Helyton Ronchi (@deegpnini)
