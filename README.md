# ARC.BI.Sales.Insight

Pipeline de Business Intelligence e Machine Learning para vendas de e-commerce, com foco em **validação rigorosa de previsão** e no relato honesto do resultado.

> **Projeto pessoal de portfólio.** Começou a partir de um mini-projeto do curso SCTEC/SENAI e hoje é **independente da avaliação acadêmica**.
>
> A versão original do curso está **somente** na tag [`v0.1-senai`](https://github.com/deegpnini/ARC.BI.Sales.Insight/tree/v0.1-senai) (`git show v0.1-senai:README.md`).
>
> **Sobre os dados:** EDA e dashboard usam dados **sintéticos** (scripts deste repositório). A validação principal de ML usa o dataset público **Olist** (dados reais).

> **Origem:** este projeto começou como o mini-projeto avaliativo do curso SCTEC/SENAI (Módulo 1, Semana 08), entregue em Python puro (biblioteca padrão), preservado em [`salesinsight-py`](https://github.com/deegpnini/salesinsight-py). A evolução para o portfólio pessoal (Pandas, Streamlit, LightGBM e Olist) foi feita **depois** da entrega e **não** faz parte da avaliação acadêmica.

---

## Resultado principal

**Pergunta:** o LightGBM supera baselines simples na previsão de receita semanal?

**Resposta neste setup:** não. O **naive de 1 passo** (último valor observado) foi igual ou melhor em todas as dobras.

**Dados:** [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle), 83 semanas úteis (2017-01-23 a 2018-08-20). Foram removidas 6 semanas truncadas (menos de 30% da mediana de pedidos). Os CSVs brutos **não** estão no Git; veja os termos de uso na página do dataset.

**Validação:** janela expansiva, 5 dobras, 12 semanas de teste (treino: 15, 27, 39, 51, 63 semanas). Avaliação **one-step** com lags observados (cada semana usa o valor real da semana anterior — comparável ao Naive1; **não** é forecast multi-step no escuro).

**Features:** `lag_1`…`lag_4`, `roll_mean_4`, `roll_mean_8`, `roll_std_4` (sempre com `shift(1)` antes do rolling). **Sem** `mes_num` / `semana_do_ano`.

**Hiperparâmetros (fixos):** 200 árvores, learning_rate 0,08, 15 folhas, max_depth 4, min_child_samples 3, random_state 42.

| Modelo | MAPE médio | WAPE médio |
|:---|:---:|:---:|
| LightGBM, alvo nível (v5c) | 26,46% | 27,00% |
| LightGBM, alvo razão y_t / y_(t-1) (v6) | 25,07% | 24,94% |
| Naive de 12 passos | 21,30% | 22,01% |
| Determinístico | 17,37% | 17,48% |
| **Naive de 1 passo** | **16,91%** | **16,93%** |

Vitórias do LightGBM contra o melhor baseline de cada dobra: **0/5** (nível e razão).

### Resultado por dobra (MAPE %, v5c)

| Dobra | Treino (semanas) | LightGBM | Naive 12 | Naive 1 | Determinístico | Semanas com fator 1,0 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 15 | 18,40 | 24,83 | 15,63 | 15,63 | 12/12 |
| 2 | 27 | 17,69 | 26,97 | 17,68 | 18,34 | 11/12 |
| 3 | 39 | 52,79 | 20,84 | 17,62 | 14,00 | 9/12 |
| 4 | 51 | 20,02 | 13,46 | 15,96 | 16,38 | 0/12 |
| 5 | 63 | 23,37 | 20,42 | 17,66 | 22,51 | 0/12 |

### Leitura dos resultados

- O LightGBM (nível) não supera o naive de 1 passo em nenhuma dobra. Na dobra 2 há empate técnico (~17,69% vs 17,68%); nas demais fica atrás. Mediana do MAPE: 20,02% vs 17,62%.
- O baseline determinístico não melhora o naive de forma consistente.
- O MAPE 52,79% na dobra 3 **não foi investigado em profundidade**.
- **Frase honesta para portfólio/entrevista:** "Com ~80 semanas e só lags/rolling, a persistência de 1 passo já captura quase todo o sinal previsível; o LightGBM não encontrou estrutura extra e aumentou o erro — por isso o baseline venceu em 5/5 dobras."

### Por que o LightGBM perdeu (hipóteses não testadas isoladamente)

1. **O naive é forte demais neste problema.** Em vendas semanais curtas, a correlação com a semana anterior é alta. O erro do naive já fica em torno de 17% MAPE.
2. **Poucos dados (fome de amostra).** Com 15–39 semanas no início do teste, árvores não encontram padrões robustos e tendem ao overfitting.
3. **Features quase iguais ao naive.** `lag_1` é a feature mais forte; as outras lags/rolling são correlacionadas. O boosting ou se comporta parecido com o naive, ou aprende padrões que só existiam no treino.
4. **Regime que muda (2017).** A série cresceu muito. O que valia no começo da série pode não valer no fim. O naive se adapta na hora (usa o nível atual); o LightGBM aprende regras no passado.
5. **Sem sinal externo permitido.** Bloqueamos calendário (`mes_num`, `semana_do_ano`) para evitar vazamento. Sem feriados, promoções, preço, o que sobra é a própria série.

### Definição dos baselines

- **Naive de 1 passo:** previsão em t = valor observado em t−1.
- **Naive de 12 passos:** último valor do treino repetido nas 12 semanas de teste (não se atualiza; mantido por transparência histórica).
- **Determinístico:** lag_1 × fator médio mensal (razões y_t/y_(t−1) no treino); fator = 1,0 se o mês não aparece no treino (32/60 semanas de teste).

Artefatos: `outputs/validate_v5c_corrigida_results.csv`, `outputs/validate_v6_ratio_results.csv`.

---

## Variantes testadas

| Versão | Dados | Alvo | Features | MAPE LightGBM |
|:---:|:---|:---|:---|:---:|
| v1 | Sintético (46 meses) | Nível | 12 (com ano/trimestre) | 9,99% |
| v2 | Sintético | Nível | 10 | 10,53% |
| v3 | Sintético | Razão | ver CSV em `outputs/` | — |
| v4 | Olist | Nível | 9 (com calendário) | 26,42% |
| v5a | Olist | Nível | 8 | 26,43% |
| v5b / v5c | Olist | Nível | 7 (só lags/rolling) | 26,46% |
| v6 | Olist | Razão | 7 | 25,07% |

A v5c = mesmo modelo da v5b + Naive1 + contagem de fallback do fator sazonal.

---

## Limitações

- Dashboard e EDA usam dados **sintéticos**.
- Série Olist curta (~1,6 ano) e com crescimento rápido em 2017.
- Dobras com treinos sobrepostos (não independentes).
- Fatores sazonais mensais com poucas observações.
- Um dataset, granularidade semanal apenas.
- Sem CI/testes automatizados neste momento.

---

## Como executar

Ambiente de referência: Python 3.14 (WSL2), pandas 3.0.6, numpy 2.5.4, lightgbm 4.7.0, streamlit 1.66.0.

### 1. Instalar dependências

```bash
git clone https://github.com/deegpnini/ARC.BI.Sales.Insight.git
cd ARC.BI.Sales.Insight
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Baixar os dados do Olist

1. Baixe o dataset público [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) no Kaggle.
2. Coloque os CSVs em `data/raw/`.
3. Rode o ETL para gerar `data/processed/`:

```bash
python src/data/process_olist.py
```

### 3. Reproduzir a validação temporal (resultado principal)

O resultado documentado no topo deste README (LightGBM perdeu para Naive1 em 5/5 dobras) é gerado por:

```bash
python src/models/validate_v6.py
```

Saída esperada: `outputs/validate_v6_ratio_results.csv` com colunas `mape_model`, `mape_naive1`, `wape_model`, `wape_naive1`, `fold`. As 5 dobras com MAPE do modelo acima do Naive1 confirmam a derrota do ML.

### 4. Rodar o dashboard (dados sintéticos)

```bash
streamlit run src/visualization/dashboard.py
```

O dashboard usa **dados sintéticos** (não Olist) — isso está avisado no próprio app.

### 5. Explorar os gráficos (dados sintéticos)

```bash
python src/visualization/explore.py
```

Gera os 7 PNGs em `outputs/` (sazonalidade, top produtos, regiões, correlação, segmentação, evolução, forecast).


