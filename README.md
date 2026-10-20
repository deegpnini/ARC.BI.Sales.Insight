# ARC.BI.Sales.Insight

Pipeline de Business Intelligence e Machine Learning para vendas de e-commerce, com foco em **validação rigorosa de previsão** e no relato honesto do resultado.

> **Projeto pessoal de portfólio.** Começou a partir de um mini-projeto do curso SCTEC/SENAI e hoje é independente da avaliação acadêmica. A versão original do curso está na tag [`v0.1-senai`](https://github.com/deegpnini/ARC.BI.Sales.Insight/tree/v0.1-senai) (README original: `git show v0.1-senai:README.md`).
>
> **Sobre os dados:** a EDA, o dashboard e a validação "sintética" usam dados **sintéticos**, gerados por scripts deste repositório. A validação com dados reais usa o dataset público do Olist.

---

## Resultado principal

**Pergunta:** o LightGBM supera baselines simples na previsão de receita semanal?

**Resposta neste setup:** não. O naive de 1 passo (último valor observado) foi igual ou melhor em todas as dobras.

**Dados:** [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle), 83 semanas úteis (2017-01-23 a 2018-08-20). Foram removidas 6 semanas truncadas (menos de 30% da mediana de pedidos da série). Os dados brutos não estão no repositório; consulte os termos de uso na página do dataset.

**Validação:** janela expansiva, 5 dobras, teste de 12 semanas; treino de 15, 27, 39, 51 e 63 semanas. Previsão de 1 passo com lags observados. Features: `lag_1` a `lag_4`, média móvel de 4 e de 8 semanas e desvio móvel de 4 semanas (todos calculados com `shift(1)`). Sem variáveis de calendário. Hiperparâmetros fixos no código: 200 árvores, taxa de aprendizado 0,08, 15 folhas, profundidade máxima 4, `min_child_samples=3`, `random_state=42`.

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

- O LightGBM (alvo nível) não supera o naive de 1 passo em nenhuma dobra. Na dobra 2 há empate (17,69% contra 17,68%); nas dobras 1, 4 e 5 fica de 2,8 a 5,7 pontos percentuais atrás. Mediana do MAPE: 20,02% contra 17,62%.
- O determinístico não melhora o naive de forma consistente. Onde o fator existe para todas as semanas (dobras 4 e 5), ele é pior; só melhora na dobra 3, em 3 semanas com fator estimado a partir de poucas observações. Não há evidência de sazonalidade mensal útil nesta série.
- O desempenho do LightGBM na dobra 3 (52,79%) **não foi investigado**.
- Hipóteses **não testadas** para o desempenho do LightGBM: série curta (menos de 2 anos) e com forte crescimento em 2017; semana de Black Friday (24/11/2017) no treino da dobra 3; árvores não extrapolam o nível da série.
- O resultado vale para este dataset e este setup. Não deve ser lido como regra geral.

### Definição dos baselines

- **Naive de 1 passo:** previsão da semana t = valor observado na semana t-1.
- **Naive de 12 passos:** último valor do treino repetido nas 12 semanas de teste. Não se atualiza e, por isso, não é comparável aos demais; foi mantido por transparência, pois aparece nos resultados das versões anteriores.
- **Determinístico:** previsão = valor da semana t-1 multiplicado por um fator por mês do calendário. O fator é a média das razões semanais y_t / y_(t-1) daquele mês no treino; vale 1,0 quando o mês não aparece no treino (32 das 60 semanas de teste).

---

## Variantes testadas

Todas foram feitas depois de ver os resultados das anteriores. Os números completos estão em `outputs/`.

| Versão | Dados | Alvo | Features | MAPE do LightGBM |
|:---:|:---|:---|:---|:---:|
| v1 | Sintético (46 meses) | Nível | 12 (com ano e trimestre) | 9,99% |
| v2 | Sintético | Nível | 10 (sem ano e trimestre) | 10,53% |
| v3 | Sintético | Razão | ver `outputs/validate_v3_results.csv` | - |
| v4 | Olist semanal | Nível | 9 (com `mes_num` e `semana_do_ano`) | 26,42% |
| v5a | Olist semanal | Nível | 8 (`semana_do_ano` permaneceu por engano) | 26,43% |
| v5b / v5c | Olist semanal | Nível | 7 (só lags e médias móveis) | 26,46% |
| v6 | Olist semanal | Razão | 7 | 25,07% |

A v5c usa o mesmo modelo da v5b; a diferença é a inclusão do naive de 1 passo e da contagem de semanas com fator 1,0.

---

## Limitações

- O dashboard e a EDA usam dados sintéticos (com tendência e sazonalidade definidas no gerador).
- A série do Olist é curta (cerca de 1,6 ano) e tem crescimento rápido em 2017.
- As 5 dobras têm treinos sobrepostos, então não são independentes.
- Fatores sazonais por mês foram estimados com poucas observações.
- Um único dataset e uma única granularidade (semanal).
- Sem integração contínua (CI).

---

## Como executar

Testado com Python 3.14 (WSL2, Ubuntu 26.04), pandas 3.0.6, numpy 2.5.4, lightgbm 4.7.0 e streamlit 1.66.0.

```bash
git clone https://github.com/deegpnini/ARC.BI.Sales.Insight.git
cd ARC.BI.Sales.Insight
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dados sintéticos, EDA, dashboard e previsão:

```bash
python3 src/data/generate_complex.py
python3 src/data/clean_complex.py
python3 src/visualization/explore.py
streamlit run src/visualization/dashboard.py
python3 src/models/forecast.py
python3 src/models/validate_v3.py
```

Dados reais (Olist): baixe o dataset no Kaggle, extraia em `data/raw/olist/` (os arquivos usados estão em `src/data/process_olist.py`) e rode:

```bash
python3 src/data/process_olist.py
python3 src/models/validate_v5c.py
python3 src/models/validate_v6.py
```

---

## Versão do curso

A versão original do mini-projeto SCTEC/SENAI (script Python puro com a biblioteca padrão, 400 registros) está nos arquivos `salesinsight.py` e `vendas.csv` e na tag `v0.1-senai`.

Vídeo de demonstração do curso: https://youtu.be/nOI1rsR_if0

---

**Autor:** Helyton Ronchi ([@deegpnini](https://github.com/deegpnini))
