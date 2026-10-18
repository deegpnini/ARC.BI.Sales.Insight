# SalesInsight PY — Análise de Dados de Vendas

**Mini-projeto avaliativo — Módulo 1 — SCTEC/SENAI T4**

---

## Objetivo

Simular o papel de um analista de dados júnior: limpar um CSV de vendas bagunçado e gerar um relatório para a diretoria, usando apenas a biblioteca padrão do Python.

---

## O que o script faz

- **RF01** — Criar ou carregar dataset de vendas (sintético ou Superstore)
- **RF02** — Inspecionar dados (total, colunas, ausentes, primeiros registros)
- **RF03** — Limpar dados com datetime e regex (datas, tipos, campos vazios)
- **RF04** — Criar colunas derivadas (receita_total, mês, trimestre, ano, faixa)
- **RF05** — Calcular métricas agregadas (mês, produto, categoria, região)
- **RF06** — Segmentar clientes (Bronze, Prata, Ouro) com função lambda
- **RF07** — Organizar código em funções reutilizáveis e ordem superior
- **RF08** — Exportar resultados em CSV e JSON
- **RF09** — Executar fluxo completo (main) com if __name__ == __main__

## Datasets

- **vendas.csv** — dataset sintético (gerado pelo próprio código)
- Suporte adicional ao dataset real **Superstore** (Kaggle), usado para extrapolar a análise além do escopo mínimo

Colunas esperadas: data, cliente, produto, categoria, quantidade, preco_unitario, regiao

## Como rodar

python salesinsight.py              # modo automático
python salesinsight.py sintetico    # força dataset sintético
python salesinsight.py real         # força dataset real (Superstore)

**Requisitos:** apenas Python 3.x — sem dependências externas (sem Pandas/NumPy/Matplotlib), por exigência do escopo do Módulo 1.

## Saídas geradas

Após a execução, os seguintes arquivos são criados na pasta outputs/:

**Relatórios (RF08):**
- outputs/metricas_por_mes.csv
- outputs/segmentacao_clientes.csv
- outputs/estatisticas_gerais.json

**Visualizações (Bônus B04):**
- outputs/grafico_receita_mensal.svg
- outputs/grafico_segmentacao.svg
- outputs/grafico_barras_agrupadas.svg
- outputs/dashboard.html

## Bônus B04 — Gráficos SVG + Dashboard HTML

Como Matplotlib e Seaborn são proibidos pelo escopo, os gráficos foram gerados em **SVG puro** com Python (apenas strings) e o dashboard em **HTML estático** (sem JavaScript).

**Dashboard publicado:** https://deegpnini.github.io/salesinsight-py/

## Estrutura do repositório

| Branch | Propósito |
|--------|-----------|
| main | Versão estável/entregável |
| develop | Integração de features |
| feat/pipeline-dados | Desenvolvimento do pipeline de limpeza/análise |
| docs/readme | Documentação |

**Vídeo de demonstração (até 5 min):** https://youtu.be/nOI1rsR_if0?is=AWv913vM3qQWR6ky

---

## Ferramentas de apoio e declaração de autoria

**IAs generativas usadas como apoio:** DeepSeek, Claude, ChatGPT, Gemini e Grok — para geração de código, revisão técnica e auditoria cruzada de resultados.

**Sou o autor intelectual do projeto.** As decisões de arquitetura (Python puro, estrutura do pipeline), as escolhas metodológicas e a lógica geral (RF01–RF09) foram definidas e validadas por mim. Estou em transição de carreira e uso IAs como ferramenta de produtividade — não sou especialista em cada linha de código, mas sou capaz de explicar e defender qualquer decisão tomada no projeto, em qualquer avaliação.

Nenhuma solução externa foi copiada; o código foi gerado com apoio de IA e revisado/validado sob minha direção.

---

**Helyton Renato Gonçalves Ronchi**
SCTEC/SENAI — Turma T4

