# Validacoes Legadas (v1 - v5c)

Esta pasta contem as versoes anteriores do pipeline de validacao temporal.
Foram movidas para ca em 2026-11 para manter apenas o validate_v6.py ativo.

## Ordem cronologica

| Arquivo | Versao | O que testava |
|---------|--------|---------------|
| validate.py | v1 | Primeira validacao (baseline inicial) |
| validate_v3.py | v3 | Ajustes de features |
| validate_v4.py | v4 | Introducao de lags e medias moveis |
| validate_v5.py | v5 | Exploratoria com diferentes alvos |
| validate_v5b.py | v5b | Refinamento do alvo |
| validate_v5c.py | v5c | Alvo nivel (resultado final: MAPE 26.46%) |

## Resultado final consolidado

Ver  e .

**Honestidade Radical:** em todas as 5 dobras, o LightGBM perdeu para o Naive1
(MAPE medio 25.07% vs 16.91%). Isso esta documentado no README principal.
