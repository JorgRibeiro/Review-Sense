# Validação da Fase 1

Execução realizada em **28/10/2026**, no ambiente local Linux, Python **3.11.14**.
Commit da implementação validada: `2f99568c6cdf91c38127d834180ef8a79feca87b`.

A Fase 1 foi implementada conforme `ReviewSense_PTBR_PLANO.md`; nenhum treinamento,
avaliação de modelo, publicação no Hub ou demonstração Gradio foi executado.

## Verificações executadas

- Instalação do pacote e dependências em `.venv` isolado.
- `pytest -q`: **74 testes aprovados** na verificação final (0,85 s neste ambiente).
- `ruff check src tests`: aprovado.
- `ruff format --check src tests`: aprovado (12 arquivos).
- `python -m pip check`: sem dependências quebradas.
- Resolução de `requirements-lock.txt` com `pip install --dry-run`: aprovada
  no ambiente local instalado; o lock não foi testado em outro sistema operacional.
- Download real do CSV oficial de aproximadamente 49 MB, na revisão
  `4639429ec698d7821fc99a0bc665fa213d9fcd5a`; checksum verificado pelo carregador.
- CLI executado no corpus completo, com artefatos finais em
  `data/processed/phase1-final/` (locais, ignorados pelo Git).
- Releitura dos CSVs pelo `load_artifacts`: textos/IDs, rótulos, atribuição aos
  splits, contagens e fingerprints verificados contra o manifest.
- Notebook validado com `nbformat` e executado integralmente com `nbclient`, kernel
  Python da `.venv`: **8 células de código, zero erros**. Figuras e tabelas geradas
  localmente; execução da célula de exportação com `EXPORT=False`.
  O caminho ativo de exportação foi validado separadamente pelo CLI e testes.
  O notebook versionado permanece sem saídas.

Os testes automatizados usam somente dados sintéticos e mocks, sem rede ou dados
pessoais. Cobrem mapeamento de notas, entradas inválidas, normalização Unicode,
preservação de negações/pontuação, nulos, CSVs com aspas/múltiplas linhas, dois
separadores, duplicatas, conflitos, integridade do cache, falhas de download,
reprodutibilidade, estratificação, cobertura, detecção de vazamento, persistência,
recusa de sobrescrita e detecção de adulteração dos artefatos.

## Resultado real dos dados

| Etapa | Avaliações |
| --- | ---: |
| Fonte original | 132.373 |
| Removidas por texto vazio/ausente | 3.275 |
| Removidas por classes conflitantes (21 grupos) | 43 |
| Repetições removidas após excluir conflitos | 2.363 |
| Conjunto preparado | **126.692** |

Política: normalização NFC/espaços/controles ASCII, remoção integral de grupos com
classes conflitantes e um exemplar por texto normalizado restante. Notas diferentes
na mesma classe não são conflito de classe. A política altera as proporções das
classes, e as contagens antes/depois são registradas.

| Split | Negativo | Neutro | Positivo | Total |
| --- | ---: | ---: | ---: | ---: |
| Treino | 22.818 | 11.089 | 54.777 | **88.684** |
| Validação | 4.889 | 2.377 | 11.738 | **19.004** |
| Teste | 4.890 | 2.376 | 11.738 | **19.004** |

Semente 42; frações solicitadas 70/15/15. Cobertura integral do conjunto preparado
e ausência de textos normalizados idênticos entre splits foram verificadas.

Fingerprint SHA-256 do conjunto preparado:
`d3a10093139aba35805224ccc041bab5a9b51314791dfcb6f94186e908b10801`.

O resumo completo de contagens, comprimentos, versões, fonte, checksum e
fingerprints está em [data_audit_summary.json](data_audit_summary.json).
Esse resumo não contém textos nem IDs individuais das avaliações. Os manifests
com IDs e os CSVs permanecem somente em `data/`, ignorados pelo Git.

## Ambiente principal e limites da validação

Por decisão do desenvolvedor, o computador local é o ambiente principal. As
validações locais acima atendem à Fase 1; Colab passa a ser uma alternativa opcional
caso os recursos locais se tornem limitantes.

A instalação e execução **dentro do Google Colab não foram realizadas**. Se essa
alternativa for adotada, confirmar instalação, execução das células, acesso à fonte
e persistência dos artefatos fora de `/content`. A GPU local também não foi validada
nesta fase, pois o pipeline de dados roda em CPU. Seu suporte será verificado antes
do treinamento dos modelos.

A execução técnica não substitui auditoria humana de exemplos, dados sensíveis e
discrepâncias semânticas entre texto e nota. A normalização não anonimiza o conteúdo.
A garantia de disjunção não inclui quase duplicatas, autores, produtos ou tempo.
SVM, BERTimbau, métricas de classificação e publicação são trabalho de fases futuras.
