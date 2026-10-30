# ReviewSense PT-BR — Plano técnico v1.4

**Objetivo:** projeto reprodutível de classificação de sentimentos em avaliações brasileiras, com baseline TF-IDF + LinearSVC, fine-tuning de BERTimbau, avaliação rigorosa, modelo publicado no Hugging Face Hub e demonstração via Gradio.

**Data de referência:** 30/10/2026. **Estado:** Fase 1 validada localmente; Fase 2 com Dummy e SVM treinados apenas no treino e avaliados na validação. Fase 3A concluída com smoke BERTimbau 256/96 na GPU local. Fase 3B concluída com três épocas e seleção pela validação. Teste final e publicação continuam pendentes.

**Adendo de ambiente:** o computador do desenvolvedor é o ambiente principal, usando CPU e GPU local conforme a tarefa e o suporte do hardware. Usar Colab ou outra infraestrutura externa apenas se memória ou tempo de execução se tornarem limitantes. A validação no Colab é opcional e não bloqueia a conclusão da Fase 1. Não é necessário manter ambientes separados de CPU, GPU e TPU; não há uso de TPU previsto.

## 1. Escopo e fontes

- Dados: `ruanchaves/b2w-reviews01` — 132.373 avaliações em português da Americanas.com, coletadas de jan–mai/2018. Fonte original: https://github.com/americanas-tech/b2w-reviews01; espelho: https://huggingface.co/datasets/ruanchaves/b2w-reviews01.
- Atenção: o espelho lista CC BY 4.0, mas o repositório original declara **CC BY-NC-SA 4.0**. Respeitar as restrições da fonte original e atribuir crédito; não redistribuir a base original sem necessidade. Confirmar o uso específico antes da publicação dos artefatos.
- Colunas principais para treino: `review_text` (entrada) e `overall_rating` (origem da classe). `review_title` é opcional para um experimento posterior, não faz parte do primeiro baseline.
- Rótulos definidos por estrelas (proxy, não anotação direta de sentimento): 1–2 = negativo (`0`); 3 = neutro (`1`); 4–5 = positivo (`2`).
- Excluir como *features* as colunas de nota, recomendação a amigo, dados demográficos, reviewer_id, data, identificadores de produto e demais metadados. Usar algumas dessas colunas apenas para auditorias de qualidade e análises agregadas quando justificadas. Não expor dados pessoais na aplicação.

## 2. Arquitetura

```
B2W-Reviews01
    |
    v
Validação e EDA -> limpeza / deduplicação -> splits imutáveis
    |                                  |
    +---- TF-IDF + LinearSVC           +---- tokenizer + BERTimbau
                       \             /
                        \           /
                 Avaliação comparável
                          |
               Métricas + relatório + erros
                          |
                Model Hub + Gradio demo
                          |
                     GitHub README
```

**Princípio:** notebooks são pontos de entrada/relatórios; código reutilizável vive em `src/`.

## 3. Tecnologias

| Área | Ferramentas |
| --- | --- |
| Ambiente | Python 3.11+, ambiente virtual local, CPU/GPU local conforme suporte, Git |
| Dados e EDA | `datasets`, `pandas`, `numpy`, `matplotlib` |
| Baseline | `scikit-learn`, `joblib` |
| Fine-tuning | `torch`, `transformers`, `accelerate` |
| Avaliação | `scikit-learn` (F1 macro, precision, recall, matriz de confusão), `pandas` |
| Hub | `huggingface_hub`, Model Card |
| Demo | `gradio`; hospedagem no Spaces sujeita ao plano da conta |
| Qualidade | `pytest`, `ruff`, GitHub Actions (etapa posterior) |

Registrar as versões do ambiente local validado em `requirements-lock.txt`; manter as dependências do pacote em `pyproject.toml`. A compatibilidade com Colab será verificada apenas se ele for utilizado.

## 4. Estrutura de pastas

```text
reviewsense-ptbr/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── pyproject.toml
├── configs/
│   └── train.yaml
├── data/
│   └── README.md                  # Fonte e instruções, sem dados brutos versionados
├── notebooks/
│   ├── 01_data_audit_eda.ipynb
│   ├── 02_tfidf_svm_baseline.ipynb
│   └── 03_bertimbau_finetune.ipynb
├── src/
│   └── reviewsense/
│       ├── __init__.py
│       ├── data.py                # Download, limpeza, normalização e split
│       ├── labels.py              # Definição de classes
│       ├── baseline.py            # Pipeline TF-IDF + LinearSVC
│       ├── train_transformer.py   # Fine-tuning
│       ├── evaluate.py            # Métricas e comparação
│       └── predict.py             # Inferência carregando modelo do Hub
├── app/
│   ├── app.py                     # Gradio
│   └── requirements.txt
├── tests/
│   ├── test_labels.py
│   └── test_preprocessing.py
├── reports/
│   └── figures/                  # Figuras versionadas sem PII
└── .github/
    └── workflows/
        └── checks.yml           # Testes/lint (após MVP)
```

## 5. Protocolo de dados e avaliação

1. Inspecionar tipos, nulos, distribuição de notas, textos vazios, duplicações, discrepâncias texto-nota e conteúdo potencialmente sensível.
2. Preservar acentos, negações e pontuação informativa. Normalizar apenas o necessário (espaços, caracteres inválidos), sem limpeza agressiva orientada à bag-of-words para o Transformer.
3. Evitar que textos normalizados idênticos apareçam em splits diferentes. Investigar textos iguais com notas conflitantes, sem esconder a decisão de limpeza.
4. Definir train/val/test em torno de 70/15/15, por semente fixa 42, procurando manter distribuição de classes e separar grupos de textos duplicados; registrar os índices/fingerprints finais. Avaliar alternativa temporal adicional, caso viável.
5. Ajustar tokenização e vocabulário TF-IDF **somente no treino**. Usar validação para escolhas de hiperparâmetros; teste fica intocado até a avaliação final.
6. Medir F1 macro como métrica principal e incluir precision/recall macro, F1 por classe, matriz de confusão e tempo de inferência. Mostrar contagem de amostras e limitações.
7. Para comparação justa na versão MVP, treinar baseline e Transformer no mesmo subconjunto de treino quando for preciso limitar custo, com os mesmos conjuntos de validação e teste. Evolução: treino com todo o conjunto de treinamento.
8. Não usar `overall_rating`, `recommend_to_a_friend` ou outras variáveis derivadas do alvo na inferência. A entrada principal é o texto da avaliação.

## 6. MVP e critérios de aceite

**MVP 1 — Pesquisa e modelo (prioridade máxima):**
- [x] Script que baixa os dados, verifica o esquema e constrói as classes
- [ ] EDA com distribuição de classes, nulos, duplicados e exemplos auditados
- [x] Splits reproduzíveis e ausência de vazamento por textos idênticos
- [x] Baseline TF-IDF + LinearSVC treinado e avaliado na validação; teste reservado
- [x] BERTimbau Base fine-tuned nos 88.684 textos de treino, checkpoint e estado retomável salvos (Fase 3B)
- [x] Comparativo de validação usando F1 macro, métricas por classe e matriz de confusão; teste final pendente
- [ ] Modelo e tokenizer publicados no HF Hub com Model Card (dados, método, limitações, avaliação)
- [x] README com instruções para reproduzir e links dos resultados de validação

**MVP 2 — Produto demonstrável:**
- [ ] Gradio local com textarea, classe prevista e probabilidades se devidamente calibradas
- [ ] Space publicado se o plano/permissões permitirem; alternativa: demo local e vídeo/GIF no README
- [ ] Testes mínimos de mapeamento de classes e pré-processamento

**Posterior ao MVP:** experimentos em dados completos, análise temporal e por categorias, calibração, CI, simplificação/quantização de inferência se necessária, comparação com modelo menor.

### Critério de encerramento

Concluir o MVP apenas com links verificáveis de GitHub e HF Hub, resultados reais reproduzíveis (não estimados) e limitações registradas. **Nenhuma métrica está definida de antemão.**

## 7. Recursos e publicação

- Modelo base: https://huggingface.co/neuralmind/bert-base-portuguese-cased (BERTimbau Base, 110M parâmetros, MIT).
- Guia de classificação: https://huggingface.co/docs/transformers/tasks/sequence_classification
- Regras atuais do Spaces: https://huggingface.co/docs/hub/spaces-overview — Static Spaces gratuitos; Gradio/Docker em compute requerem plano pago na criação, com exceção anunciada de até 2 Gradio ZeroGPU para contas pessoais gratuitas elegíveis. Não presumir que sua conta tem acesso.
- Treinamento: priorizar a GPU local, após verificar hardware, drivers e suporte da biblioteca na fase correspondente. Usar CPU quando apropriado. Começar com smoke test, lotes pequenos e mixed precision quando suportada; ampliar conforme RAM, VRAM e tempo medidos. Avaliar Colab ou outra infraestrutura externa apenas se os recursos locais se tornarem limitantes, sem prometer duração fixa nem resultados.

## 8. Primeira etapa da implementação

Criar repositório local, ambiente/dependências, `src/reviewsense/labels.py`, `src/reviewsense/data.py`, `tests/test_labels.py` e `notebooks/01_data_audit_eda.ipynb`. Validar o dataset e congelar o protocolo de splits antes de treinar os modelos.

## 9. Fase 2 — Baseline clássico implementado

Branch: `feat/phase2-classical-baseline`. Usar os artefatos congelados da Fase 1,
com leitura exclusiva de treino e validação. Comparar DummyClassifier e 12
configurações de TF-IDF + LinearSVC em CPU, conforme `configs/baseline.json`.
Selecionar por F1 macro de validação, sem refit em treino+validação. Não abrir o
CSV de teste, alterar splits, treinar BERTimbau ou usar GPU nesta fase.

Protocolo, conceitos e funções: `docs/FASE2_BASELINE.md`. Resultados reais,
condições de medição e evidências: `reports/validation_phase2.md`. Os pesos e o
vocabulário do modelo permanecem locais; somente métricas agregadas são versionadas.

## 10. Fase 3A — Diagnóstico e smoke concluído

Branch: `feat/phase3a-bertimbau-smoke`, a partir da main atualizada. Ciclo limitado
a 256 exemplos de treino e 96 de validação, estratificados com semente 42, sem
alterar os splits. BERTimbau Base cased, revisão fixa, max_length 128, batch 2,
acumulação 8, learning rate 2e-5, uma época e FP16 na RTX 3050 Laptop de 6 GB.

Treino, avaliação e recarga concluídos na configuração inicial, sem OOM. O PyTorch
2.14.1+cu132 foi preservado. Não houve leitura do teste, Colab, publicação no Hub
ou comparação definitiva com o SVM. O orçamento do smoke não demonstra qualidade
preditiva: as 96 predições foram positivas. O treinamento completo foi realizado posteriormente na Fase 3B.

Configuração: `configs/bertimbau_smoke.json`. Decisões e reprodução:
`docs/FASE3A_BERTIMBAU_SMOKE.md`. Métricas agregadas, GPU/VRAM, testes e recomendações:
`reports/validation_phase3a.md`. Modelos, cache e IDs individuais continuam locais.


## 11. Fase 3B — Piloto e fine-tuning completo concluídos

A branch `feat/phase3b-bertimbau-local` preserva as fases anteriores. Todos os textos
de treino/validação tiveram comprimentos medidos antes do corte. A configuração
escolhida no piloto foi max_length 256, batch 2/acumulação 8, FP16 e LR 2e-5.
Batch 4 coube na RTX 3050 de 6 GB, mas apresentou overflow; a decisão não usou
métricas de qualidade. Uma prova em processos separados confirmou retomada exata.

Três épocas nos 88.684 textos de treino, com seleção pelo F1 Macro dos 19.004 textos
de validação: **0,755522 / 0,741218 / 0,749669**. O checkpoint da primeira época
(update 5543) foi selecionado e salvo para inferência. Checkpoints completos
preservam estado para retomada. Não houve nova busca de hiperparâmetros.

Resultados completos, medições de GPU, incidentes numéricos e limitações:
[validation_phase3b.md](reports/validation_phase3b.md). Reprodução e funções:
[FASE3B_BERTIMBAU.md](docs/FASE3B_BERTIMBAU.md). O notebook 03 apenas inspeciona
resultados salvos. O teste final não foi aberto; publicação e demonstração continuam
pendentes. A comparação de validação com SVM não constitui avaliação independente.
