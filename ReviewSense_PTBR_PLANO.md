# ReviewSense PT-BR — Plano técnico v1.8

**Objetivo:** projeto reprodutível de classificação de sentimentos em avaliações brasileiras, com baseline TF-IDF + LinearSVC, fine-tuning de BERTimbau, avaliação rigorosa, modelo publicado no Hugging Face Hub e demonstração via Gradio.

**Data de referência:** 30/10/2026. **Estado:** Fase 1 validada localmente; Fase 2 com Dummy e SVM treinados apenas no treino e avaliados na validação. Fase 3A concluída com smoke BERTimbau 256/96 na GPU local. Fase 3B concluída com três épocas e seleção pela validação. Fase 3C concluída com avaliação independente de teste. Fase 4A concluída com inferência reutilizável e Gradio local em GPU/CPU; Fase 4B tecnicamente concluída com Model Card e pacote independente. Publicação não autorizada: REQUIRES_REVIEW, pendente de direitos e autorização na Fase 4C.

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
│       └── predict.py             # Inferência local verificada; futuro resolvedor do Hub
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
- [x] Comparativo independente no teste usando F1 macro, métricas por classe, matrizes e concordância pareada (Fase 3C)
- [ ] Modelo e tokenizer publicados no HF Hub com Model Card (dados, método, limitações, avaliação)
- [x] README com instruções para reproduzir e links dos resultados de validação

**MVP 2 — Produto demonstrável:**
- [x] Gradio local com textarea, classe/ID e scores normalizados não calibrados no BERT; SVM sem probabilidades (Fase 4A)
- [ ] Space publicado se o plano/permissões permitirem; alternativa: demo local e vídeo/GIF no README
- [x] Testes de classes, validação, cache, integridade, CPU/GPU e interface; 217 testes offline aprovados

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


## 12. Fase 3C — Avaliação final independente concluída

Branch `feat/phase3c-final-evaluation`, a partir da main 5d18377. Código/configuração
com protocolo imutável congelados em `90893aa` antes da abertura do teste. Preflight
com zero acessos, modelos offline e benchmark na validação. Uma execução definitiva
com uma leitura de test.csv e uma passagem por modelo nos mesmos 19.004 exemplos.
Sem treinamento, refit, calibração, tuning ou seleção orientada pelo teste.

F1 Macro de teste: **SVM 0,723094; BERTimbau 0,754461**.
BERTimbau da época 1/update 5543 permanece o candidato definido anteriormente.
Comparação descritiva, sem afirmação de superioridade estatística. Relatório e
métricas: [validation_phase3c.md](reports/validation_phase3c.md) e
[phase3c_final_metrics.json](reports/phase3c_final_metrics.json). O notebook 04
reabre somente agregados e não executa inferência. Os relatórios e artefatos
anteriores foram preservados. O teste não deve orientar novos ajustes no experimento.

A parte de treinamento e avaliação planejada está encerrada. Fase 4 (publicação,
Model Card e Gradio) permanece pendente; não houve publicação, push ou merge
nesta entrega. Limitações de domínio, proxy por estrelas e neutralidade continuam
explicitadas nos relatórios.


## 13. Fase 4A — Inferência e Gradio local concluídos

Branch `feat/phase4a-inference-gradio`, baseada na main da340b8 após merge da Fase 3C.
Serviço em `src/reviewsense/predict.py`, Gradio em `app/app.py`, pesos originais
validados por SHA-256 e carregados sob demanda. BERT cased, max_length 256,
FP16/CUDA quando compatível e FP32/CPU funcional. SVM reutiliza o TF-IDF salvo.
Sem treino, refit, calibração, seleção ou nova avaliação do teste.

BERT é o padrão e exibe scores normalizados não calibrados; SVM retorna apenas a
classe. Até 8.000 caracteres por texto, microbatches BERT de 2, API reutilizável,
cache/lock, erros públicos seguros e CPU como alternativa. Gradio é local em
127.0.0.1:7860, share/analytics/histórico desligados, sem logging de entradas.

Modelos reais validados na interface e no serviço em GPU/CPU, exemplos sintéticos,
layout de 320 a 1440 px, 217 testes offline e verificações de integridade aprovados.
Evidências e medições em `reports/phase4a_inference_smoke.json`; decisões e comandos
em [FASE4A_INFERENCE_GRADIO.md](docs/FASE4A_INFERENCE_GRADIO.md).
Os artefatos e resultados anteriores continuam preservados, sem acesso ao teste.

Não houve publicação, push ou merge nesta entrega. A Fase 4B tratará Model Card,
licenciamento e preparação da publicação. O projeto ainda não está publicado.


## 14. Fase 4B — Model Card e pacote local tecnicamente concluídos

Branch `feat/phase4b-model-card-packaging`, base main d3d71f3. Auditoria separa
MIT do código, MIT declarado do base, CC BY-NC-SA 4.0 do dataset original e a
situação dos pesos ajustados. O espelho declara CC BY 4.0, divergência não resolvida.
Status dos pesos **REQUIRES_REVIEW**, upload **não autorizado**; Model Card
`license: unknown`. Prontidão técnica não equivale a autorização jurídica.

Pacote candidato `1.0.0-rc.1`, identidade `reviewsense-ptbr-bertimbau-v1`, com
quatro artefatos Transformers e cinco documentos por allowlist explícita. Bytes
preservados, sem dados, estados de treino, SVM ou cache. Manifest técnico fora do
pacote, com hashes, métricas congeladas e repo_id HF ainda nulo. O código literal
do Model Card carrega isolado, com cache vazio e zero acesso à origem/rede.

Equivalência confirmada em CPU/FP32 e GPU/FP16 em cinco entradas sintéticas por modo,
sem novas métricas de dataset. **253 testes offline aprovados**. Nenhuma leitura
adicional de test.csv, treinamento, calibração, alteração da aplicação 4A, publicação,
push, deploy ou post no LinkedIn nesta execução. Pesos permanecem ignorados pelo Git.

Referências: [auditoria](docs/FASE4B_LICENSE_REVIEW.md),
[Model Card](publishing/model_card/README.md),
[guia](docs/FASE4B_MODEL_PACKAGING.md) e
[relatório](reports/validation_phase4b.md).

## 15. Fase 4C — Engenharia de preparação local; lançamento bloqueado

[FASE4C_RELEASE_PLAN.md](docs/FASE4C_RELEASE_PLAN.md) organiza resolução dos direitos,
autorização explícita e confirmação do destino, publicação da allowlist aprovada,
validação por revisão fixa, adaptação do resolvedor de artefatos, demonstração
pública, links e release estável. Requer descritor público de inferência separado
dos relatórios internos do desenvolvedor. Não se presume hospedagem disponível.

Somente após artefatos públicos funcionarem será escrito e revisado o post final
no LinkedIn, com métricas reais, treinamento local, engenharia, limites, visual
autorizado e links verificáveis. Não haverá publicação automática nem alegação
de parceria ou de rotulagem manual. Se direitos relevantes não forem esclarecidos,
o gate de publicação permanece fechado e o lançamento precisa de nova decisão.


### Preparação local da primeira execução 4C

Branch `feat/phase4c-release-readiness`, a partir da main 562e0fb (Fases 1–4B).
Descritor mínimo separado do pacote 4B, resolvedor local independente de relatórios,
adaptador HF somente cache offline/mock por SHA imutável e preflight com gates
independentes. Não alterar allowlist/candidato original nem pesos/datasets.

Direitos `REQUIRES_REVIEW`, upload false, destino e autorização pendentes.
Comunicação formal preparada, sem envio. CPU Basic condicionado a plano/elegibilidade;
ZeroGPU exige adaptação e confirmação, sem compra. UI distingue local/hospedado,
mantém limites, erros seguros e privacidade sem promessas absolutas.

[Guia 4C](docs/FASE4C_RELEASE_READINESS.md),
[direitos](docs/FASE4C_RIGHTS_OUTREACH.md),
[hospedagem](docs/FASE4C_HOSTING_DECISION.md) e
[evidências](reports/validation_phase4c_readiness.md).
O lançamento público e o post definitivo de LinkedIn continuam pendentes.
