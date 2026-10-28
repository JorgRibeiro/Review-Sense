# ReviewSense PT-BR — Plano técnico v1.0

**Objetivo:** projeto reprodutível de classificação de sentimentos em avaliações brasileiras, com baseline TF-IDF + LinearSVC, fine-tuning de BERTimbau, avaliação rigorosa, modelo publicado no Hugging Face Hub e demonstração via Gradio.

**Data de referência:** 28/10/2026. **Estado:** planejado; ainda não treinado, avaliado ou publicado.

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
| Ambiente | Python 3.11+, Google Colab com GPU disponível, Git |
| Dados e EDA | `datasets`, `pandas`, `numpy`, `matplotlib` |
| Baseline | `scikit-learn`, `joblib` |
| Fine-tuning | `torch`, `transformers`, `accelerate` |
| Avaliação | `scikit-learn` (F1 macro, precision, recall, matriz de confusão), `pandas` |
| Hub | `huggingface_hub`, Model Card |
| Demo | `gradio`; hospedagem no Spaces sujeita ao plano da conta |
| Qualidade | `pytest`, `ruff`, GitHub Actions (etapa posterior) |

Preferir versões fixadas em `requirements.txt` após validar a primeira execução no Colab.

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
- [ ] Script que baixa os dados, verifica o esquema e constrói as classes
- [ ] EDA com distribuição de classes, nulos, duplicados e exemplos auditados
- [ ] Splits reproduzíveis e ausência de vazamento por textos idênticos
- [ ] Baseline TF-IDF + LinearSVC treinado e avaliado
- [ ] BERTimbau Base fine-tuned com PyTorch/Transformers e checkpoint salvo
- [ ] Comparativo real usando F1 macro, métricas por classe e matriz de confusão
- [ ] Modelo e tokenizer publicados no HF Hub com Model Card (dados, método, limitações, avaliação)
- [ ] README com instruções para reproduzir e links dos resultados

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
- Treinamento: Colab GPU, preferindo lotes pequenos e mixed precision quando suportada. Começar com smoke test em subconjunto pequeno e depois ampliar, sem prometer duração fixa nem resultados.

## 8. Primeira etapa da implementação

Criar repositório local, ambiente/dependências, `src/reviewsense/labels.py`, `src/reviewsense/data.py`, `tests/test_labels.py` e `notebooks/01_data_audit_eda.ipynb`. Validar o dataset e congelar o protocolo de splits antes de treinar os modelos.
