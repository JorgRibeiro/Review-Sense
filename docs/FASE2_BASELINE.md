# Fase 2 — Baseline clássico em CPU

Esta fase compara modelos usando exclusivamente os splits congelados de treino e
validação. Não recria dados, não acessa `test.csv`, não treina BERTimbau e não usa GPU.
A seleção é pelo **F1 macro de validação**, não por accuracy. Não há refit usando
validação depois da escolha. O teste fica reservado para uma avaliação final futura.

## Execução

A partir da raiz, com os artefatos da Fase 1 já presentes:

```bash
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m reviewsense.baseline \
  --artifacts-dir data/processed/phase1-final \
  --reference reports/data_audit_summary.json \
  --config configs/baseline.json \
  --output-dir models/baseline/phase2-v1
pytest -q
ruff check src tests
ruff format --check src tests
```

**Não execute novamente o pipeline da Fase 1 para treinar esta fase.** A referência
versionada ancora os fingerprints de treino/validação. O manifest local também é
conferido. Alteração de dados, IDs, rótulos ou distribuição causa erro. Não é preciso
abrir o teste para validar as partições permitidas; sua integridade já foi verificada
na Fase 1 e seu conteúdo não é reexaminado agora.

O diretório de saída deve estar ausente. Para outra execução, use outro nome, como
`models/baseline/phase2-v2`. Os arquivos salvos são:

- `selected_model.joblib`: pipeline ajustado (TF-IDF e SVM) ou Dummy, se ele vencer.
- `results.json`: configuração completa, fonte, fingerprints, métricas de todos os
  candidatos, convergência, tempos, hardware, versões e hashes do código/modelo.
- `config.json`: cópia da configuração usada.

Os modelos e seu vocabulário ficam locais e ignorados pelo Git: termos podem
revelar informações dos textos. Joblib usa pickle; carregue somente artefatos
locais confiáveis. O checksum detecta corrupção, mas não autentica a origem.
A licença do dataset continua sendo a original CC BY-NC-SA 4.0. A entrega versiona
somente código, documentação e resultados agregados, sem avaliações ou IDs individuais.

## O que os algoritmos fazem

**DummyClassifier (`most_frequent`)** aprende a classe mais comum no treino e prevê
sempre essa classe, ignorando o conteúdo do texto. É uma referência mínima para
mostrar se um modelo de fato acrescenta valor. Uma accuracy aparentemente alta
pode ser obtida só por causa do desbalanceamento.

**TF-IDF** converte cada avaliação em um vetor esparso de pesos de palavras. A
frequência no texto aumenta a importância de um termo; sua presença em muitos
textos de treino reduz essa importância relativa. Vocabulário e IDF são aprendidos
somente no treino. A validação é apenas transformada pelo vocabulário já aprendido.

**LinearSVC** aprende separações lineares entre as classes no espaço TF-IDF. Aqui
usa uma classificação um-contra-restante, penalidade L2 e perda squared hinge.
Não precisa de GPU e costuma ser uma referência útil para texto esparso. Suas
margens de decisão não são probabilidades calibradas.

**F1 macro** calcula o F1 de cada classe e depois tira a média, dando a negativo,
neutro e positivo o mesmo peso. Precision responde quantas previsões de uma classe
estão corretas; recall responde quanto dos exemplos reais daquela classe foi
recuperado. A matriz de confusão mostra os erros entre pares de classes, com classe
real nas linhas e prevista nas colunas. Divisões indefinidas recebem zero.

## Busca limitada e decisões

A grade de `configs/baseline.json` tem **12 combinações**, sem validação cruzada ou
busca aleatória adicional. Todos os candidatos recebem exatamente o mesmo treino
e a mesma validação. Cada pipeline é ajustado do início; o tempo inclui TF-IDF e
SVM. O vocabulário não é cacheado entre candidatos, mantendo a implementação direta
e os tempos completos comparáveis.

| Hiperparâmetro | Valores | Propósito |
| --- | --- | --- |
| `C` | 0,1; 1; 10 | Explorar regularização maior, intermediária e menor |
| `ngram_range` | (1,1); (1,2) | Comparar palavras isoladas com inclusão de pares como “não gostei” |
| `class_weight` | `None`; `balanced` | Comparar pesos padrão e pesos inversos à frequência das classes do treino |

Escolhas mantidas fixas:

- `min_df=2`: excluir termos que aparecem em apenas um texto de treino.
- `max_features=80000`: limitar a dimensão e o custo da representação; a seleção
  das features também usa somente o treino. Não é garantia de anonimização.
- `sublinear_tf=True`, IDF suavizado e norma L2: reduzir influência de repetições e
  normalizar a escala dos vetores. Os demais defaults estão presos à versão registrada.
- `lowercase=True`, `strip_accents=None`, sem stopwords: unir caixa sem remover
  acentos ou negações. O padrão de tokens usa palavras com dois ou mais caracteres;
  pontuação, muitos emojis e palavras de uma letra não são representados.
- `dual=False`, `tol=1e-4`, `max_iter=5000`: solução primal e orçamento explícito.
  Avisos de não convergência são registrados e impedem a seleção do candidato.
- Semente 42 e candidatos sequenciais, com limite de uma thread nas bibliotecas
  numéricas controladas por `threadpoolctl`.

Vence o maior F1 macro de validação. Empates exatos entre SVMs mantêm a primeira
configuração da grade (ngram, depois C, depois peso). Se Dummy empatar ou superar
o melhor SVM, ele será salvo como vencedor. A grade não é ampliada após observar
os resultados. O pipeline vencedor permanece ajustado apenas no treino.

## Tempos e reprodução

O treino é medido com relógio de parede (`perf_counter`), incluindo vetorização e
ajuste. A inferência final compara Dummy e melhor SVM com lote de 256, um lote de
aquecimento e três passagens completas sobre a validação em memória. A mediana
inclui transformação TF-IDF e classificação, quando aplicáveis. Leitura de disco,
cálculo de métricas e serialização ficam fora dessa medição de inferência.

`ms/texto` é custo amortizado em lote; não mede latência de uma requisição isolada.
O pico de memória é do processo inteiro, incluindo dados, busca e serialização,
e não de um modelo específico. A medição registra CPU, RAM, bibliotecas e threads;
não controla processos externos, temperatura ou frequência da CPU. Não se promete
igualdade de tempos entre execuções ou computadores.

Use as versões de `requirements-lock.txt` para repetir o ambiente local validado.
O resultado registra os hashes dos módulos e do modelo, a semente e os fingerprints
dos splits. Após salvar, o pipeline é recarregado e suas predições de validação são
comparadas com as anteriores, sem publicar essas predições individuais.

## Funções principais

| Função | Propósito |
| --- | --- |
| `load_training_data` | Abrir somente treino/validação e conferir os fingerprints congelados |
| `validate_training_pair` | Validar os dados preparados e detectar sobreposição entre os dois conjuntos |
| `candidate_parameters` | Expandir e validar a grade, recusando mais de 12 candidatos ou repetições |
| `build_svm_pipeline` | Construir TF-IDF e LinearSVC ainda não ajustados |
| `run_experiment` | Coordenar ajuste, comparação, seleção, medição e persistência |
| `classification_metrics` | Produzir métricas macro/por classe e matriz de confusão agregadas |
| `benchmark_inference` | Medir predição em lotes após aquecimento, incluindo o pipeline de texto |
| `load_experiment` | Verificar checksum e reabrir o modelo local e seus resultados |
| `plot_confusion_matrix` | Gerar figura apenas com contagens agregadas |

O notebook `02_tfidf_svm_baseline.ipynb` chama essas funções. Seu modo padrão
reabre uma execução salva, verifica suas métricas e mostra comparações e figuras.
O treinamento pelo notebook é opcional, com `TRAIN=True` e uma saída nova.

## Limitações

- A validação foi usada para escolher entre 12 configurações: seu resultado tem
  viés de seleção e não é uma estimativa final independente de generalização.
- As estrelas são supervisão indireta. A classe neutra pode conter opiniões mistas.
- O corpus é de comércio eletrônico de 2018. Outros setores e épocas requerem avaliação própria.
- Quase duplicatas, produtos e autores compartilhados não foram separados na Fase 1.
- TF-IDF de palavras não modela contexto longo, ironia ou composição semântica de forma explícita.
- Esta fase não calibra probabilidades, não mede desempenho no teste e não compara com BERTimbau.

Referências técnicas primárias: [DummyClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html),
[TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html),
[LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) e
[prevenção de vazamento](https://scikit-learn.org/stable/common_pitfalls.html).
