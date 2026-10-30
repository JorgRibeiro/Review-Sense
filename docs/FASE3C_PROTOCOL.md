# Fase 3C — Protocolo congelado antes da abertura do teste

Este protocolo é registrado junto ao código em commit anterior à avaliação.
Não será alterado após observar resultados. Referência inicial: main `5d18377`,
merge da Fase 3B. Os hashes exatos dos artefatos e arquivos do tokenizer estão em
`configs/phase3c_final.json`; o preflight ancora também protocolo, código,
configuração, referências e manifest. A avaliação exige árvore Git limpa e o mesmo
commit do preflight aprovado. Nenhum treino, refit, calibração ou seleção é permitido.

## Modelos e hipótese descritiva

- SVM da Fase 2: pipeline salvo em `models/baseline/phase2-v1/`, C=0,1,
  ngrams=(1,2), class_weight=balanced, máximo 80.000 features, treino exclusivo.
- BERTimbau da Fase 3B: época 1/update 5543, cópia `best_model/` conferida contra
  `checkpoint-5543/`. Origem neuralmind/bert-base-portuguese-cased, revisão
  94d69c95f98f7d5b2a8700c420230ae10def0baa. Não usar a última época.

O BERTimbau **já é o candidato principal da futura demonstração**. O SVM é o
baseline comparativo. O teste não escolherá novamente modelo, checkpoint,
hiperparâmetro, threshold, pré-processamento ou arquitetura. Qualquer regressão
será relatada. Pergunta: a diferença descritiva observada na validação se mantém
no teste independente? Não há meta mínima de F1 nem teste de significância.

## Dados e integridade

B2W-Reviews01, fonte/revisão e fingerprints ancorados no resumo da Fase 1.
Licença original CC BY-NC-SA 4.0, sem redistribuição ou presunção de uso comercial.
Treino 88.684, validação 19.004, teste 19.004, em splits congelados e inalterados.
Classes/ordem fixas: 0 negativo (1–2 estrelas), 1 neutro (3), 2 positivo (4–5).
Estrelas são proxy de sentimento. Única feature: review_text.

Preflight confere existência do teste e metadados/referências, mas não seu
conteúdo, nem para checksum. Uma guarda de abertura bloqueia o arquivo real.
Depois dos testes, preflight aprovado e commit de congelamento, a CLI
`--execute-final` cria registro exclusivo antes de autorizar uma leitura do CSV.
O buffer dessa leitura fornece SHA-256, parsing e validação, sem reabrir o arquivo.
Validar esquema, nulos, normalização existente, IDs SHA-256, unicidade, ordem do
manifest, fingerprint, contagens/distribuição, disjunção com treino/validação e
fingerprint global. Falha interrompe sem corrigir os dados.

## Inferência e métricas

Ambos recebem exatamente os mesmos 19.004 textos na ordem original, com a mesma
verdade. SVM predict em CPU/uma thread, lotes 256, sem fit/fit_transform.
BERT local_files_only=True, trust_remote_code=False, cuda:0, eval(),
inference_mode(), FP16 autocast, atenção eager, TF32 desligado, duas threads CPU,
batch 2, max_length=256, truncamento e padding dinâmico ao múltiplo de 8.
Não normalizar novamente textos nem modificar o tokenizer cased.
Logits devem ser finitos, shape (N,3); conferir ordem/identidade das predições.
Sem fallback automático, repetição, backpropagation ou atualização de pesos.

Métrica principal: F1 Macro com peso igual nas três classes. Secundárias:
Precision/Recall Macro, Accuracy, Precision/Recall/F1 e suporte por classe,
matrizes (linhas reais, colunas previstas, ordem [0,1,2]), erros/total/fração.
Calcular diferença BERT−SVM absoluta e relativa (dividida pelo SVM, se não zero).
Examinar neutros recuperados, neutro→positivo/negativo e diferenças de P/R/F1
em todas as classes, sem atribuir causas que os agregados não demonstram.

Comparação pareada: ambos acertam, ambos erram, somente SVM e somente BERT.
**Não haverá bootstrap ou intervalo de confiança nesta fase.** A comparação é
descritiva, sem alegação automática de superioridade estatística.

## Custo computacional pré-definido

Antes do teste, benchmark em **1.024 textos de validação**, amostragem estratificada
determinística (seed 42), índices ordenados, mesmas entradas nos dois modelos.
32 textos de warm-up/modelo; três passagens completas, mediana e tempos individuais.
Batch SVM 256 e BERT 2; sincronização CUDA nas fronteiras de medição.
Inclui tokenização/TF-IDF, transferências e predição; exclui carga dos modelos,
I/O, métricas e aquecimento. Não medir latência individual: ms/texto é amortizado
em batch, throughput é textos/s e duração é passagem pelo conjunto especificado.

No teste, somente uma passagem/modelo, também cronometrada. Registrar picos de
VRAM PyTorch e NVML a cada 50 ms (inclui desktop/driver), RSS do processo Linux
amostrado a cada 10 ms (inclui modelos carregados; não é memória exclusiva do SVM),
hardware, versões e tamanhos dos artefatos. Medições CPU/GPU usam infraestruturas
diferentes e carga/temperatura não isoladas. Tempos históricos de treino vêm dos
relatórios anteriores; BERT inclui três épocas/validações/checkpoints, SVM inclui
ajuste do candidato vencedor. Não repetir treinamento para comparar custo.

## Execução definitiva e publicação de agregados

Registro local exclusivo em `outputs/phase3c-final-v1/started.json` impede segunda
execução mesmo após falha. Resultados locais e públicos são escritos atomicamente;
não sobrescrever resultados existentes. Falha após abertura gera registro técnico
sem textos/IDs e **não será recuperada automaticamente**. Uma recuperação futura
exigiria documentação própria, sem ajustes guiados por desempenho.

Uma execução concluída permite somente ler resultados e regenerar figuras.
Notebook 04 lê JSON agregado e nunca carrega dados ou modelos. Relatórios/figuras
públicos não incluem avaliações, IDs, predições individuais, dados pessoais ou
caminhos absolutos do computador. Preservar artefatos anteriores; não publicar
modelos/Hub, fazer push ou merge. Fase 4 permanece fora do escopo.
