# Fase 3A — Diagnóstico e smoke do BERTimbau local

Esta fase confirma a viabilidade técnica de fine-tuning em uma RTX 3050 Laptop de
6 GB. Não é treinamento completo, busca de hiperparâmetros ou avaliação definitiva.
Só utiliza **256 exemplos de treino e 96 de validação**. Não utiliza Colab, não
abre `test.csv`, não altera splits e não publica pesos ou dados no Hugging Face.

## Ambiente e dependências

O ambiente já possui Python 3.11 e **PyTorch 2.14.1+cu132**, com CUDA 13.2 funcional.
A instalação do PyTorch é preservada. Os extras Transformer são opcionais: os
módulos das Fases 1 e 2 continuam sem exigir Transformers para seus comandos.

Na raiz do repositório:

```bash
.venv/bin/python -m pip install -r requirements-transformer-lock.txt
.venv/bin/python -m pip check
.venv/bin/python -m reviewsense.train_transformer --diagnose-only
```

`constraints-transformer.txt` fixa o Torch já instalado, impedindo que o pip o
substitua por uma variante CPU. Em uma máquina nova, instale primeiro a versão
CUDA apropriada e confirme sua disponibilidade; este arquivo é específico do
ambiente local validado. `requirements-transformer.txt` oferece os intervalos do
extra; o arquivo `requirements-transformer-lock.txt` repete as versões usadas e
inclui o lock anterior sem modificá-lo. As adições são registradas em
`reports/phase3a_dependency_changes.json`. Não foram atualizados pacotes existentes.

O diagnóstico executa uma operação real na GPU sob autocast FP16, registra nome,
compute capability, memória visível/livre, versão CUDA, cuDNN e driver. CUDA ausente
causa erro explícito, sem treinamento silencioso na CPU. Se a operação FP16 não for
suportada, o pipeline registra a mudança para FP32, ainda em CUDA.

## Execução

```bash
.venv/bin/python -m reviewsense.train_transformer \
  --artifacts-dir data/processed/phase1-final \
  --reference reports/data_audit_summary.json \
  --config configs/bertimbau_smoke.json \
  --output-dir models/bertimbau/phase3a-v1 \
  --cache-dir cache/huggingface/hub
```

Depois do primeiro download, use `--local-files-only` para exigir cache local.
Use outra saída para repetir (`phase3a-v2`, por exemplo): diretórios existentes não
são sobrescritos. Execute normalmente com Python, sem `accelerate launch`; o smoke
exige um único processo em uma GPU. Nenhuma conta ou token do Hub é necessário
para o checkpoint público; não há integração de tracking ou publicação remota.

Os limites de `SmokeConfig` recusam amostras maiores que 256/96 e mais de uma época.
A expansão da Fase 3B será uma mudança explícita futura, sem remover as proteções
contra acesso ao teste ou adulteração dos dados.

## Dados e tokenização

`load_training_data` reutiliza a verificação de fingerprints da Fase 2 e abre
somente treino/validação. Uma guarda adicional interrompe qualquer tentativa de
abrir o CSV de teste. Checksums de train.csv, val.csv e manifest.json são verificados
antes/depois; o teste não é lido sequer para calcular seu checksum.

Cada amostra é estratificada, sem reposição, com semente 42, a partir do respectivo
split original. A ordenação por sample_id torna a escolha independente da ordem
das linhas. Os dados de origem não são modificados. Fingerprints e contagens por
classe entram no relatório; IDs individuais ficam somente no manifest local.

`AutoTokenizer` usa `do_lower_case=False` e `strip_accents=False`. Os textos já
normalizados da Fase 1 são fornecidos sem limpeza adicional. O tokenizer cased
preserva distinções de capitalização e acentuação. Truncamento em 128 tokens limita
o consumo; padding é feito por lote, até múltiplos de oito, pelo
`DataCollatorWithPadding`. A fração que chega ao limite é registrada como indício
para a Fase 3B, não como taxa exata de truncamento.

## Modelo e treino

Fonte: `neuralmind/bert-base-portuguese-cased`, revisão fixada
`94d69c95f98f7d5b2a8700c420230ae10def0baa`. O checkpoint original tem pesos PyTorch
em `pytorch_model.bin`; a carga usa esse arquivo explicitamente, sem conversão
remota. `trust_remote_code=False`. O salvamento local usa Safetensors.

`AutoModelForSequenceClassification` carrega o encoder pré-treinado e cria uma
cabeça para três classes. O aviso de inicialização dessa cabeça é esperado: ela
não existe como classificador de polaridade no modelo base. Todos os parâmetros
são treináveis. Os mapas são os mesmos das fases anteriores:
`0=negativo`, `1=neutro`, `2=positivo`, com `id2label` e `label2id` explícitos.

| Escolha inicial | Valor | Propósito |
| --- | --- | --- |
| Comprimento | 128 | Limitar memória de ativações |
| Batch de treino/avaliação | 2 / 2 | Ser conservador na placa de 6 GB |
| Acumulação | 8 | Batch efetivo 16 sem manter 16 exemplos simultâneos na GPU |
| Épocas | 1 | Diagnóstico limitado a um ciclo |
| Learning rate | 2e-5 | Ponto inicial fixo, sem tuning |
| Precisão | FP16 com AMP | Reduzir memória/custo das operações elegíveis |
| Otimizador | AdamW, weight decay 0,01, foreach=False | Estado em FP32 e menos temporários |
| Scheduler | Decaimento linear, sem warmup | Orçamento de passos explícito |
| Gradient clipping | 1,0 | Limitar norma de gradientes |
| Atenção | eager | Execução simples, sem kernels opcionais |

Accelerate cuida de autocast, GradScaler, transferência para CUDA e acumulação.
Loss deve ser finita. Atualizações efetivas e eventuais passos pulados pelo scaler
são registrados; o hash da cabeça deve mudar para comprovar atualização dos pesos.
A avaliação ocorre apenas ao terminar a época, sem escolher o melhor checkpoint.

## CUDA Out of Memory

No máximo **duas tentativas**, sem alteração da amostra, learning rate ou número
de épocas. Só `torch.cuda.OutOfMemoryError` permite a alternativa:

1. Inicial: batch 2, acumulação 8, sem gradient checkpointing.
2. Se houver OOM: batch de treino/avaliação 1, acumulação 16 e gradient checkpointing.

Gradient checkpointing recalcula ativações para economizar memória, com custo de
tempo. Antes de repetir, referências do traceback são liberadas e o cache CUDA é
limpo. Se a segunda tentativa falhar, o comando termina com erro e um diagnóstico
local de OOM. Erros de dependências, dados ou programação não iniciam outras tentativas.
O caminho alternativo é verificado por testes simulados, sem forçar OOM na GPU real.

## Evidências e artefatos

Tempos de treino e avaliação usam relógio de parede e sincronização CUDA nas
fronteiras. Incluem collator, transferências e operações do loop; tokenização,
download, métricas e escrita de arquivos ficam fora dessas duas medições. O tempo
total inclui preparação, recarga e I/O. Um snapshot nvidia-smi durante o treino
acrescenta evidência de utilização, e seu pequeno custo entra no tempo de treino.

Os picos de memória alocada e reservada são reiniciados por etapa. Memória alocada
é ocupada por tensores; reservada também inclui espaço que o allocator guarda
para reutilização. Esses números não representam toda a VRAM da placa: o desktop,
o driver e outros processos também consomem memória. A avaliação mantém o modelo
e o estado do otimizador; o cache livre é esvaziado antes da medição. A recarga
ocorre após liberar o modelo/otimizador anteriores.

Depois de salvar, modelo e tokenizer são carregados somente do disco. A tokenização
da amostra de validação deve ser idêntica, os mapas são conferidos e as predições
devem permanecer iguais. Os logits são comparados com tolerância rtol/atol de 1e-4.

Em `models/bertimbau/phase3a-v1/` (ignorado pelo Git):

- `attempt-1/checkpoint/`: pesos Safetensors, configuração do modelo e tokenizer
  (ou `attempt-2/` caso tenha sido necessária a alternativa).
- `samples_manifest.json`: IDs individuais das duas amostras, somente locais.
- `config.json` e `results.json`: configuração efetiva e diagnóstico completo.

O cache fica em `cache/huggingface/hub/`, também ignorado. Não se grava o estado do
otimizador para retomada: este checkpoint é para inferência e continuação a partir
dos pesos, não para reproduzir exatamente o meio de um treino interrompido.
A licença do checkpoint base é MIT; o corpus mantém a declaração original
CC BY-NC-SA 4.0. Não se redistribuem modelos, vocabulários ou dados nesta fase.

## Funções e testes

| Função | Responsabilidade |
| --- | --- |
| `SmokeConfig.validate` | Impedir expansão acidental do orçamento do smoke |
| `make_smoke_samples` | Amostrar treino/validação sem misturar suas origens |
| `tokenize_reviews` | Preparar entradas sem alterar caixa ou acentos |
| `diagnose_cuda` | Confirmar CUDA/FP16 e registrar o dispositivo |
| `train_attempt` | Executar treino, avaliação, salvamento e recarga |
| `execute_with_oom_retry` | Limitar OOM a uma alternativa documentada |
| `run_smoke` | Proteger dados, coordenar a execução e persistir evidências |

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/pytest -q
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
```

A suíte não baixa BERTimbau nem exige GPU. Testes com Torch/Transformers usam um
vocabulário local e um BERT minúsculo em CPU; são ignorados se os extras não
estiverem instalados. Configuração e amostragem usam apenas as dependências base.
Não há comparação de desempenho definitivo com o SVM: 256/96 exemplos servem apenas
para verificar funcionamento, memória e registro dos resultados.

Referências: [checkpoint original](https://huggingface.co/neuralmind/bert-base-portuguese-cased),
[Accelerate e acumulação](https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation),
[classificação com Transformers](https://huggingface.co/docs/transformers/tasks/sequence_classification).
