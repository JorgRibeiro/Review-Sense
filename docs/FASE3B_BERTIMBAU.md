# Fase 3B — Comprimentos, piloto computacional e treino retomável

Esta fase mantém os splits da Fase 1 e utiliza somente treino/validação. O teste
final não é aberto nem para checksum. Os arquivos das fases anteriores não são
alterados. Modelos, tokens individuais, logs de processo e checkpoints ficam locais.

## Etapas e barreiras de execução

Instalação no ambiente CUDA já funcional:

```bash
.venv/bin/python -m pip install -r requirements-phase3b.txt
```

Isso mantém o lock Transformer e a constraint de PyTorch CUDA e acrescenta apenas
`nvidia-ml-py==13.615.71`, usado para medir a memória total da placa por NVML.
Nenhuma biblioteca existente foi atualizada. Com os dados congelados disponíveis,
uma reprodução em pastas novas segue:

```bash
.venv/bin/python -m reviewsense.phase3b prepare --features-dir cache/phase3b/replay
.venv/bin/python -m reviewsense.phase3b pilot --features-dir cache/phase3b/replay \
  --work-dir models/bertimbau/phase3b-pilot-replay
.venv/bin/python -m reviewsense.phase3b resume-proof --features-dir cache/phase3b/replay \
  --work-dir models/bertimbau/phase3b-pilot-replay \
  --resume-dir models/bertimbau/phase3b-resume-replay
.venv/bin/python -m reviewsense.phase3b accept \
  --work-dir models/bertimbau/phase3b-pilot-replay \
  --resume-dir models/bertimbau/phase3b-resume-replay
.venv/bin/python -m reviewsense.phase3b train --features-dir cache/phase3b/replay \
  --work-dir models/bertimbau/phase3b-pilot-replay \
  --run-dir models/bertimbau/phase3b-full-replay
```

Use pastas ausentes para novas execuções. O cache inicial requer os pesos originais
já disponíveis em `cache/huggingface/hub/`. A execução é local/offline, sem Colab,
tracking remoto, publicação no Hub ou acesso ao teste. Os workers usam processos
separados para isolar memória CUDA e estado do Accelerate.

`train` exige `acceptance.json`, com configuração aprovada, prova de retomada,
fingerprint do cache e hash do código de treino. Rejeita código/configuração/dados
diferentes e memória livre inicial insuficiente. Não inicia treino completo apenas
porque um batch coube na placa.

## Comprimentos antes do truncamento

Todos os 88.684 textos de treino e 19.004 de validação são tokenizados sem
truncamento ou padding, incluindo CLS/SEP. Os percentis usam interpolação linear;
as porcentagens são estritamente `comprimento > 128` e `> 256`. Só agregados entram
no Git. O cache guarda features truncadas em 256 e comprimentos individuais em
`cache/phase3b/`, ignorado. Caixa e acentos continuam preservados.

A preparação reutiliza `load_training_data`, conferindo fingerprints com o manifest
e a referência versionada da Fase 1. Não recria splits. Uma guarda de abertura
bloqueia o teste. Checksums dos três arquivos permitidos são verificados antes/depois.

## Piloto controlado

Três variantes fixas: 128/batch 2/accum 8; 256/batch 2/accum 8;
256/batch 4/accum 4. Batch efetivo sempre 16, FP16, LR 2e-5, AdamW, atenção eager.
Não há F1 ou seleção por qualidade no piloto.

Para cada variante, uma inicialização nova do checkpoint original e os mesmos:

- 256 exemplos do treino, cobrindo cinco faixas de comprimento; semente 42.
- 64 textos mais longos do treino: todos acima de 256 tokens antes do corte.
- 32 exemplos de aquecimento (dois updates) e duas passagens medidas por regime.

São 42 updates por variante se nenhum for pulado. O benchmark inclui forward,
backward, clipping e optimizer.step, com sincronização CUDA nas fronteiras.
Tokenização, carregamento de pesos e aquecimento ficam fora do throughput medido.
O padding é dinâmico; o regime longo força o comprimento máximo em todos os lotes.
A redução do cache para 128 preserva SEP e campos alinhados.

Os picos alocado/reservado do PyTorch abrangem inclusive inicialização do estado do
AdamW no aquecimento. NVML amostra a placa a cada 50 ms, incluindo outros processos
e driver. Seu máximo é amostrado; não é um pico contínuo exato. A margem mínima
exigida é **1 GiB livre pelo driver**, além de loss/gradientes finitos e zero updates
pulados no piloto. Falhas de monitoramento impedem aprovação.

A preferência é 256/batch 4; se não passar os critérios, 256/batch 2/accum 8.
Existe somente mais uma alternativa: 256/batch 1/accum 16 + gradient checkpointing,
com batch de avaliação 1. Ela só é medida se não houver configuração aprovada.
Não se expande a grade com base em métricas de qualidade.

### Decisão observada

Batch 4 coube, com aproximadamente 1,90 GiB livres pelo driver, mas teve um overflow
de gradiente protegido pelo GradScaler no regime longo. A configuração escolhida
foi **256/batch 2/accum 8**, que passou os critérios de memória e estabilidade.
A escolha não decorreu de uma diferença de F1, nem de OOM.

A escala FP16 padrão 65.536 produziu overflows tratados nas variantes de 256 tokens.
Foi feita uma única adaptação numérica: escala inicial 16.384 e intervalo de
crescimento de 1.000.000 updates (acima do orçamento deste treino), seguida da
mesma comparação fixa. Batch 2 ficou estável; batch 4 ainda teve um update pulado.
Uma rodada intermediária com repasse incorreto da escala foi descartada, sem usar
seus tempos na decisão. A escala efetiva é registrada nos resultados válidos.

## Retomada antes do treino completo

A prova usa 64 exemplos permitidos, incluindo textos longos, sem avaliação de F1:

1. Controle contínuo de quatro updates, scheduler planejado para quatro.
2. Outro processo interrompe após dois updates e salva o checkpoint completo.
3. Um processo novo retoma até quatro, com o mesmo scheduler e a ordem de dados.
4. Pesos, estado do AdamW, scheduler e scaler finais devem ser idênticos ao controle.

Os checkpoints incluem pesos, optimizer.pt, scheduler.pt, scaler.pt, rng_state.pth,
trainer_state.json e training_args.bin. O RNG inclui Python, NumPy, CPU e CUDA.
A posição dos dados é restaurada pelo Trainer com data_seed 42 e ignore_data_skip=False.
A prova não é uma simples recarga para inferência: verifica a continuação efetiva
dos updates. O gate compara também o código e os dados usados.

### Loss e acumulação

Na versão instalada, BERT calcula CE média por microbatch e ignora
num_items_in_batch na sua loss, embora sua assinatura aceite kwargs. Depender da
inferência automática do Trainer multiplicava o gradiente pela acumulação.
O pipeline fornece uma CE explícita: soma por microbatch dividida pelo número
real de exemplos no grupo acumulado. Assim, os grupos usuais usam 16; o último
grupo parcial de cada época usa 12, sem descartar exemplos. Um teste compara
loss/gradientes com o batch inteiro. A prova GPU foi refeita depois da correção,
antes de liberar o treino completo. A implementação da Fase 3A não foi alterada.

## Treinamento completo

Configuração computacional selecionada; LR 2e-5, até três épocas (orçamento desta
execução: três), sem busca adicional. A inicialização vem do checkpoint original,
não do modelo do smoke. A validação usa os 19.004 exemplos ao fim de cada época;
seleciona-se o checkpoint pelo maior F1 macro. Há métricas por classe e matriz de
confusão, sem leitura do teste final.

O Trainer gerencia AMP, scheduler linear, clipping e checkpoints. AdamW usa
foreach=False; bias e LayerNorm ficam sem weight decay pelo agrupamento padrão
do Trainer. Os demais parâmetros usam weight decay 0,01. Atenção eager e TF32
desativado mantêm as condições do piloto. A loss é verificada a cada microbatch. Loss/pesos não finitos interrompem. Overflows
isolados de gradientes são tratados pelo GradScaler: o update deve ser pulado,
a escala reduzida e os pesos permanecerem finitos. Três skips consecutivos ou mais
de 1% após mil updates interrompem por persistência. Eventos recuperados são
contados explicitamente; nenhuma norma fictícia substitui Infinity nos logs.

Checkpoints completos são salvos a cada 500 updates e ao fim das épocas. O limite
de dois preserva o melhor e o mais recente; apenas checkpoints desta nova execução
podem ser rotacionados. A seleção do melhor e a retomada cumprem propósitos distintos:

- `best_model/`: pesos e tokenizer do melhor checkpoint para inferência.
- `checkpoint-N/`: estado completo para retomar os updates.
- `phase3b_state.json` no checkpoint: configuração, posição, fonte e arquivos obrigatórios.
- `validation_history.json`, `progress.json`, `results.json`: métricas e andamento agregados.

Para retomar a execução original em uma pasta existente:

```bash
.venv/bin/python -m reviewsense.phase3b train \
  --resume models/bertimbau/phase3b-full-v2/checkpoint-N
```

Use o checkpoint completo mais recente que existe. O comando exige os mesmos
dados, configuração e gate. Não retome a partir de `best_model/`, pois ele não
contém o estado do otimizador. Arquivos joblib/pickle devem ser somente os locais
confiáveis produzidos por este pipeline.

## Testes e limites

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/pytest -q
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
```

Os testes não baixam pesos; a prova de retomada comum usa BERT minúsculo em CPU.
Além dela, a prova descrita acima foi executada com BERTimbau real em CUDA.

Validação usada para escolher épocas não é estimativa independente de generalização.
Tokens no limite não revelam, por si só, importância semântica do trecho cortado.
O benchmark curto não mede aquecimento térmico sustentado ou garante a mesma
margem se outros processos passarem a usar a GPU. Pesos e dados não são publicados.

Referências primárias: [checkpointing do Accelerate](https://huggingface.co/docs/accelerate/usage_guides/checkpoint)
e [acumulação de gradientes](https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation).

### Correção da supervisão AMP antes de concluir o treino

A execução inicial foi interrompida antes do primeiro checkpoint, após o último
log de 200 updates. O verificador considerava qualquer norma não finita fatal,
impedindo o mecanismo normal de recuperação do GradScaler. Essa supervisão foi
corrigida sem alterar LR, dados, batch ou época. Uma prova CUDA com overflow
intencional confirmou: update pulado sem alterar os pesos, escala 16.384→8.192 e
update seguinte finito alterando pesos normalmente. A retomada com BERTimbau foi
refeita em processos separados depois da correção. O treino final usa pasta nova;
a tentativa interrompida permanece local para diagnóstico.
