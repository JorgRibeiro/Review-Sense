# Continuidade da Fase 3B

O usuário encerrou a sessão por limite de uso e pediu um resumo para continuar
posteriormente com os resultados das épocas 2 e 3. O treino NÃO foi interrompido.
O objetivo original continua: concluir até três épocas, selecionar pelo F1 macro
de validação e consolidar medições, métricas por classe e retomada.

Branch: `feat/phase3b-bertimbau-local`, baseada na revisão e42fb01 da main remota.
Não alterar main, fazer merge automático ou publicar no Hub.

## Processo/execução

- Execução final: `models/bertimbau/phase3b-full-v3/`.
- Log: `models/bertimbau/phase3b-full-v3-train.log`.
- Últimos PIDs observados: controlador 84664, worker 84743. Podem já ter terminado;
  não enviar sinais a esses números sem conferir a linha de comando atual.
- Sessão exec observada: 91818 (pode não estar disponível na nova sessão).
- Início registrado: `models/bertimbau/phase3b-full-v3-launch.json`.
- `progress.json`: andamento e contadores numéricos.
- `validation_history.json`: métricas completas por época.
- `results.json`: só é gravado após conclusão, seleção e nova predição da validação.

Antes de fazer qualquer treino, verificar esses arquivos e os processos. Se o
results.json existir, NÃO treinar de novo. Se ainda ativo, apenas acompanhar.

Se parado antes de completar, localizar o checkpoint de maior número que contenha
phase3b_state.json e todos os arquivos de estado. Não fixar checkpoint-7000:
a retenção pode removê-lo. Retomar na mesma pasta, configuração e gate:

```bash
.venv/bin/python -m reviewsense.phase3b train \
  --resume models/bertimbau/phase3b-full-v3/checkpoint-N
```

O default atual aponta a full-v3 e o piloto-v3. O gate está em
`models/bertimbau/phase3b-pilot-v3/acceptance.json`. A prova de retomada atual está
em `models/bertimbau/phase3b-resume-v4/verification.json`.

## Decisões já fechadas

- Fonte original BERTimbau, revisão 94d69c95f98f7d5b2a8700c420230ae10def0baa.
- max_length 256, batch treino/avaliação 2, acumulação 8, batch efetivo 16.
- FP16, escala inicial 16.384, growth_interval 1.000.000; backoff dinâmico padrão.
- LR inicial 2e-5, três épocas, AdamW foreach=False, linear sem warmup, clipping 1.
- Validação completa por época; escolher maior F1 macro. Não fazer novas buscas.
- Checkpoints completos a cada 500 updates e por época; manter melhor + mais recente.
- Batch 4 coube, mas apresentou overflow no piloto; batch 2 foi a escolha conservadora.
- Função explícita de CE normaliza pelo número real de exemplos no grupo acumulado,
  corrigindo a interação BERT/Trainer 5.22 com num_items_in_batch.
- Overflows AMP isolados devem pular update, reduzir escala e manter pesos finitos.
  Não tratar todo overflow como fatal; também não descalar duas vezes. Há teste CUDA
  dentro do Trainer que comprovou esse fluxo e teste de regressão offline.
- Três skips consecutivos ou mais de 1% após mil updates interrompem por persistência.

As full-v1/full-v2 pararam antes do primeiro checkpoint por supervisão AMP/dupla
unscale. Permanecem para diagnóstico e NÃO devem ser retomadas. A full-v3 usa o
runtime corrigido do commit 0dd1ed9, validado novamente pela prova de retomada GPU.

## Já obtido

Comprimentos completos: treino P50/P90/P95/P99 = 26/69/95/177; acima128 = 2,3522%,
acima256 = 0,3586%. Validação = 26/71/98/181,97; acima128 = 2,4469%, acima256 = 0,4157%.

Época 1, update 5543: F1 macro 0,7555224979, precision macro 0,7715705073,
recall macro 0,7473980979, accuracy 0,8636602820. F1 negativo 0,8976567194,
neutro 0,4479571984, positivo 0,9209535759. Não são resultados do teste final.

Snapshot completo em `reports/phase3b_progress_snapshot.json`, evidências do piloto
em `reports/phase3b_gpu_pilot.json` e comprimentos em `reports/phase3b_token_lengths.json`.

## Finalização pendente

- Aguardar/obter épocas 2 e 3, selecionar checkpoint pelo F1 macro e verificar estados.
- Consolidar `reports/validation_phase3b.md`, substituindo o status parcial;
  gerar `reports/bertimbau_phase3b_metrics.json` e figuras agregadas.
- Existe gerador pronto em `/tmp/reviewsense_phase3b_final_report.py`; verificar se
  ainda está presente. Usa a full-v3 e confere métricas, hashes, fontes e três épocas.
  Se tiver sido removido, reconstruir a partir dos resultados locais completos.
- Verificar checksums anteriores em
  `models/bertimbau/phase3b-pilot-v3/source_checksums_before_full.json`.
- Atualizar README e plano para a conclusão real, executar notebook
  `notebooks/03_bertimbau_finetune.ipynb` sem repetir treino e manter arquivo sem outputs.
- Rodar testes/checks finais apropriados; última suíte: 143 testes offline aprovados.
- Fazer commit dos resultados/documentação, mantendo pesos, cache, dados e IDs fora do Git.
- Preservar integralmente Fases 1, 2 e 3A. Não abrir/ler/validar test.csv nem para checksum.
