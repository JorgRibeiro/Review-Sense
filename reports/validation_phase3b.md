# Fase 3B — Estado parcial para continuidade

Snapshot: **2026-10-30T13:50:39.128899-03:00**. Branch `feat/phase3b-bertimbau-local`.
**Fase ainda em andamento**: não confundir este snapshot com a conclusão das três épocas.
O processo de treino foi deixado em execução; não foi interrompido a pedido do usuário.

## Concluído e validado

- Comprimentos de todos os 88.684 textos de treino e 19.004 de validação, sem truncamento.
- Piloto fixo 128/batch 2, 256/batch 2 e 256/batch 4, com representativos e longos.
- Batch 4 coube em VRAM, mas teve overflow tratado no piloto. Escolhido 256/batch 2/acumulação 8.
- FP16, LR inicial 2e-5, batch efetivo 16, três épocas e seleção pelo F1 macro de validação.
- Prova CUDA de interrupção/retomada: pesos e estados finais de optimizer/scheduler/scaler idênticos.
- Normalização explícita de loss para acumulação; recuperação AMP integrada ao Trainer.
- 143 testes offline aprovados; lint, formatação e pip check aprovados.
- Nenhum acesso ao teste final, alteração de splits ou publicação de dados/modelos.

## Resultado da primeira época

| F1 macro | Precision macro | Recall macro | Accuracy |
| ---: | ---: | ---: | ---: |
| 0.755522 | 0.771571 | 0.747398 | 0.863660 |

F1 por classe: negativo 0.897657, neutro
0.447957, positivo 0.920954.
Checkpoint da época: `checkpoint-5543`.

No snapshot: update **7500 / 16629**, época
**1.3531**. Escala FP16 4096.0; dois overflows isolados
recuperados, zero losses/pesos não finitos. Checkpoints completos naquele instante:
checkpoint-5543, checkpoint-7500. A rotação pode mudar esses nomes depois.

## Continuar na próxima sessão

1. Ler `docs/CONTINUAR_FASE3B.md` e verificar se os processos ainda estão ativos.
2. Se `models/bertimbau/phase3b-full-v3/results.json` existir, as três épocas terminaram;
   não repetir treinamento. Inspecionar histórico e consolidar segunda/terceira épocas.
3. Se ainda estiver ativo, acompanhar progress.json e validation_history.json.
4. Se estiver parado antes de concluir, identificar o checkpoint completo de maior N
   que ainda existe e retomar com a mesma configuração/gate. Não usar um nome antigo
   sem verificar, pois a retenção mantém apenas melhor e mais recente.
5. Ao terminar: executar `/tmp/reviewsense_phase3b_final_report.py` se disponível (gera
   relatório final, JSON agregado e figuras), ou reconstruir a consolidação pelos dados
   salvos. Atualizar README/plano, executar notebook 03, checks finais e commit de resultados.
6. Não abrir test.csv, alterar splits, fazer tuning adicional, publicar no Hub ou fazer merge automático.

Arquivos de evidência atuais:
[phase3b_token_lengths.json](phase3b_token_lengths.json),
[phase3b_gpu_pilot.json](phase3b_gpu_pilot.json),
[phase3b_progress_snapshot.json](phase3b_progress_snapshot.json).
