# Continuidade da Fase 3B — execução concluída

Status em 30/10/2026: a execução `models/bertimbau/phase3b-full-v3/` terminou.
`results.json` existe, as três épocas foram consolidadas e **não se deve treinar
novamente ou retomar esta execução para obter os mesmos resultados**.

Branch: `feat/phase3b-bertimbau-local`, baseada em e42fb01 da main remota.
Não fazer merge automático, publicar no Hub ou acessar o teste final.

## Resultado confirmado

| Época | Update | F1 Macro de validação |
| --- | ---: | ---: |
| 1 | 5543 | 0,755522497920137 |
| 2 | 11086 | 0,741218100434194 |
| 3 | 16629 | 0,749668750254570 |

Melhor: **época 1**, `checkpoint-5543/`. A cópia `best_model/` foi recarregada com
sucesso offline em CPU e seus pesos conferidos por hash. O checkpoint completo
mais recente é `checkpoint-16629/`, com optimizer, scheduler, scaler, RNG e estado
do Trainer. `best_model/` serve para inferência, não para retomar updates.

Configuração: 256 tokens, batch treino/avaliação 2, acumulação 8, FP16, LR 2e-5,
seed 42, três épocas. Fonte original BERTimbau, revisão
94d69c95f98f7d5b2a8700c420230ae10def0baa. Nenhum split foi regenerado.

Treino/validações/checkpoints: 6.344,631 s. Picos PyTorch: 2,232/2,287 GiB
alocados/reservados. Pico amostrado driver: 4,118 GiB; livre mínimo: 1,882 GiB.
Três overflows isolados recuperados, sem loss/peso não finito, escala final 2.048.

## Documentos para uma próxima etapa

- Relatório final: `reports/validation_phase3b.md`.
- Métricas completas: `reports/bertimbau_phase3b_metrics.json`.
- Comprimentos: `reports/phase3b_token_lengths.json`.
- Piloto e prova de retomada: `reports/phase3b_gpu_pilot.json`.
- Guia de configuração, funções e reprodução: `docs/FASE3B_BERTIMBAU.md`.
- Notebook de inspeção, sem treino: `notebooks/03_bertimbau_finetune.ipynb`.
- `reports/phase3b_progress_snapshot.json` permanece como registro **histórico**
  da pausa anterior; não representa o status final.

As tentativas full-v1/full-v2 ficam preservadas para diagnóstico e não devem ser
retomadas. O runtime final do commit 0dd1ed9 foi validado novamente antes da full-v3.
Os arquivos de código/relatórios anteriores permanecem intactos; os hashes de
train.csv, val.csv, manifest.json e modelos anteriores foram reconferidos.
`test.csv` não foi aberto, lido, validado nem usado para checksum.

## Limites para continuar

A Fase 3B está concluída. Não adicionar épocas, tuning, avaliação do teste,
publicação ou merge como continuação implícita. A classe neutra continua sendo a
principal limitação e a seleção usa validação, não uma avaliação independente.
Antes de uma nova etapa, seguir o escopo explicitamente solicitado pelo usuário.
