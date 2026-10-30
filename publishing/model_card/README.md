---
language: pt
library_name: transformers
pipeline_tag: text-classification
base_model: neuralmind/bert-base-portuguese-cased
base_model_relation: finetune
license: unknown
datasets:
- ruanchaves/b2w-reviews01
tags:
- pt-br
- nlp
- sentiment-analysis
- text-classification
- e-commerce
- reviewsense
- license-under-review
model-index:
- name: ReviewSense PT-BR BERTimbau v1
  results:
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      type: ruanchaves/b2w-reviews01
      name: B2W-Reviews01 / ReviewSense frozen test (official CSV)
      split: test
    metrics:
    - type: f1
      name: F1 Macro
      value: 0.7544611794363947
    - type: precision
      name: Precision Macro
      value: 0.7704919166001464
    - type: recall
      name: Recall Macro
      value: 0.7462112828169135
    - type: accuracy
      name: Accuracy
      value: 0.8633971795411493
    source:
      name: ReviewSense Phase 3C — independent final evaluation
      url: https://github.com/JorgRibeiro/Review-Sense/blob/5441d02/reports/phase3c_final_metrics.json
---

# ReviewSense PT-BR — BERTimbau para polaridade de avaliações

> **Candidato local `1.0.0-rc.1`, não publicado. Licenciamento dos pesos:
> `REQUIRES_REVIEW`; upload não autorizado.** `license: unknown` descreve uma
> pendência, não uma concessão de direitos. Veja [DISTRIBUTION_STATUS.md](DISTRIBUTION_STATUS.md).

## English summary

ReviewSense PT-BR fine-tunes BERTimbau Base for three-class sentiment classification
of Brazilian Portuguese product reviews. Labels are derived from star ratings,
not manual sentiment annotations. The selected checkpoint achieved independent
test macro F1 0.7544611794363947 on 19,004 examples. Results are descriptive;
no statistical superiority is claimed. Fine-tuned weight distribution requires
licensing review. No weights have been published by this preparation phase.

## Visão geral

Projeto educacional e de engenharia de ML/NLP: classificação de polaridade em
português brasileiro. Arquitetura BERT Base cased, 12 camadas e dimensão oculta 768,
com cabeça supervisionada de três classes. Base NeuralMind
[`neuralmind/bert-base-portuguese-cased`](https://huggingface.co/neuralmind/bert-base-portuguese-cased),
revisão `94d69c95f98f7d5b2a8700c420230ae10def0baa`. O pré-treinamento do base e o fine-tuning
do ReviewSense são etapas diferentes. O pacote contém somente o BERTimbau final;
o SVM é uma referência comparativa e não acompanha os pesos.

## Dados

B2W-Reviews01, da B2W Digital/Americanas Tech: **132.373** avaliações de produtos
coletadas na Americanas.com de janeiro a maio de 2018. Fonte usada:
[CSV original, revisão fixada](https://github.com/americanas-tech/b2w-reviews01/tree/4639429ec698d7821fc99a0bc665fa213d9fcd5a).
SHA-256 original: `821fb0bf9f7230b0fba4e4f9fadd75a66d1a9ff0b1657810791d33007eb2ab38`.

Preparação congelada: Unicode NFC, controles ASCII/espaços normalizados; acentos,
caixa, negações e pontuação preservados. Foram removidos 3.275 textos ausentes/vazios,
43 linhas de grupos com conflito de classe e 2.363 duplicatas adicionais de texto,
restando **126.692** textos. Somente review_text foi usado como feature. Metadados
pessoais, estrelas e variáveis associadas ao alvo não foram entradas dos modelos.

Divisão estratificada determinística, seed 42, aproximadamente 70/15/15:
**88.684 treino, 19.004 validação, 19.004 teste**. Textos idênticos normalizados não
se sobrepõem; possíveis dependências de autores/produtos e quase duplicatas não
foram eliminadas exaustivamente. Fingerprint do teste, sem reabri-lo nesta fase:
`3c2c5d9cf600821a8fdf7be5fc630c621bb4453616fdffc5d45d69baf40a5a57`.

A licença declarada pela fonte original é **CC BY-NC-SA 4.0**. O identificador
`datasets` no YAML aponta ao [espelho do mesmo corpus no Hub](https://huggingface.co/datasets/ruanchaves/b2w-reviews01)
apenas como referência de catálogo. Sua declaração CC BY 4.0 é divergente e não
foi adotada como autorização. Os splits locais ReviewSense não são os splits
padrão do espelho. Nenhuma avaliação do corpus é redistribuída neste pacote.

## Rótulos

| Estrelas de origem | ID | Classe |
| --- | ---: | --- |
| 1–2 | 0 | negativo |
| 3 | 1 | neutro |
| 4–5 | 2 | positivo |

Esses alvos são **proxy por estrelas**, não sentimentos anotados manualmente por
especialistas. Avaliações com opiniões mistas podem não corresponder a uma única
polaridade textual, mesmo quando a nota é conhecida.

## Treinamento

Tokenizer cased original salvo, sequência máxima **256 tokens**, padding dinâmico
múltiplo de 8, batch de treino 2, acumulação 8, batch efetivo alvo 16 (último grupo
parcial de cada época com 12). LR inicial **2e-5**, AdamW, weight decay 0,01 com
exceções de bias/LayerNorm, scheduler linear sem warmup e clipping 1,0. Seed 42,
FP16, três épocas completas. Overflows AMP isolados recuperados estão documentados
na Fase 3B; não houve calibração probabilística.

Selecionado **checkpoint da época 1, update 5543**, pelo maior F1 Macro na validação
completa, antes de abrir o teste. F1 Macro de validação por época: 0.755522497920137 / 0.7412181004341937 / 0.7496687502545695.
Hardware: NVIDIA RTX 3050 Laptop 6 GB; Python 3.11.14, PyTorch 2.14.1+cu132,
Transformers 5.22.1, Accelerate 1.15.0 e CUDA 13.2.

## Avaliação independente

Fase 3C: protocolo e código congelados em `90893aa`, primeira avaliação real de
**19.004 exemplos** em uma execução definitiva, uma leitura do CSV e uma passagem
por modelo, com a mesma ordem e rótulos. O BERTimbau já era o candidato principal;
o teste não selecionou novamente checkpoint, hiperparâmetros ou arquitetura.
Nenhum refit em treino+validação. Nenhuma nova leitura do teste na preparação do pacote.

Valores exatos dos registros estruturados, sem recálculo de métricas:

| Modelo | F1 Macro | Precision Macro | Recall Macro | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF + LinearSVC | 0.7230938667235973 | 0.7228302528752341 | 0.7262901588268904 | 0.832508945485161 |
| BERTimbau | 0.7544611794363947 | 0.7704919166001464 | 0.7462112828169135 | 0.8633971795411493 |

| Classe BERT | Precision | Recall | F1 | Suporte |
| --- | ---: | ---: | ---: | ---: |
| negativo | 0.8795749704840614 | 0.9141104294478528 | 0.8965102286401926 | 4890 |
| neutro | 0.5273459988485896 | 0.38552188552188554 | 0.44541697058108437 | 2376 |
| positivo | 0.9045547804677883 | 0.9390015334810019 | 0.921456339087907 | 11738 |

Diferença absoluta de F1 Macro BERT−SVM: **0.03136731271279736**;
relativa: **0.04337930959769503** (aproximadamente
+4,34%). Comparação descritiva, sem bootstrap, intervalo de confiança ou demonstração
de superioridade estatística. Baseline: C=0,1, ngrams=(1,2), class_weight=balanced,
80.000 features. A classe neutra segue limitante: Recall 0.38552188552188554.
Não houve validação em datasets de delivery/restaurantes ou de outros domínios.

Fonte da verdade: [JSON imutável da Fase 3C](https://github.com/JorgRibeiro/Review-Sense/blob/5441d02/reports/phase3c_final_metrics.json),
SHA-256 `8f32d5fa4dccd74b47f2b6c26455372bf7d50415390548eaa2085ed9e21bf302`.
[Relatório técnico](https://github.com/JorgRibeiro/Review-Sense/blob/5441d02/reports/validation_phase3c.md).

## Desempenho computacional

Treino completo full-v3: **6344.631124032001 s**, aproximadamente **105,74 min**,
incluindo validações e checkpoints. Esse orçamento inclui três épocas, apesar de
o checkpoint da primeira ter sido escolhido. Não é custo universal de treinamento.

No smoke da Fase 4A, a mediana aquecida do serviço foi aproximadamente **6,81 ms em
RTX 3050/FP16** e **31,57 ms em CPU/FP32**. Carga inicial GPU: aproximadamente
**3,40 s**. Condições: batch 1, 12 requisições por cenário, três passagens por quatro
exemplos sintéticos, duas threads intraoperador PyTorch. Exclui fila, transporte
HTTP, renderização e carga (medida separadamente). A medição CPU veio depois da
GPU e reutilizou bibliotecas já importadas. Não houve isolamento térmico/de carga.

São latências individuais, distintas do throughput em batch da Fase 3C. Não são
SLAs ou garantias de produção. [Condições e valores completos da Fase 4A](https://github.com/JorgRibeiro/Review-Sense/blob/d05921c/reports/phase4a_inference_smoke.json).

## Limitações

- Baixo Recall da classe neutra; ambiguidades, sentimentos mistos, ironia/sarcasmo
  não foram avaliados como subgrupos específicos.
- Alvos por estrelas, desbalanceamento e avaliações de produtos de 2018.
- Mudança temporal/de domínio; falta de validação específica para delivery.
- Possíveis dependências entre autores/produtos e quase duplicatas entre splits.
- Truncamento à direita acima de 256 tokens; trechos posteriores podem ser relevantes.
- Softmax não calibrado: score normalizado não é certeza ou confiança calibrada.
- Memorização ou exposição de características dos dados não foi exaustivamente auditada;
  não se afirma ausência de vazamento, dados pessoais ou vieses nos pesos.
- O teste foi observado uma vez e não deve orientar novos ajustes neste experimento.

## Uso responsável

Uso pretendido: estudo e demonstração de engenharia/NLP, com avaliações sintéticas
ou dados sobre os quais o usuário tenha direitos. Uso em outro domínio exige
avaliação específica. Não é recomendado para decisões automatizadas de alto impacto,
punição de consumidores/trabalhadores ou decisões sem revisão contextual.
Não é um sistema validado de satisfação de delivery nem uma parceria institucional
com Americanas, iFood ou NeuralMind.

Pesos ajustados: **REQUIRES_REVIEW; publicação não autorizada**. MIT do código e
MIT do base não autorizam automaticamente os novos pesos. A incidência das condições
CC BY-NC-SA sobre treinamento/distribuição depende de análise jurídica e do caso.
Veja [atribuições](ATTRIBUTIONS.md) e os avisos de licença de escopo específico.

## Inferência local

Disponibilize um diretório do pacote local validado e defina
`REVIEWSENSE_MODEL_DIR` com esse diretório. Não há ID público do modelo nesta fase.
O exemplo é autocontido e não importa o pacote reviewsense nem lê relatórios/datasets.
Os arquivos têm os bytes originais; especifique **truncation=True, max_length=256**
explicitamente. O limite posicional 512 do BERT não muda o orçamento usado no projeto.
O exemplo usa FP32; a aplicação também suporta autocast FP16 em GPU compatível.

```python
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_dir = Path(os.environ["REVIEWSENSE_MODEL_DIR"])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(
    model_dir, local_files_only=True, trust_remote_code=False, token=False
)
model = AutoModelForSequenceClassification.from_pretrained(
    model_dir, local_files_only=True, trust_remote_code=False, token=False,
    attn_implementation="eager",
).to(device).eval()
labels = {0: "negativo", 1: "neutro", 2: "positivo"}
assert model.config.id2label == labels
assert model.config.label2id == {name: index for index, name in labels.items()}
assert tokenizer.do_lower_case is False
inputs = tokenizer(
    "O produto chegou rápido e funciona bem.", truncation=True, max_length=256,
    padding=True, pad_to_multiple_of=8, return_tensors="pt",
).to(device)
with torch.inference_mode():
    logits = model(**inputs).logits.float()
    assert logits.shape == (1, 3) and torch.isfinite(logits).all()
    scores = torch.softmax(logits, dim=-1)[0].cpu().tolist()
class_id = max(range(3), key=scores.__getitem__)
print(json.dumps({
    "class_id": class_id, "label": labels[class_id],
    "scores_normalized_uncalibrated": {labels[i]: scores[i] for i in range(3)},
}, ensure_ascii=False))
```

Os textos do exemplo são sintéticos. O snippet foi executado localmente contra o
pacote isolado na verificação da Fase 4B; os scores produzidos não são calibrados.

## Referências

- Fábio Souza, Rodrigo Nogueira e Roberto Lotufo (2020), *BERTimbau: Pretrained BERT Models for Brazilian Portuguese*. [Repositório NeuralMind e citação](https://github.com/neuralmind-ai/portuguese-bert).
- B2W Digital/Americanas Tech, [B2W-Reviews01](https://github.com/americanas-tech/b2w-reviews01), fonte de dados e declaração de licença original.
- [Transformers: classificação de sequências](https://huggingface.co/docs/transformers/tasks/sequence_classification).
- [ReviewSense no GitHub](https://github.com/JorgRibeiro/Review-Sense), código de inferência, treinamento e protocolos.
- [Model Card do base na revisão usada](https://huggingface.co/neuralmind/bert-base-portuguese-cased/blob/94d69c95f98f7d5b2a8700c420230ae10def0baa/README.md).
