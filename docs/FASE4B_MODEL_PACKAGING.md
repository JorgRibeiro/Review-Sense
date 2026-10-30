# Fase 4B — Pacote local verificável do BERTimbau

O candidato `1.0.0-rc.1`, identidade `reviewsense-ptbr-bertimbau-v1`, prepara somente
o BERTimbau da época 1/update 5543. O SVM continua como baseline documentado.
**Licenciamento dos pesos: REQUIRES_REVIEW; upload_authorized: false.**
Não há uploader, criação de repositório, login, treinamento ou avaliação de dataset.

## Entradas e seleção dos arquivos

Os quatro artefatos são lidos de `models/bertimbau/phase3b-full-v3/best_model/`.
Seus hashes são ancorados nos relatórios imutáveis das Fases 3B e 3C. A referência
da Fase 3C também é conferida contra seu recibo de verificação. O peso esperado é
obtido dos registros, não de um valor digitado na linha de comando.

O pacote contém exatamente:

```text
outputs/phase4b/model-repo-staging/
  model.safetensors
  config.json
  tokenizer.json
  tokenizer_config.json
  README.md
  BASE_MODEL_LICENSE.txt
  PROJECT_CODE_LICENSE.txt
  ATTRIBUTIONS.md
  DISTRIBUTION_STATUS.md
```

Os cinco documentos têm origem em `publishing/model_card/`. Os avisos MIT têm
escopos separados; não licenciam automaticamente os pesos fine-tuned. Não é criado
um LICENSE genérico para todo o pacote. README declara `license: unknown`, valor
previsto na [documentação de licenças do Hub](https://huggingface.co/docs/hub/repositories-licenses).

`training_args.bin` é reconhecido e excluído na origem. Outros arquivos extras,
ausências, links simbólicos (inclusive em ancestrais) e travessia de diretórios
são rejeitados. A cópia usa somente a allowlist, arquivos novos e proteção contra
seguir links no arquivo aberto. Não usa save_pretrained, conversão ou quantização.
O staging temporário é validado antes da renomeação; uma saída existente não é
substituída. Datasets, estados de treino, SVM, cache, Git e credenciais não entram.

## Preparação reproduzível

Usar a `.venv` existente; nenhuma dependência nova é necessária. Faça commit do
código, Model Card e avisos antes de preparar um candidato real. Então:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
  -m reviewsense.model_packaging --prepare
```

Gera o staging ignorado pelo Git e o manifest técnico
`reports/phase4b_package_manifest.json`. O manifest permanece fora do pacote.
Contém versão, identidade, commit, origem/revisão, arquivos/tamanhos/SHA-256,
fingerprint de conteúdo, labels, max_length, software, métricas agregadas congeladas
e bloqueio de publicação. Não contém caminhos absolutos, namespace HF inventado,
timestamp variável ou dados individuais. O repo_id definitivo fica nulo até a 4C.

Não repetir sobre a saída existente. Para uma reprodução em destino novo, após
commit das alterações locais, use explicitamente outras saídas:

```bash
.venv/bin/python -m reviewsense.model_packaging --prepare \
  --destination outputs/phase4b/reproduction/model \
  --manifest outputs/phase4b/reproduction/manifest.json
```

Mesmos bytes, software e proveniência geram o mesmo manifest; não se inserem datas
de build. Um commit de referência diferente fica registrado como tal. Nenhum comando
recria splits, altera checkpoint original ou escreve no marcador final da Fase 3C.

## Verificação independente

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
  -m scripts.verify_phase4b_package --device cpu --precision fp32

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
  -m scripts.verify_phase4b_package --device cuda --precision fp16
```

Cada verificação usa cinco entradas sintéticas, incluindo texto longo que força
truncamento/padding em 256 tokens. Não calcula F1. Compara o original com o pacote
no **mesmo dispositivo e mesma precisão**: classes, logits finitos e scores
normalizados. Tolerâncias pré-definidas: FP32 atol=1e-6/rtol=1e-5;
FP16 atol=1e-3/rtol=1e-3. Não se exige igualdade bit a bit entre CPU e GPU.

O snippet Python literal do Model Card é executado adicionalmente em CPU/FP32,
em processo Python isolado, fora do repo, com cópia permitida dos nove arquivos,
cache HF vazio e rede bloqueada. Aberturas de models/reports/src/cache/docs/data
do projeto original são bloqueadas. O runtime PyTorch/Transformers continua sendo
uma dependência instalada; o código reviewsense não é importado pelo snippet.
`-I` não remove sozinho caminhos de instalações editable via .pth: o verificador
remove também os caminhos internos do sys.path do filho.

Guardas verificam zero acesso ao teste. Autenticação é desativada (`token=False`)
nos carregamentos locais. O verificador checa bytes do pacote e da origem novamente
ao terminar. Resultados vão para `outputs/phase4b/verification_DEVICE_PRECISION.json`,
sem textos, IDs ou predições individuais.

## Model Card e validação

O Model Card descreve fonte original, espelho e divergência de licença; o ID do
espelho no YAML é uma referência de catálogo, não altera a procedência/licença ou
o split local. O model-index contém os quatro números exatos de avaliação do BERT,
e o corpo compara o SVM e detalha classes, recursos computacionais e limitações.

YAML é interpretado por huggingface_hub.ModelCard e conferido localmente contra
o subconjunto usado do [esquema oficial](https://huggingface.co/docs/hub/model-cards).
As métricas são comparadas aos agregados congelados, sem recalculá-las. Não se
chama ModelCard.validate(), pois esse método usa um endpoint remoto. Validação
remota final do Hub fica para a 4C autorizada; não se afirma aprovação desse serviço.

## Funções principais

| Função | Propósito |
| --- | --- |
| `release_inputs` | Ancora seleção e hashes nos registros anteriores |
| `safe_file` / `inspect_directory` | Restringem caminhos, tipos de arquivo e nomes permitidos |
| `build_package` | Copia bytes permitidos e valida o candidato antes de disponibilizá-lo |
| `validate_model_card` | Confere metadata, seções, código Python e números congelados offline |
| `validate_manifest` / `validate_package` | Conferem integridade, configuração e bloqueio de publicação |
| `local_logits` / `compare_outputs` | Verificam equivalência de inferência sem treinamento ou métricas de dataset |
| `verify_independent_example` | Comprova execução do snippet sem dependência dos arquivos originais |

## Testes e preservação

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/pytest -q
.venv/bin/ruff check src tests app scripts
.venv/bin/ruff format --check src tests app scripts
.venv/bin/python -m pip check
```

Testes de empacotamento usam BERT minúsculo e artefatos sintéticos. Os agregados e
documentos versionados servem de referência, nunca o CSV real. Incluem adulteração,
ausências/extras, symlinks, configuração, licença, determinismo, precisão numérica,
ausência de treino/upload e execução isolada com cache vazio. Nenhum teste anterior
ou implementação da Fase 4A é alterado.

Auditoria: [FASE4B_LICENSE_REVIEW.md](FASE4B_LICENSE_REVIEW.md).
Publicação depende dos gates e da autorização da
[FASE4C_RELEASE_PLAN.md](FASE4C_RELEASE_PLAN.md), não do sucesso técnico do pacote.


## Resultado do candidato desta entrega

Preparado e verificado localmente a partir do commit `9ee530c7c7b9b79076c7328062b345cf67cec0ea`:
**9 arquivos / 436422685 bytes**. Original e pacote coincidiram nas cinco entradas
sintéticas em CPU/FP32 e GPU/FP16; diferenças máximas observadas de logits/scores
iguais a zero em cada modo. O exemplo literal do Card executou isolado, sem acesso
às origens ou rede. **253 testes offline aprovados** e checks de qualidade verdes.

O manifest técnico versionado está em `reports/phase4b_package_manifest.json`;
o pacote com pesos continua apenas em outputs, ignorado. Relatório completo:
[validation_phase4b.md](../reports/validation_phase4b.md). Licenciamento permanece
**REQUIRES_REVIEW** e publicação **não autorizada**; a Fase 4B está tecnicamente
concluída e os gates da 4C continuam pendentes.

Correção pré-merge: namespace GitHub `jorgeribeiru/Review-Sense`; staging reconstruído e equivalência CPU/GPU repetida. O candidato anterior permanece arquivado localmente. Auditoria de links e novos hashes estão no relatório da Fase 4B.
