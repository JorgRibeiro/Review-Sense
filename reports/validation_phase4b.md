# Fase 4B — Model Card, revisão de licenças e pacote independente

**Tecnicamente concluída. Publicação dos pesos: REQUIRES_REVIEW; upload não autorizado.**
Branch `feat/phase4b-model-card-packaging`, base main d3d71f3. Código de preparação
commit `a95cd2c66f3544a24a2338367487e3f73a3ebd36`, árvore limpa antes do build.
Nenhum treinamento, refit, calibração, recálculo de F1 ou nova avaliação de teste.

## Licenças e decisão de distribuição

A declaração MIT do BERTimbau base e a MIT do código ReviewSense são distintas da
CC BY-NC-SA 4.0 declarada para o dataset original. O espelho do Hub declara CC BY 4.0;
não foi encontrada uma autorização que resolva essa divergência. Não se infere
que os pesos ajustados sejam automaticamente MIT ou automaticamente uma adaptação
submetida à mesma CC. A orientação conservadora da CC não decide o caso concreto.

**REQUIRES_REVIEW** registra a incerteza material. Esclarecimentos/permissões e
avaliação jurídica pertinente são pendentes antes de distribuição pretendida ou
uso comercial. A conclusão técnica desta fase não libera upload. Model Card usa
`license: unknown`, não uma licença conclusiva. Detalhes, fontes oficiais e
recomendação: [FASE4B_LICENSE_REVIEW.md](../docs/FASE4B_LICENSE_REVIEW.md).

## Conteúdo exato e hashes

Local: `outputs/phase4b/model-repo-staging/`, ignorado pelo Git. São nove arquivos,
**436422679 bytes (416.205100 MiB)**, sem compressão. Não há SVM, corpus,
IDs/predições individuais, estados de treino, training_args.bin ou caches.
Apenas textos sintéticos do exemplo de inferência aparecem na documentação.

| Arquivo | Bytes | SHA-256 |
| --- | ---: | --- |
| `ATTRIBUTIONS.md` | 1688 | `ea9b7fc056e9c6f0d21b03dcd1e0a3ba8802ab7b7248a7a78c266f5b40eed433` |
| `BASE_MODEL_LICENSE.txt` | 1138 | `f52509e0695fdc19752f725f3d58ba06cc7842d5dc8d828b8d821c8ee2c92e67` |
| `DISTRIBUTION_STATUS.md` | 976 | `7203975b2d30cbd1cd2e62bf10cdae2df742636da3ca0209990223a410bf86f8` |
| `PROJECT_CODE_LICENSE.txt` | 1233 | `802ff8d320c2f6bdc3849bae730c5892090bb3de9f1dabd9c005f25c1455d499` |
| `README.md` | 12907 | `28ad39c4b8575166a8e0b6746405351855ef0d256b60333f2f6cc3f5ca2c4bec` |
| `config.json` | 1061 | `2389ccb3801ce52974ec8501f258cdb3ebc8006bdea439f49e47a330dcd93223` |
| `model.safetensors` | 435725300 | `a65f1ee302987c5f0b01cd7981722e92f17ff332358b8e19b8dc01ee05753c70` |
| `tokenizer.json` | 677889 | `ed6341298b0e748c076332b22e895ec601d33053de0ff6b8dae8c846cce961bb` |
| `tokenizer_config.json` | 487 | `32f7a4b0527717d3c7bd79f5c0edda7baeb8a5f3b9e16c6cb22df5fd581c9a1a` |

Fingerprint do conjunto de arquivos: `40b214be9bcc62e1713acd933c46bf9ca1c5e9d2ac84e3b690c1cda58946e22e`.
O fingerprint usa nomes/tamanhos/hashes em ordem canônica. Não há timestamp variável
no pacote ou manifest. Os quatro artefatos Transformers preservam exatamente os
bytes da origem; pesos não foram reserializados ou quantizados. Os avisos MIT
explicam seus escopos e não criam uma licença geral para os pesos ajustados.

Manifest técnico, separado dos arquivos destinados ao futuro repositório de modelo:
[phase4b_package_manifest.json](phase4b_package_manifest.json). Repo_id HF é nulo;
identidade local `reviewsense-ptbr-bertimbau-v1`, versão candidata `1.0.0-rc.1`.

## Equivalência, finitude e independência

Cinco entradas sintéticas, incluindo texto longo com truncamento efetivo em 256.
Cada comparação usa o mesmo dispositivo/precisão para original e pacote:

| Modo | Classes iguais | Máx. diferença de logits | Máx. diferença de scores | atol | rtol |
| --- | ---: | ---: | ---: | ---: | ---: |
| cpu_fp32 | 5/5 | 0.0 | 0.0 | 1e-06 | 1e-05 |
| cuda_fp16 | 5/5 | 0.0 | 0.0 | 0.001 | 0.001 |

Diferença observada zero significa igualdade nesta comparação controlada. Não
foi exigida nem afirmada igualdade bit a bit entre CPU/FP32 e GPU/FP16. Classes
coincidem em 5/5 entradas por modo; isso não é Accuracy em um dataset anotado.
Nenhuma métrica de qualidade foi calculada a partir desses exemplos.

Confirmados logits finitos, labels 0/1/2, tokenizer cased, padding/truncamento até
256, eval e inference_mode. Em ambos os modos o snippet Python literal do Model
Card foi executado adicionalmente em CPU/FP32 fora do repositório, com cópia dos
nove arquivos, cache HF vazio e Python -I. Caminhos internos de modelos, relatórios,
dados, src, docs e cache foram bloqueados e removidos do sys.path do processo filho.
**Zero aberturas dessas origens, zero conexões de rede e nenhum import reviewsense**
no snippet. O runtime PyTorch/Transformers instalado é necessário; dados e código
interno do projeto não são dependências dos artefatos de inferência.

## Model Card

[publishing/model_card/README.md](../publishing/model_card/README.md) é utilizável
como README na raiz de um futuro repo de modelo. Inclui resumo inglês, dados e
limpeza, classes por estrelas, treinamento, seleção antes do teste, avaliação
independente e comparação SVM, custos em contextos documentados, limitações,
atribuições, uso responsável e exemplo local validado. Resultados provêm dos
registros estruturados da Fase 3C; latências da Fase 4A são identificadas como
medições sintéticas, sem SLA. Não se afirma validação em delivery.

Metadata foi interpretado por huggingface_hub e validado offline no subconjunto
usado do esquema oficial. model-index contém os quatro valores exatos do BERT.
O identificador do espelho é um vínculo de catálogo explicitamente distinguido
no corpo da procedência original e do split local. O validador remoto do Hub não
foi chamado; sua aprovação não é presumida e fica para uma 4C autorizada.

## Qualidade e preservação

- **253 testes offline aprovados**, 217 anteriores intactos e 36 novos; zero falhas/skips.
- Ruff, formatação e pip check aprovados; nenhuma dependência nova.
- Guardas: zero leituras/tentativas de abrir test.csv em preparação, verificadores,
  snippet isolado e suíte final. A integridade dos splits foi conferida por metadados
  de arquivo; os CSVs não foram abertos nem para checksum nesta fase.
- Modelos e relatórios anteriores conferidos por hashes; marcador e resultado local
  da execução definitiva da Fase 3C mantiveram seus hashes. App, notebooks e testes
  anteriores não foram modificados. Main não recebeu alterações desta branch.
- Allowlist, arquivos ausentes/extras/adulterados, links simbólicos, paths de escape,
  configuração, Model Card, tolerâncias, ausência de treino/upload e bloqueio de
  publicação cobertos com modelos minúsculos e exemplos sintéticos.
- O isolamento detectou um detalhe relevante: Python -I preservava src inserido
  por .pth de uma instalação editable. A verificação remove esse caminho no filho;
  depois disso, também as sondagens de metadata do runtime não acessaram a origem.
- Sem criação de repo, upload, login, manipulação de tokens, deploy ou LinkedIn.
  O empacotador não oferece operação remota.

Evidências estruturadas: [phase4b_package_verification.json](phase4b_package_verification.json).
Guia e reprodução: [FASE4B_MODEL_PACKAGING.md](../docs/FASE4B_MODEL_PACKAGING.md).
Uma saída existente nunca é substituída; verificações podem ler o pacote preparado.
Não regenerar datasets ou pesos para reproduzir esta preparação.

## Próxima etapa

[FASE4C_RELEASE_PLAN.md](../docs/FASE4C_RELEASE_PLAN.md) fixa os gates para resolver
licenciamento, confirmar destino e autorização, publicar somente a allowlist,
verificar download por revisão imutável e integrar um diretório/descritor verificado
sem relatórios locais do desenvolvedor. Prevê validação da demonstração pública,
release e somente ao final redação/revisão/publicação humana do post no LinkedIn.
Nenhum texto de post ou artefato público do modelo foi publicado nesta fase.
