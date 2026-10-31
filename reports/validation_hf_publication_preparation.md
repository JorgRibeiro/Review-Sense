# Preparação da publicação HF — confirmação operacional pendente

Branch feat/hf-public-launch, base de13cf5. Código efetivamente executado:
`ff1c7e0588c2e2d7ca6541036cbef7b3bb9b8d24`.
Verificado em 2026-10-31T01:27:25.633093+00:00.
[Registro estruturado](hf_publication_preparation.json).

## Decisão e termos preparados

O responsável autorizou preparar distribuição educacional, sem monetização e com
atribuição integral, ciente da incerteza residual. Isso não é autorização de
terceiros nem parecer jurídico. A situação permanece **REQUIRES_REVIEW / NOT_ESTABLISHED**.
O [Model Card](../publishing/hf_release/model_card/README.md) usa **license: other**
com license_name e license_link=LICENSE. A [oferta de licença](../publishing/hf_release/model_card/LICENSE)
é CC BY-NC-SA 4.0 somente nos direitos controlados pelo responsável, se existentes;
não resolve direitos sobre comentários ou outros direitos de terceiros.
Os avisos MIT originais do base e do código são preservados separadamente.
O texto CC integral é da fonte oficial, SHA e66c269d4819aaab34b49ef5220c4ddab6756f21bb5180761a4eb8561f2b7bbd;
foram normalizadas somente quebras de linha extras finais para o diff Git.

A [decisão documentada](../docs/RELEASE_RIGHTS_DECISION.md) preserva a análise anterior,
registra a nova intenção e separa direitos demonstrados, finalidade e autorização.
O [guia operacional](../docs/HF_PUBLIC_LAUNCH.md) documenta o diff/semântica dos gates,
primeiro upload sem SHA inventado, provisão privada de registros e critérios reais
para verificar o futuro Space. Nenhum recibo de aprovação do modelo real foi criado.

## Destinos e candidato exato

| Destino | Visibilidade solicitada para confirmação |
| --- | --- |
| Modelo: `JorgeRibeiru/reviewsense-ptbr-bertimbau` | Pública |
| Space: `JorgeRibeiru/reviewsense-ptbr` | Pública, somente ZeroGPU gratuito elegível |

Consultas públicas somente de metadata retornaram “not found” para o modelo e
“not found or authentication required” para o Space. Isso não prova disponibilidade
ou ausência de repo privado, titularidade autenticada ou elegibilidade da conta.
Nada foi criado. Identidade/autenticação e condições gratuitas reais serão conferidas
após a confirmação, no terminal local; não contratar plano, hardware ou créditos.

Candidato local: outputs/hf-public-launch/reviewed-candidate/, versão 1.0.0-rc.2.
[Manifest](../publishing/hf_release/package_manifest.json),
[descritor v2](../publishing/hf_release/inference_descriptor.json) e
[plano de upload](../publishing/hf_release/launch_plan.json).
O manifest registra o commit do preparador e os hashes do código, não um commit
fictício de implementação. Somente model/ e space/ serão enviados, sem os JSONs
externos. O descritor acompanha o Space como model_descriptor.json.

Digest canônico do plano, usado para vincular a confirmação:
`4c9baf15fd7a4e4a503690cbeee9d529d46785f698914188eb2590f82be3a340`.
SHA-256 dos bytes do arquivo launch_plan.json:
`afa2a374531b836a6e3c5867dfbbedab4c6b9c5a601f8eda7c8b8db6abd430ca`. Não confundir os dois.
A revisão Hub é **null**, até existir o commit real após upload autorizado; então
será verificada e usada na configuração/autorizações privadas do Space, sem mudar
o bundle aprovado. O identificador de API zero-a10g é o enum ZeroGPU do SDK instalado,
não garantia de um chip A10G físico ou de disponibilidade. Quotas/versões são do provedor.

### Allowlist do modelo — 10 arquivos / 436449591 bytes

| Arquivo | Bytes | SHA-256 |
| --- | ---: | --- |
| `ATTRIBUTIONS.md` | 2343 | `dbfcc4583ce15b8029d5ed097f921e41c3e3961e94ad141831ec22eafe6dcf4e` |
| `BASE_MODEL_LICENSE.txt` | 1138 | `f52509e0695fdc19752f725f3d58ba06cc7842d5dc8d828b8d821c8ee2c92e67` |
| `DISTRIBUTION_STATUS.md` | 2054 | `3e28c42600fdd4dc8754bf8f8739910773e5808cac7498f302a4994e70b7024b` |
| `LICENSE` | 22426 | `a3bc83c509fb1ed75b6679a6bc89e38123de0da624b8509457cd0491ff832195` |
| `PROJECT_CODE_LICENSE.txt` | 1233 | `802ff8d320c2f6bdc3849bae730c5892090bb3de9f1dabd9c005f25c1455d499` |
| `README.md` | 15660 | `321e7a11ea4d4be7c8c89166ea8b06e8667edd7e93f4b1e6d1b6c2ecf2265725` |
| `config.json` | 1061 | `2389ccb3801ce52974ec8501f258cdb3ebc8006bdea439f49e47a330dcd93223` |
| `model.safetensors` | 435725300 | `a65f1ee302987c5f0b01cd7981722e92f17ff332358b8e19b8dc01ee05753c70` |
| `tokenizer.json` | 677889 | `ed6341298b0e748c076332b22e895ec601d33053de0ff6b8dae8c846cce961bb` |
| `tokenizer_config.json` | 487 | `32f7a4b0527717d3c7bd79f5c0edda7baeb8a5f3b9e16c6cb22df5fd581c9a1a` |

O LICENSE acima contém a concessão de escopo limitado; não é MIT global dos pesos.
Não há CSV, corpus, IDs, avaliações reais, estados de treinamento ou tokens como
arquivos do pacote. Não se afirma ausência de memorização ou dados pessoais nos tensores.

### Allowlist do Space — 17 arquivos / 87046 bytes

| Arquivo | Bytes | SHA-256 |
| --- | ---: | --- |
| `LICENSE` | 1233 | `802ff8d320c2f6bdc3849bae730c5892090bb3de9f1dabd9c005f25c1455d499` |
| `README.md` | 4867 | `55f4be05bbcf95d5c6cc3b716b3c4dc3e1f68a1b33ccf5ca1e5a0e8e54902018` |
| `app.py` | 2407 | `c4e338037589b129af487a1653ea3bd316df75f5a09be28cb8da67ebb0a87474` |
| `app/__init__.py` | 76 | `958f55fb4956e523b20275b5bd0fb7c6d1a1dde8f47ce417b7e716635ce3abf4` |
| `app/app.py` | 14673 | `f4ec3b9fc3ba01d13d6bfa82f2180d51863e21dfa0c586ba7a3cee7205971b42` |
| `app/config.py` | 2649 | `27cfe5c8dda651ddae09b316e77f3b9a61f1187729bfda0ddf6d1608749ac080` |
| `app/style.css` | 7790 | `858de5a4d3b7ef431ed12a4d5912acd43ed7a50a00478056bb2c4a2771effb0d` |
| `model_descriptor.json` | 2328 | `04dc6c98457588175bc85f667ab72eff4484eec55403e7e517e6468093d2f2e6` |
| `requirements.txt` | 296 | `1c4a97e71d235ad1c5e3c9b75da451a35cbfb30747617d91eecb0e6af61c9001` |
| `reviewsense/__init__.py` | 104 | `bcadbfc9c96b14d4cc9a94509fe54d981db51d6793a126706b9edb8640f5f973` |
| `reviewsense/artifacts.py` | 3687 | `70d9d2aa0c7ce02dac3a614520396f92497c9cbfa4ad331e8ddc760b79a5cac4` |
| `reviewsense/hub_cache.py` | 4742 | `2f44a77b37950bef3b3f9074434136ef920429ecd0abebd6fcf74b06010256a6` |
| `reviewsense/labels.py` | 814 | `4916278d0dee2e3784a615742204c4e1ca14719d6a4edfd0e6d0858f8eb89ec1` |
| `reviewsense/model_resolution.py` | 17056 | `dcd472305eb364930dce979a10e6f6616a665010fb30365476471df2fcc184e2` |
| `reviewsense/predict.py` | 15601 | `b92c0512e0e52d0537539168cfdeeafe0e69bc1768eec85fc776841a99f94070` |
| `reviewsense/zerogpu.py` | 8111 | `d3dad8e83ddcc274030c9a78817d2def7ff6f3a6e8f205ca0bea9c6e4e5392c8` |
| `runtime.json` | 612 | `f8ac520f5e1a554ce13d1b2eb05c11ca950d0d028e9c109b6f76a21d5bff9307` |

O LICENSE do Space é o MIT do código próprio. Não há pesos embutidos, datasets,
relatórios internos, fixtures SDK ou aprovações privadas. O runtime obtém somente
o snapshot público aprovado por SHA real e materializa a allowlist de dez arquivos.
O Hub pode gerir .gitattributes; não será enviado arquivo extra do projeto por isso.

## Verificação técnica executada

- **432 testes aprovados em 19.12 s**, incluindo os 393 anteriores e 39 novos.
  Suíte do commit do preparador, offline, sem download de checkpoint ou CUDA real nos testes.
- Ruff aprovado; **59 arquivos já formatados**; pip check sem requisitos quebrados;
  git diff --check aprovado. Nenhuma dependência instalada/substituída.
- Pacote, descritor e manifest validados por allowlist exata, bytes/SHA, labels
  0 negativo/1 neutro/2 positivo, tokenizer cased e fonte congelada de métricas.
- Equivalência original/candidato em **cinco textos sintéticos** por modo, CPU/FP32
  e GPU/FP16: classes 5/5 iguais; diferença máxima de logits/scores **0,0** em cada
  modo. Tolerâncias mantidas: 1e-6/1e-5 (CPU), 1e-3/1e-3 (FP16).
  Não é Accuracy de dataset nem comparação de igualdade bit a bit entre CPU/GPU.
- Esse ciclo executou no candidato preliminar; os **quatro artefatos e o snippet
  literal são byte a byte idênticos ao candidato final**. Somente documentação e
  inventários foram finalizados. O exemplo literal também foi **reexecutado no
  candidato final**, fora do repositório, CPU, cache vazio, sem rede/import reviewsense
  ou acesso às origens. Scores finitos e normalizados, três classes confirmadas.
- Guardas da suíte, equivalência e exemplo final: **zero leituras e zero tentativas
  bloqueadas do test.csv real**. CSVs reais não foram abertos nem para checksum.
- **17 URLs** do Card/atribuições/avisos responderam HEAD 200. O artigo BERTimbau foi
  confirmado no editor e o PDF do corpus somente por HEAD, sem ler avaliações.
  Validação remota de metadata no Hub não foi executada nesta etapa.
- **50 destinos Markdown locais** existem, incluindo a âncora do README. O payload
  de **24 arquivos** contém somente código, testes, documentos e metadata agregada.
  Busca por padrões HF/GitHub/AWS, chaves privadas, atribuições de credenciais e
  caminhos pessoais, com revisão dos textos: sem ocorrências. Isso não é auditoria
  universal de segredos/histórico ou de memorização nos pesos.

O verificador independente foi ampliado somente com opção das duas allowlists fixas
(nove ou dez arquivos), preservando seu padrão 4B. Um teste inicial identificou o
uso inadequado do inventário de arquivos planos para subpastas do bundle; foi
corrigido com inventário específico dos 17 caminhos fixos, sem relaxar safe_file.
A primeira chamada do script temporário de equivalência falhou por import de scripts;
foi reexecutada com PYTHONPATH=. no pai e isolamento mantido no filho. Falhas não
são contadas como execuções aprovadas. Evidências finais estão no JSON.

## Preservação e limitações

Dos 183 arquivos anteriores monitorados, **173 mantiveram seus hashes**;
as dez alterações intencionais de código/documentação constam no JSON. Todos os
relatórios/métricas anteriores, pacote 4B/manifest/descritor, pesos/tokenizer,
notebooks, baseline e marcadores finais permanecem iguais. Três splits mantêm
inode/tamanho/mtime_ns; isso é controle complementar de metadata, não novo fingerprint.
Peso original e candidato: a65f1ee302987c5f0b01cd7981722e92f17ff332358b8e19b8dc01ee05753c70.
F1 Macro de teste permanece 0,7545, extraído do agregado congelado, sem recálculo.

Os três gates v2 ainda retornam true sem registros reais. O v1 original continua
REQUIRES_REVIEW/upload_authorized=false. A política nova admite autorização operacional
específica sob incerteza explicitamente mantida; não chama isso de aprovação jurídica.
Não há confirmação operacional fictícia nem recibo real criado antes da resposta humana.
Os registros sintéticos dos testes são vinculados a BERT minúsculo, não ao modelo real.

Não foram executados login, obtenção de credenciais, upload, download do Hub,
criação de repositório/Space, configuração de Secrets remotos, deploy, compra,
e-mail, LinkedIn, push/merge ou alteração da tag/release v1.0.0. Nenhum treinamento
ou novo benchmark/avaliação do teste. A identidade HF autenticada, elegibilidade,
build, ABI/versões, scheduler/GPU, quotas e experiência móvel no Space real ainda
serão verificados **após confirmação operacional**, sem prometer sucesso antecipado.

A confirmação pendente é somente dos arquivos/destinos públicos e das operações
condicionadas do plano, inclusive configurar o Secret privado legítimo e testar
inferência sintética real. Não é pedido de nova análise jurídica ou autorização
inventada de terceiros. Parar se upload, integridade ou verificação do modelo falhar;
sem opção gratuita elegível, não contratar infraestrutura nem usar outro modelo.
