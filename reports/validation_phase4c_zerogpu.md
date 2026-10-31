# Fase 4C — Validação da preparação ZeroGPU

**Adaptação preparada localmente; ZeroGPU real não validado. Publicação bloqueada.**
`license_status=REQUIRES_REVIEW`, `upload_authorized=false`. Fase 4C não encerrada.
Branch `feat/phase4c-zerogpu-readiness`, main atualizada em c977087 (PR #6).
Código/conteúdo verificado: `6ffd035f72c583d7a4770b9af98ce4effbf80062`. Nenhum Space foi criado.

## Resultado e decisões

O adaptador isolado `ZeroGPUInferenceService` reutiliza predição e carregamento
verificado. Pré-carga e colocação CUDA emulada acontecem antes da UI/callback;
não usa detecção de GPU local como emulação ZeroGPU, lazy load por requisição ou
fallback CPU no backend ZeroGPU. O local mantém CPU/GPU, cache, lazy load e SVM.
`create_app` reutiliza o design e transforma opcionalmente o callback GPU; wrapper
preserva metadata/contexto e sanitiza erros do scheduler. BERT-only em hosted,
fila 8, concorrência 1, 8.000 caracteres, 256 tokens. Duração solicitada de 10 s é
orçamento provisório, não medição de hardware, SLA ou benchmark.

Integridade/configuração foram extraídas para artifacts.py; funções anteriores
são reexportadas por model_packaging. Sete funções de segurança extraídas preservam
AST idêntica. Caminhos históricos/SVM ficam em local_inference.py, carregado apenas
quando necessário e excluído do bundle. A API pública local é preservada.
Essa refatoração evita imports de pandas/sklearn/dados/treinamento em BERT-only.

Conta HF `JorgeRibeiru` confirmada pelo usuário. Os nomes de modelo
`JorgeRibeiru/reviewsense-ptbr-bertimbau` e Space `JorgeRibeiru/reviewsense-ptbr`
são candidatos; existência/elegibilidade não consultadas. Nenhum login/token.
A política de não solicitar autorizações por e-mail permanece, sem mensagens.
Transparência/reclamações/retirada não são autorização de distribuição.

## Arquivos adicionados e modificados

| Grupo | Arquivos |
| --- | --- |
| Entrada Space, novos | `deploy/zerogpu/app.py`, `README.md`, `requirements.txt`, `runtime.json` |
| Runtime/preparação, novos | `src/reviewsense/artifacts.py`, `local_inference.py`, `zerogpu.py`, `space_bundle.py` |
| Runtime refatorado | `src/reviewsense/predict.py`, `model_resolution.py`, `model_packaging.py`, `hub_cache.py` |
| UI/configuração | `app/app.py`, `app/config.py`; CSS preservado |
| Testes novos | `tests/phase4c/test_zerogpu.py` |
| Documentação | `docs/FASE4C_ZEROGPU.md`, `README.md` da raiz e `ReviewSense_PTBR_PLANO.md` |
| Evidências novas | `reports/phase4c_zerogpu_dependencies.json`, `phase4c_zerogpu_verification.json`, `validation_phase4c_zerogpu.md` |

## Dependências e validação possível

Alvo: Python 3.12.12, PyTorch 2.13.0 CUDA gerido pelo provedor, spaces 0.51.3,
Gradio 6.31.0, huggingface-hub 2.4.0, Transformers 5.22.1, tokenizers 0.23.3,
safetensors 0.8.0 e NumPy 2.4.6. Metadata oficial confirma requisitos Python e
relações entre os pacotes selecionados, mas não substitui solver/ABI/build real.
Evidência: [phase4c_zerogpu_dependencies.json](phase4c_zerogpu_dependencies.json).
Torch/spaces/Gradio/hub geridos pelo runtime não são reinstalados por requirements;
startup recusa desvio da combinação preparada e build +cpu de Torch.

Ambiente local preservado: Python 3.11.14, Torch 2.14.1+cu132 e CUDA
existentes. spaces não instalado na .venv; fonte Python pública foi inspecionada
em wheel temporário de 0.51.3, sem instalação. Não houve download de pesos nem
provisionamento real. Não foi instalada ou resolvida a matriz alvo completa.
O comando pip check validou somente a instalação local, não a imagem ZeroGPU.

## Testes executados

**370 testes aprovados em 14.83 s**, sendo 342 anteriores e 28 novos, zero falhas/skips.
Ruff, formatação de 54 arquivos Python e pip check aprovados. Suíte offline sob
guarda do test.csv real: zero leituras e zero tentativas bloqueadas.

Cobertos import local sem spaces/treinamento, revisão imutável, marcador/emulação
ativos, ordem de importação, gates antes da carga/download/UI, preparo único,
transferência CUDA solicitada uma vez, rejeição de SVM, pesos ausentes/adulterados,
callback decorado compartilhado, limites/fila, erros de scheduler/startup,
matriz divergente/CPU-only, provisionamento simulado por SHA/allowlist/token=False,
e bundle com extras/symlinks/segredos/adulteração rejeitados.

Matemática dos testes novos foi executada **em CPU, com BERT minúsculo sintético**.
Chamadas CUDA/GPU decoradas foram simuladas. A comparação dos scores do adaptador
com o backend local nesse cenário usou atol=1e-6. Não foram retreinados modelos,
recalculadas métricas finais ou executado benchmark com a RTX 3050/ZeroGPU.
Esses testes não comprovam GPU emulada real, escalonamento, fork/hidratação,
latências, quotas ou VRAM da infraestrutura compartilhada.

## Bundle de código independente

Gerado localmente em `outputs/phase4c-zerogpu/space-candidate-v2/`, ignorado pelo Git,
com **17 arquivos / 76404 bytes**. Fingerprint do inventário:
`4e9aa7f7ab889893b0f8254635e89577b03cf844b5c96fd08fb432d96d8d135e`. Nada foi enviado ao Hub.

| Arquivo do futuro Space | SHA-256 |
| --- | --- |
| `LICENSE` | `802ff8d320c2f6bdc3849bae730c5892090bb3de9f1dabd9c005f25c1455d499` |
| `README.md` | `762d396dc967960e4b0b736ce012674b07a2b555858868261115cd2e0aa29bf4` |
| `app.py` | `ae4902eb7cd0a5b9e9595b8d325cbc432d8772915e90fe24814e95ead607cad3` |
| `app/__init__.py` | `958f55fb4956e523b20275b5bd0fb7c6d1a1dde8f47ce417b7e716635ce3abf4` |
| `app/app.py` | `f4ec3b9fc3ba01d13d6bfa82f2180d51863e21dfa0c586ba7a3cee7205971b42` |
| `app/config.py` | `27cfe5c8dda651ddae09b316e77f3b9a61f1187729bfda0ddf6d1608749ac080` |
| `app/style.css` | `858de5a4d3b7ef431ed12a4d5912acd43ed7a50a00478056bb2c4a2771effb0d` |
| `model_descriptor.json` | `bb51b7f6a0f774871975d744d0c7d582fd6301a324deb72951ec906e1a4aa117` |
| `requirements.txt` | `1c4a97e71d235ad1c5e3c9b75da451a35cbfb30747617d91eecb0e6af61c9001` |
| `reviewsense/__init__.py` | `dba8fd99295b2efd2109db101eee1bf5434749e98e71c9d59af0c7e378c6ac6c` |
| `reviewsense/artifacts.py` | `70d9d2aa0c7ce02dac3a614520396f92497c9cbfa4ad331e8ddc760b79a5cac4` |
| `reviewsense/hub_cache.py` | `7d4ceb323d244c8109db16f7aa3792dbb9af55e00e2497bb77f1b302cc50f2af` |
| `reviewsense/labels.py` | `4916278d0dee2e3784a615742204c4e1ca14719d6a4edfd0e6d0858f8eb89ec1` |
| `reviewsense/model_resolution.py` | `5037bdbd52b48aff18de3a8dd66793cffeb1125d442dc7615efeb0eb54ac0dcf` |
| `reviewsense/predict.py` | `b92c0512e0e52d0537539168cfdeeafe0e69bc1768eec85fc776841a99f94070` |
| `reviewsense/zerogpu.py` | `a90a3d93ce2f80f6f65df31a5110f42b68a681fd1510b693bbed2194dc8230c6` |
| `runtime.json` | `d223c90dbe707b7a6552974fdc5b909fcb53f25b2cbf2b2c739119714e31d681` |

A cópia do descritor é idêntica ao original, ainda bloqueado. Não há pesos, dados,
relatórios, treinamento, cache ou recibos de aprovação. Preparador por allowlist,
sem sobrescrita de destino existente; alterações exigem outro candidato.
O candidato preliminar de código permanece localmente separado, sem substituir
qualquer pacote validado em fase anterior.

O bundle final foi executado adicionalmente com Python -I -B, origens internas
bloqueadas e rede proibida: zero conexões e zero aberturas das origens. Com a
configuração negativa de revisão e o descritor pendente, a entrada recusou antes
de importar spaces/torch ou criar UI/servidor. Nenhuma aprovação fictícia fora dos
testes foi criada. Testes de integração positiva usam somente modelo/cache/recibos
sintéticos no diretório temporário do pytest.

## Preservação e bloqueios

67 arquivos anteriores mantiveram SHA-256, incluindo todos os relatórios/métricas,
publicações/descritor, pacote 4B, pesos/tokenizer selecionados, baseline e marcador/
resultado definitivo anteriores. Três splits mantiveram inode/tamanho/mtime;
nenhum CSV foi aberto. Código de dados/treinamento não mudou. Pesos selecionados:
`a65f1ee302987c5f0b01cd7981722e92f17ff332358b8e19b8dc01ee05753c70`.

Provisionamento público futuro é opcional e sujeito aos gates e SHA; modelo atual
é recusado antes da operação. A autorização humana futura também vincula Space
exato e ação host_zerogpu. Não há fallback para main, trust_remote_code=True,
autenticação automática ou compra de hardware/créditos. Nada foi publicado,
nenhum push/merge/tag/release/deploy ou post de LinkedIn foi realizado por esta etapa.

Pendências: direitos/licença, candidato e revisões públicas aprovados, autorização
de modelo/Space, recibos legítimos, elegibilidade/quotas da conta, matriz/imagem e
GPU reais, monitoramento do canal de reclamações e verificação pública. Reversão
futura deve suspender demo defeituosa e usar apenas revisões anteriormente aprovadas;
sem uma versão aprovada, manter indisponível. Não foi executado rollback remoto.

Guia completo, reprodução, fontes e procedimento de validação/reversão:
[FASE4C_ZEROGPU.md](../docs/FASE4C_ZEROGPU.md).
Evidência estruturada: [phase4c_zerogpu_verification.json](phase4c_zerogpu_verification.json).

**Adaptação ZeroGPU preparada localmente. Deploy, distribuição dos pesos e encerramento da Fase 4C ainda pendentes.**
