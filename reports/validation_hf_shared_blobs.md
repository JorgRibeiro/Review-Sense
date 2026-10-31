# Verificação pós-upload — configuração de shared blobs

Execução em 2026-10-30T23:06:57.776624-03:00 (America/Recife).
[Registro estruturado](hf_shared_blobs_verification.json). Complementa a
[primeira tentativa interrompida](validation_hf_publication_execution.md), sem reescrevê-la.

**Verificação integral do download e inferência independente aprovadas.**
Não houve upload novo, alteração de pesos, criação de Space ou gasto.
REQUIRES_REVIEW / NOT_ESTABLISHED permanecem inalterados.

## Diagnóstico e limite da inspeção anterior

O cache original era TemporaryDirectory, prefixo reviewsense-hub-independent-.
A inspeção das entradas correspondentes em /tmp, por lstat/readlink sem resolver
links, não encontrou esse diretório. O contexto já o havia eliminado. Portanto,
o destino do link original não foi inspecionado e não é apresentado como fato observado.

O [guia oficial de cache](https://huggingface.co/docs/huggingface_hub/guides/manage-cache#shared-blobs-across-repos)
e o código instalado de huggingface_hub **2.4.0** descrevem o compartilhamento Xet:

```text
snapshot/model.safetensors -> repo/blobs/etag -> cache/blobs/prefix/xet_hash
```

O primeiro link é aceito pelo materializador aprovado; o segundo viola sua condição
de blob regular dentro do repo. No Linux, O_NOFOLLOW pode retornar ELOOP quando a
folha é um symlink: a mensagem não prova um ciclo de links ou corrupção do modelo.
Ver [open(2)](https://man7.org/linux/man-pages/man2/open.2.html).
As fontes explicam a falha anterior e a correção funciona; o layout antigo exato
continua não observado porque seu cache não existe mais.

## Suporte e ordem de configuração

A [referência oficial de variáveis](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfhubdisablesharedblobs)
e constants.py da versão instalada reconhecem HF_HUB_DISABLE_SHARED_BLOBS.
_shared_blobs.shared_blobs_enabled() consulta essa constante antes de criar/compartilhar
blobs. A flag é capturada na importação. Dois processos novos/offline confirmaram:

| Configuração | Constante disable | Compartilhamento |
| --- | --- | --- |
| 1 antes de importar | true | false |
| 0 na importação e 1 somente depois | permanece false | permanece true |

A nova execução definiu **HF_HUB_DISABLE_SHARED_BLOBS=1 antes do processo importar
huggingface_hub**. HF_HUB_DISABLE_XET permaneceu false: não foi necessário desativar
Xet nem os links normais entre snapshots e blobs. hf_xet instalado: 1.8.0.
Hashes dos arquivos de implementação consultados constam no JSON.

## Download e integridade efetivamente verificados

Modelo: [JorgeRibeiru/reviewsense-ptbr-bertimbau](https://huggingface.co/JorgeRibeiru/reviewsense-ptbr-bertimbau).
Revisão fixa: **ae89cf168ffaa64115c097204e678ed513a54265**.
Download público com **token=False**, cache temporário novo/vazio, allow_patterns
contendo somente os dez nomes aprovados; sem dependência de main ou credencial local.
A metadata remota inclui .gitattributes da plataforma; esse arquivo não foi baixado.

Após o download, a inspeção por descritores de diretório, stat(follow_symlinks=False)
e readlink registrou os dez links folha e **dez blobs regulares**. Não se seguiram
links arbitrários ou links adicionais dentro de blobs. O materializador existente
foi executado sem alteração de O_NOFOLLOW, O_NONBLOCK, proteção dos ancestrais,
raiz, allowlist, tamanho ou SHA-256. O pacote regular passou validate_candidate,
configuração, Model Card e comparação integral com o inventário autorizado.

**Todos os dez hashes e tamanhos abaixo foram calculados sobre os arquivos baixados
e materializados e conferidos antes/depois das inferências**; não são apenas
valores declarados pelo servidor. Total: 436449591 bytes.

| Arquivo | Bytes | SHA-256 verificado |
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

Em particular, model.safetensors mantém 435725300 bytes e SHA-256
**a65f1ee302987c5f0b01cd7981722e92f17ff332358b8e19b8dc01ee05753c70**.

## Inferência independente

O snippet literal do Model Card foi executado fora do repositório, em CPU/FP32,
com dez arquivos copiados/verificados, cache vazio, Python isolado e bloqueios de
rede/origens. Um segundo processo independente executou cinco textos sintéticos,
incluindo o longo com truncamento efetivo em 256 tokens. Não importou reviewsense
nem acessou modelos/dados/relatórios/código internos. Logits e scores finitos,
labels 0/negativo, 1/neutro, 2/positivo, tokenizer cased e soma dos scores confirmados.

| Caso sintético | Saída | Score negativo | Score neutro | Score positivo | Tokens com padding |
| --- | --- | ---: | ---: | ---: | ---: |
| 1: Produto funcional/entrega rápida | positivo | 0.002819 | 0.281412 | 0.715770 | 16 |
| 2: Produto não funcionou | negativo | 0.929052 | 0.058420 | 0.012529 | 16 |
| 3: Descrição sem opinião explícita | neutro | 0.022297 | 0.568116 | 0.409587 | 16 |
| 4: Opinião mista sobre produto/entrega | neutro | 0.048032 | 0.750338 | 0.201630 | 16 |
| 5: Texto sintético longo, repetido | neutro | 0.109573 | 0.751471 | 0.138956 | 256 |

Scores Softmax **não calibrados**, arredondados somente nesta tabela. Não são certeza,
Accuracy em dataset ou novas métricas de qualidade. Não houve ajuste, treino,
calibração ou reabertura do teste final. Os resultados originais de ML permanecem congelados.
Guardas: **zero leituras/tentativas bloqueadas de test.csv**, zero acessos às origens
e zero conexões de inferência após o download. Evidências completas no JSON.

## Aplicação no startup ZeroGPU, sem mudar os 17 arquivos

A matriz aprovada já fixa huggingface-hub 2.4.0. No startup, ZeroGPUSettings.source
chama o mesmo snapshot_download com cache_dir explícito, revisão fixa, dez nomes e
token=False, antes do mesmo HubSnapshot/materializador. Uma **Variable pública**
configurada antes de iniciar o processo é suficiente para selecionar o comportamento
comprovado nesta verificação:

```text
HF_HUB_DISABLE_SHARED_BLOBS=1
REVIEWSENSE_CACHE_ROOT=/tmp/reviewsense-hub-no-shared-v1
REVIEWSENSE_MODEL_REVISION=ae89cf168ffaa64115c097204e678ed513a54265
```

Manter as demais configurações/Secrets legítimos já previstos no guia. Usar cache
próprio vazio no primeiro startup. A variável **não converte blobs com symlink já
existentes**: file_download.py pode reutilizar um blob existente antes desse ramo.
Não reaproveitar o cache problemático nem definir a flag depois dos imports.

Uma sonda offline importou a entrada e o módulo ZeroGPU dos **arquivos aprovados**
e confirmou a constante true/compartilhamento false sem importar torch/spaces,
inicializar CUDA ou abrir rede. Bytecode transitório da sonda foi removido; os
17 nomes, tamanhos e hashes do bundle foram reconferidos. Isso é teste de importação
e decisão de cache, **não startup completo, build ou execução real ZeroGPU**.
Esses pontos só podem ser verificados no runtime real posteriormente.

## Decisão sobre continuidade e preservação

Os dez arquivos do modelo, os 17 do Space e o plano canônico
**4c9baf15fd7a4e4a503690cbeee9d529d46785f698914188eb2590f82be3a340**
permanecem idênticos. Nenhuma alteração de payload/controles foi necessária, portanto
não há novo inventário para aprovar. O pedido atual autorizou expressamente a nova
verificação e a preferência por configuração externa.

**A autorização operacional anterior cobre a continuação com os mesmos arquivos,
destinos públicos e condições gratuitas, usando essa configuração necessária.**
Ela não cobre gastos, outros modelos ou mudanças de arquivos. Elegibilidade gratuita,
versões/ABI do runtime real, scheduler, quotas, VRAM e experiência móvel ainda
precisam ser confirmados no deploy. Se for necessário mudar o payload ou contratar
recursos, parar e apresentar a divergência antes de qualquer upload.

**Nenhum Space foi criado nesta execução**, respeitando a pausa solicitada para
confirmar essa cobertura. Nenhum Secret remoto ou repositório foi modificado.
A conclusão técnica não é resolução jurídica: REQUIRES_REVIEW / NOT_ESTABLISHED
continuam iguais. Não houve publicação no LinkedIn ou alteração da release v1.0.0.

Validação adicional com a flag definida antes da importação: **432 testes
aprovados em 19.85 s**, sob guarda do CSV real. Ruff aprovado, **59 arquivos
já formatados**, pip check sem conflitos. Nenhuma dependência instalada/substituída.
Os arquivos de código e documentos aprovados, pesos/tokenizer, métricas e relatórios
históricos mantiveram seus hashes. Os três splits mantêm inode/tamanho/mtime_ns por
stat, sem ler seu conteúdo ou recalcular fingerprints. As únicas alterações Git
são documentação atualizada e novas evidências, sem push/merge/tag.
