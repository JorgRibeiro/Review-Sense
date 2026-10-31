# Verificação da preparação v1.0.0 — código e documentação

Branch: `feat/release-v1-code-docs`. Base: `09d18b0`, main após PR #7.
Execução local em 30/10/2026 (America/Recife; timestamp UTC no JSON).
Esta é evidência de preparação; não declara tag, GitHub Release, pesos ou Space publicados.
[Registro estruturado](release_v1_verification.json).

## Alterações e escopo

README orientado ao produto, resultados congelados, figuras existentes e condições
reais de uso local. O README anterior foi preservado integralmente no histórico,
com somente ajuste do título e dos links relativos, conferido contra a base Git.
Guia, changelog, notas prontas e checklist manual não executam publicação.
Versão fonte alinhada a 1.0.0; as únicas mudanças Python/metadata são esses dois
valores de versão. Nenhum algoritmo, dependência ou arquitetura foi alterado.

| Arquivo | Alteração |
| --- | --- |
| [README.md](../README.md) | Apresentação, resultados, demonstração e limites |
| [CHANGELOG.md](../CHANGELOG.md) | Escopo da primeira versão, sem tags retroativas |
| [Histórico técnico](../docs/HISTORICO_TECNICO.md) | Detalhes anteriores preservados |
| [Guia da release](../docs/RELEASE_v1.0.0.md) | Reprodução condicional, artefatos, direitos e checklist |
| [Notas de lançamento](../docs/RELEASE_NOTES_v1.0.0.md) | Texto pronto para copiar no GitHub Releases |
| [pyproject.toml](../pyproject.toml) | Versão 1.0.0 |
| [__init__.py](../src/reviewsense/__init__.py) | Versão 1.0.0 |
| Este relatório e [JSON](release_v1_verification.json) | Evidências novas, sem reescrever relatórios anteriores |

## Verificações executadas

| Verificação | Resultado |
| --- | --- |
| Suíte completa, HF_HUB_OFFLINE=1 e TRANSFORMERS_OFFLINE=1 | **393 passed in 14.77s** |
| Ruff: src tests app scripts deploy/zerogpu | **All checks passed!** |
| Ruff format --check: mesmos diretórios | **57 files already formatted** |
| pip check | **No broken requirements found.** |
| git diff --check | Sem erros |
| Versões fonte: pyproject e reviewsense.__version__ | Ambas 1.0.0 |
| Bundle ZeroGPU em diretório temporário | 17 arquivos permitidos, validado, sem pesos/dados |

A suíte foi executada por pytest.main(["-q"]) dentro de
reviewsense.final_evaluation.test_access_policy, com final=False, apontando para
**o test.csv real**. Resultado: **read_opens=0, blocked_opens=0**. Portanto não houve
abertura nem tentativa bloqueada de acesso a esse arquivo. Fixtures com arquivos
sintéticos de mesmo nome continuam permitidas nos diretórios temporários dos testes.

A suíte usa modelos/dados sintéticos e contratos SDK offline. Não houve inferência
com os pesos treinados, reavaliação de teste, treinamento ou novo benchmark.
Não foi necessário adicionar testes para a mudança de documentação/versão.

Ambiente: Python 3.11.14/Linux, PyTorch 2.14.1+cu132. Versões adicionais no JSON.
A metadata da instalação editável existente ainda informa 0.1.0: não foi reinstalada
para atualizar o número. O código importado e o pyproject já informam 1.0.0;
uma instalação posterior atualizará a metadata. pip check verifica o ambiente
existente, não comprova instalação limpa em outro computador/SO ou stack CPU-only.
Nenhuma dependência foi instalada ou substituída nesta tarefa.

## README, links e imagens

Foram verificados **78 destinos Markdown locais**, incluindo fragmentos e **três
ocorrências de imagens**, em README, CHANGELOG e nos três documentos novos.
Todos existem após a inclusão deste relatório. Os quatro links externos únicos
responderam **HTTP 200**, sem redirecionamento: repositório do projeto, repositório
original do dataset, página do BERTimbau e documentação oficial de GitHub Releases.
O endereço 127.0.0.1:7860 é instrução local; não foi tratado como serviço público
nem foi iniciada uma aplicação para verificá-lo. URLs, destinos e resultados no JSON.

As duas figuras do README foram inspecionadas visualmente e reaproveitadas sem
alteração: F1 Macro agregado e captura histórica Gradio com texto sintético.
A captura não é benchmark novo nem evidência de hospedagem; sua legenda explicita
isso. Nenhum exemplo de avaliação real foi criado. Conferência final da renderização
GitHub e do canal Issues está no checklist manual, antes da publicação.

F1 e Accuracy mostrados no README foram conferidos por arredondamento do
phase3c_final_metrics.json existente: BERT 0,7545/0,8634; SVM 0,7231/0,8325.
A classe neutra BERT permanece F1 0,4454 e recall 0,3855. Não foram calculadas
métricas a partir de textos/predições, nem mudados os agregados ou figuras.

## Preservação e conteúdo do commit

Comparação SHA-256 com snapshot anterior às alterações: **141 arquivos inalterados**,
abrangendo código de inferência/aplicação, testes, notebooks, configurações,
relatórios históricos, descritor, manifesto e os nove arquivos do pacote 4B,
checkpoint selecionado, tokenizer, baseline e marcadores da avaliação encerrada.
Os dois números de versão alterados estão fora dessa comparação de preservação.
Referências SHA-256 essenciais estão no JSON; nenhum manifesto anterior foi refeito.

Treino, validação e teste mantêm inode/tamanho/mtime_ns do snapshot, verificados
somente por stat; **nenhum conteúdo CSV foi aberto ou hasheado**. Esse controle
por metadata é complementar e não equivale a recalcular fingerprints dos dados.

O payload é limitado aos nove arquivos de documentação, evidências e versão listados
no JSON, adicionado por nomes explícitos. Conferência de caminhos, tamanhos e busca
por padrões de tokens HF/GitHub/AWS, chaves privadas, credenciais e caminhos pessoais,
mais revisão do texto: sem ocorrências. Não há dataset, modelo, cache, token ou segredo
identificado no conteúdo preparado. Em data/models/cache/outputs/checkpoints, somente
data/README.md é versionado. Artefatos locais continuam ignorados. Não houve reescrita
nem auditoria exaustiva de todo o histórico Git; a busca por padrões tem esse limite.

## Direitos e operações que permanecem separadas

Código próprio MIT; licenças de terceiros preservadas. Dados/pesos não recebem MIT
por consequência desta release. Descritor/manifesto mantêm
**license_status=REQUIRES_REVIEW; upload_authorized=false**. Nenhum pedido por e-mail.
A release de código não equivale a aprovação jurídica da distribuição do modelo.

Não foram executados push, merge da branch de release, tag, GitHub Release, upload,
deploy, criação de Space, obtenção de credenciais ou postagem. A atualização inicial
ff-only da main local até 09d18b0 foi apenas preparação da base solicitada.

A eventual distribuição de pesos/ZeroGPU ainda exige revisão material de direitos,
licença/atribuições e canais responsáveis; candidato/revisão/destinos e autorização
humana específicos; infraestrutura elegível, provisionamento autorizado e validação
real do scheduler/GPU/filas/erros. Esses pontos não são novas funcionalidades desta
release nem condições declaradas resolvidas pelos testes locais. Publicar o código
seguirá somente o checklist de PR/revisão/tag/release do guia, por decisão humana.
