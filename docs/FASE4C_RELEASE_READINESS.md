# Fase 4C — Preparação local do lançamento

**Engenharia de preparação concluída; publicação bloqueada por revisão de direitos.**
A Fase 4C não está encerrada. Direitos `REQUIRES_REVIEW`, upload não autorizado,
destino HF não confirmado. Não há uploader, login, criação de Space, deploy ou post.

## Candidato e contrato público

O pacote 4B `outputs/phase4b/model-repo-staging/` e seu manifest permanecem intactos.
O novo [descritor de inferência](../publishing/inference_descriptor.json) fica
**fora dos nove arquivos do pacote**. Não foi criada outra cópia ou versão dos
pesos: o candidato continua `1.0.0-rc.1`. Portanto não há alteração de allowlist,
manifest, Model Card ou hashes 4B. Se futuramente o descritor for incluído no
repositório do modelo, preparar novo candidato/manifest e repetir as verificações.
Nesta arquitetura ele pode acompanhar o código da aplicação separadamente.

O descritor contém somente identidade, versão, arquitetura, max_length 256,
labels 0/negativo, 1/neutro, 2/positivo, licença/status, Model Card e inventário
nome/tamanho/SHA dos nove arquivos; a referência inclui revisão base e fingerprint
do pacote. Não exporta relatórios, caminhos absolutos, dados ou histórico de treino.
O peso mantém SHA `a65f1ee302987c5f0b01cd7981722e92f17ff332358b8e19b8dc01ee05753c70`.

O operador deve aprovar/versionar o descritor por canal confiável. Hash autodeclarado
não é assinatura digital, prova de procedência ou autorização jurídica. Atualizar
manualmente todos os checksums de um pacote adulterado não o torna aprovado.

## Componentes principais

| Função/componente | Propósito |
| --- | --- |
| `descriptor_from_manifest` / `validate_descriptor` | Extrair o contrato mínimo e rejeitar campos extras/invariantes inválidas |
| `validate_runtime_package` | Verificar allowlist, tamanhos/SHA, tokenizer cased, labels, metadata e status sem registros internos |
| `LocalPackage.resolve` | Impedir saída da raiz permitida e rejeitar symlinks, inclusive em ancestrais |
| `HubSnapshot.resolve` | Resolver e materializar snapshot **somente em cache offline**, por repo_id/SHA explícitos, após gates |
| `materialize_hub_snapshot` | Copiar somente arquivos aprovados do snapshot/blobs para diretório regular, conferindo bytes e SHA-256 |
| `release_gates` | Separar direitos, destino e autorização vinculados ao candidato |
| `InferenceService.prepare` | Pré-carregar opcionalmente antes de servir, mantendo lazy load por padrão |
| `AppSettings` / `service_from_settings` | Configurar origem/ambiente explicitamente; bloquear hospedagem pendente |
| `preflight` | Conferir pacote, Card, descritor, header safetensors, links e gates |
| `verify_phase4c_readiness` | Completar preflight com equivalência real e smoke do serviço em textos sintéticos |

`InferenceService.predict`/`predict_batch` e `Prediction` não mudaram. O default
BERT agora usa pacote local + descritor; instanciar o serviço não lê arquivos.
`ArtifactPaths` explícito mantém o caminho histórico dos treinamentos. SVM segue
local, conferido pelos registros das Fases 2/3C; não acompanha o pacote público.
`AppSettings(environment="hosted")` sempre fixa `enable_svm=False`, inclusive se
construído diretamente com True. O ambiente local mantém a escolha anterior.
Cache por instância, lock, FP32/CPU, FP16/GPU compatível, recuperação CUDA limitada,
scores não calibrados e ausência de persistência de entradas foram preservados.

## Reprodução local

Usar o ambiente existente Python 3.11 e dependências de `app/requirements.txt`.
Nenhum pacote foi instalado/substituído nesta fase; preservar PyTorch CUDA funcional.
Com acesso legítimo ao pacote preparado:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m app.app --device cpu
# Pré-carga opcional, sem abrir o navegador nem criar túnel:
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m app.app --device auto --preload
```

Outro diretório requer opções explícitas, sem aceitar caminhos fornecidos pela UI:

```bash
.venv/bin/python -m app.app --model-dir /diretorio/autorizado/model \
  --allowed-root /diretorio/autorizado --descriptor /diretorio/autorizado/descriptor.json
```

Na API, para manter a resolução antiga explicitamente:

```python
from pathlib import Path
from reviewsense.predict import ArtifactPaths, InferenceService
service = InferenceService(ArtifactPaths.from_repository(Path.cwd()), device="cpu")
```

O modo `--model-origin hub-cache` exige repo_id, revisão SHA de 40 caracteres,
cache root, descritor e approval root explícitos. Não há valores de namespace
padrão, download autorizado, leitura de token ou fallback para main/modelo base.
O adaptador lê diretamente o layout local
`models--<namespace>--<repo>/snapshots/<sha>/`, sem chamar `snapshot_download`,
API, autenticação ou rede. Com o candidato real, os gates recusam antes de acessar
o snapshot. A correção foi validada com caches sintéticos realistas, sem downloads.

No cache são aceitos arquivos regulares ou links folha no formato exato
`../../blobs/<etag>` (40 ou 64 hexadecimais), somente para blobs regulares do mesmo
repo. Abertura POSIX por descritores de diretórios e O_NOFOLLOW rejeita symlinks
nos ancestrais, nos blobs, links absolutos, encadeados ou fora da raiz. Extras
pré-existentes do cache são ignorados, nunca abertos/copiados. Somente a allowlist
é copiada e seu tamanho/SHA-256 conferidos durante a leitura. A cópia temporária é
validada integralmente antes de promover o diretório regular, sob
`cache_root/reviewsense-materialized/<repo>/<sha>/<digest-do-descritor>/`.
Uma cópia existente só é reutilizada após validação; não é reparada/sobrescrita.
`LocalPackage` permanece intacto e continua recusando todos os symlinks e extras.
Ausência de cache ou de suporte POSIX a dir_fd/O_NOFOLLOW falha de forma segura.
Provisionamento remoto, downloads autenticados e adaptação ZeroGPU
**não estão habilitados/testados nesta entrega**.

## Preflight completamente offline

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
  -m scripts.verify_phase4c_readiness --dry-run --device cpu
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
  -m scripts.verify_phase4c_readiness --dry-run --device cuda
```

`--dry-run` é obrigatório: significa sem efeitos remotos, mas **executa inferência
local real**. Não é treino nem avaliação de dataset. Saídas fixas em
`outputs/phase4c/readiness_cpu.json` e `readiness_cuda.json`, ignoradas pelo Git.
O script não aceita destino de saída que possa sobrescrever pesos/resultados.
A guarda bloqueia test.csv; uma audit hook impede conexões. O exemplo literal do
Card também executa em subprocesso isolado com cache vazio e origens bloqueadas.

As cinco entradas são sintéticas, incluindo texto longo truncado em 256. Equivalência
original/pacote usa o mesmo dispositivo/precisão, com tolerâncias da 4B. O serviço
novo é carregado e exercitado separadamente. Não há nova métrica de qualidade.

O preflight usa [recibo HTTP conhecido](../publishing/readiness_links.json), obtido
separadamente por consultas públicas de documentação. Não consulta links na rede
no dry-run; confere presença/status/data do recibo e destinos locais seguros.
Não valida âncoras nem disponibilidade futura. Atualizar a auditoria separadamente
antes da publicação autorizada. Não se chama ModelCard.validate(), que usa rede.

Allowlist e hashes rejeitam datasets, estados de treino, extras e cópias duplicadas.
A análise de segredos cobre padrões HF/GitHub/AWS, chaves privadas, atribuições de
senhas e caminhos pessoais nos metadados e header safetensors; não é auditoria
exaustiva de segredos ou memorização nos tensores. Não são persistidos textos,
predições por entrada, IDs ou conteúdo pessoal.

## Gates e limites de confiança

A saída separa `TECHNICALLY_READY`, `LEGAL_REVIEW_REQUIRED`,
`DESTINATION_UNCONFIRMED` e `PUBLISH_AUTHORIZATION_REQUIRED`.
A prontidão técnica só fica true após completar inferência/equivalência.
Os outros três são bloqueios independentes; no candidato atual permanecem true.
`remote_execution_allowed` fica false nesta primeira execução. Não há `--force`.

Para uma etapa futura revisada, os recibos locais privados em approval root serão:

- `rights.json`: `descriptor_sha256` canônico, `decision: APPROVED`, scope contendo
  `public_weights` e `public_inference`, `evidence_file` (basename seguro) e seu SHA-256.
- `destination.json`: mesmo digest do descritor, `confirmed: true`, repo_id e revisão
  SHA de 40 caracteres, ambos explicitamente confirmados.
- `human.json`: mesmo digest, repo_id/revisão iguais ao destino, `authorized: true`,
  ações `download`, `public_weights` e `public_inference`.

A licença do descritor também precisa estar APPROVED e identificada, e sua
flag de upload aprovada. Alterar uma flag sem os demais recibos/evidências não
satisfaz os gates. Esses JSONs são registros de confiança do operador: **não
certificam a suficiência jurídica dos direitos, não autenticam o signatário e não autorizam
publicação só por existirem**. Não foram produzidos recibos reais de aprovação.
O candidato 4B não aceita troca de licença: resolver direitos implica novo
candidato/metadata e revisão do protocolo de preflight antes de lançamento real.
A etapa de publicação permanece sem implementação remota nesta versão.

## Aplicação preparada, hospedagem pendente

Padrão 127.0.0.1, share=False; nenhuma exposição pública nesta fase.
A pré-carga hospedada é bloqueada antes de launch quando faltam gates. O ambiente
hosted, quando futuramente autorizado, usa texto institucional apropriado,
esconde o SVM não distribuído e informa que a plataforma pode manter logs próprios.
A prévia de UI hospedada nos testes usa serviço falso e não executa servidor.

Inferência por clique/Enter compartilha limite 1, fila máxima 8, entrada até
8.000 caracteres. Erros conhecidos/inesperados não exibem textos, paths ou
tracebacks; apenas mensagens de validação enumeradas são reaproveitadas.
Manter aviso de Softmax não calibrado, domínio e ausência de parceria institucional.
CSS responsivo foi preservado. Custos/elegibilidade e limites estão na
[decisão de hospedagem](FASE4C_HOSTING_DECISION.md).

## Encerramento posterior

**Decisão de direitos atual:** o usuário não encaminhará solicitações por e-mail;
o rascunho anterior não será enviado. A pendência é resolver materialmente os
direitos e confirmar o canal de reclamações, sem presumir que finalidade educacional,
não comercial ou retirada posterior autorizem a publicação. Nenhuma mensagem é enviada.

**Cenário A — direitos resolvidos:** confirmar namespace/recursos, revisar novo
candidato/avisos e autorização humana; publicar somente a allowlist aprovada,
registrar SHA definitivo, verificar download/carregamento em ambiente limpo,
validar Gradio público, links e segurança, atualizar documentação e obter autorização
para tag/release estável. Só então escrever/revisar o post final de LinkedIn.

**Cenário B — direitos pendentes:** não publicar pesos nem inferência pública,
inclusive por hospedagem indireta. Manter código/documentação próprios disponíveis
nos limites dos seus direitos. Gravação com textos sintéticos exige verificar os
direitos aplicáveis antes de divulgação. Documentar execução local por quem tenha
acesso legítimo aos artefatos. Release só de código/documentação pode ser preparada
se autorizada; divulgação deve esclarecer o que efetivamente está disponível.

O post final não foi escrito. Ao encerrar futuramente, mencionar motivação,
pipeline/NLP, TF-IDF+LinearSVC, BERTimbau, RTX 3050 6 GB, F1 Macro congelado
0.754461 e 0.723094, ganho descritivo relativo 4,34%, engenharia/Gradio, testes,
reprodutibilidade, desafios e limitações do corpus/classe neutra. Usar somente
links definitivos verificados; não sugerir parceria, criar link HF fictício ou
anunciar lançamento inexistente. O post é a última entrega, sujeita a revisão
humana; publicação em rede social exige autorização específica.

Ver [política de direitos, atribuição e reclamações](FASE4C_RIGHTS_OUTREACH.md),
[plano de gates](FASE4C_RELEASE_PLAN.md) e
[relatório da preparação](../reports/validation_phase4c_readiness.md).


## Testes e evidências desta preparação

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/pytest -q
.venv/bin/ruff check src tests app scripts
.venv/bin/ruff format --check src tests app scripts
.venv/bin/python -m pip check
```

A suíte acrescenta modelos minúsculos, mocks de snapshot e recibos apenas sintéticos.
Não produz aprovações reais e não depende de downloads, GPU ou checkpoint real.
O smoke opcional do CLI exige os artefatos locais legítimos. As medições e checks
realmente executados, suas condições e limitações ficam no relatório estruturado
e Markdown desta fase. Não reutilizar o teste final como rotina de verificação.

Correções pré-merge e verificações adicionais: [phase4c_premerge_corrections.md](../reports/phase4c_premerge_corrections.md). Os relatórios anteriores permanecem como registros históricos da execução original.
