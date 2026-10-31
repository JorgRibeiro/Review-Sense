# Preparação imediata da publicação HF — ponto de confirmação operacional

**Atualização de execução:** o responsável confirmou o plano canônico 4c9baf15fd7a.
O modelo foi publicado na revisão ae89cf168ffaa64115c097204e678ed513a54265.
A primeira verificação falhou em Errno 40; uma nova verificação, autorizada pelo
responsável, passou usando HF_HUB_DISABLE_SHARED_BLOBS=1 antes da importação,
cache vazio, o mesmo materializador e todos os controles preservados.
Hashes/tamanhos dos dez arquivos e inferência independente passaram. Space não
criado; build/ZeroGPU real não exercitados. [Relatório atual](../reports/validation_hf_shared_blobs.md)
e [falha inicial preservada](../reports/validation_hf_publication_execution.md).
O conteúdo abaixo registra o procedimento e o ponto de confirmação da preparação,
não uma nova exigência de autorização para o plano já confirmado.

Branch feat/hf-public-launch. Preparação e operações específicas foram autorizadas;
a verificação do modelo foi concluída; **a criação do Space continua suspensa nesta execução**.
Este guia sucede a recomendação operacional inicial da Etapa A pela
[decisão explícita do responsável](RELEASE_RIGHTS_DECISION.md), sem alterar a análise
jurídica, a tag v1.0.0 ou os resultados congelados.

## Dimensões que não são equivalentes

| Dimensão | Evidência/estado |
| --- | --- |
| Integridade técnica | Allowlists, bytes/SHA-256, labels, tokenizer e equivalência sintética |
| Ciência/decisão do responsável | Intenção de preparar publicação educacional, sem monetização e com atribuição |
| Situação jurídica | REQUIRES_REVIEW, legal_clearance=NOT_ESTABLISHED; sem parecer ou novas permissões de terceiros |
| Autorização operacional | Concedida para o plano 4c9baf15fd7a; payload inalterado e verificado, sem criação de Space nesta execução |

O novo Model Card usa license: other, license_name descritivo e license_link=LICENSE.
LICENSE oferece CC BY-NC-SA 4.0 somente nos direitos que o responsável controla,
se existentes, preservando os avisos MIT do base e do código. Contém o texto CC
integral e inalterado. Isso não resolve direitos de consumidores/terceiros nem
classifica automaticamente os pesos como obra derivada. Finalidade educacional
não acrescenta uma restrição exclusiva de pesquisa à licença escolhida.

## Candidatos e reprodução local

Preparador sem APIs remotas/autenticação:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m reviewsense.hf_launch \
  --destination outputs/hf-public-launch/reproduction-new
```

Usar sempre destino novo. O preparador ancora os quatro artefatos nos registros
congelados de 3B/3C, copia somente os dez nomes permitidos e valida todos os hashes.
Não abre nenhum split; não reserializa pesos, tokenizer ou estados de treinamento.
Saídas: model/, space/, package_manifest.json, inference_descriptor.json e launch_plan.json.
Somente model/ e space/ são candidatos a upload; os três JSONs externos não vão
para o repositório do modelo. O descritor acompanha o Space como model_descriptor.json.

Documentos públicos do modelo: [publishing/hf_release/model_card](../publishing/hf_release/model_card).
Manifest, descritor e plano da entrega ficam em publishing/hf_release/, fora dos pesos.
Os candidatos binários permanecem em outputs/, ignorados pelo Git. Não sobrescrever
outputs/phase4b nem publicar o repositório de desenvolvimento inteiro.

## Alteração explícita dos gates — esquema v2

Diff revisável: git diff de13cf5 -- src/reviewsense/model_resolution.py
src/reviewsense/hub_cache.py src/reviewsense/zerogpu.py src/reviewsense/space_bundle.py
deploy/zerogpu/app.py deploy/zerogpu/runtime.json. Após commit, usar git show do commit.

O esquema v1 e suas três condições não mudam: somente APPROVED com evidências e
recibos antigos permite seu caminho remoto; o descritor 4B continua unknown/false.
LocalPackage continua rejeitando symlinks, travessia, extras, hashes e labels inválidos.

O esquema v2 tem dez nomes fixos (os nove anteriores e LICENSE), license=other,
license_status=REQUIRES_REVIEW e distribution_terms fixos. Esses termos registram
NOT_ESTABLISHED, a concessão limitada, os destinos planejados e a necessidade de
confirmação operacional. Não aceita APPROVED, MIT global, upload_authorized ou
campos extras. Não troca nomes arbitrariamente a partir de uma allowlist fornecida
pela UI. O bundle Space continua com 17 nomes fixos, sem pesos/dados/recibos.

Gates v2 retornam somente bloqueios **operacionais**:

- RIGHTS_POLICY_UNACKNOWLEDGED: falta ciência registrada da incerteza, propósito
  educacional/não comercial e evidência correspondente. Não se chama aprovação legal.
- DESTINATION_UNCONFIRMED: repo/visibilidade ou SHA real não correspondem ao previsto.
- PUBLISH_AUTHORIZATION_REQUIRED: falta consentimento específico do operador para
  aqueles bytes/destino/revisão/ações. Mesmo risco aceito não libera sozinho o runtime.

A situação jurídica permanece no descritor, sempre REQUIRES_REVIEW/NOT_ESTABLISHED
nesse esquema, mesmo quando todos os bloqueios operacionais forem atendidos.
Não se transforma ciência de risco em aprovação jurídica. Criou-se uma política
operacional explícita solicitada pelo responsável, não uma aprovação retroativa v1.

## Sequência após a confirmação — sem inventar SHA

1. Validar novamente candidato e plano. Registrar a confirmação real fora do Git,
   vinculada ao digest canônico do plano, conta JorgeRibeiru, dois destinos públicos,
   operações enumeradas e veto a recursos pagos. validate_operational_confirmation
   rejeita confirmação ausente, outro digest/destino/conta/escopo ou suposta aprovação jurídica.
2. Autenticar exclusivamente no terminal local com permissões mínimas; não imprimir,
   versionar ou incluir tokens em relatórios/recibos. Confirmar identidade e permissões
   antes de qualquer escrita. A leitura pública de fontes nesta preparação não é login.
3. Criar/publicar somente a allowlist do modelo aprovada. Não sobrescrever repo existente
   divergente nem fazer upload de pastas externas. O servidor pode gerir .gitattributes;
   isso não autoriza outros arquivos do projeto. Registrar o SHA retornado pelo Hub.
4. Baixar por esse SHA para diretório independente/cache limpo, conferir allowlist,
   hashes e Card; executar exemplo sintético sem acesso aos dados/relatórios internos.
   Somente depois dessa verificação gerar registros privados de consumo/hospedagem.
5. Conferir elegibilidade ZeroGPU gratuita e a matriz real. Não contratar assinatura,
   hardware ou créditos; se não houver opção gratuita elegível, parar antes do deploy.
6. Configurar Space com repo_id e SHA real verificado, cache privado e Secret descrito
   abaixo. Enviar só os 17 arquivos do bundle, mantendo spaces.GPU, fila 8,
   concorrência 1, limites de entrada e sanitização. Não baixar outro modelo como fallback.
7. Conferir build, versões, startup, alocação real, inferência sintética, erros/quota,
   recursos e layout móvel. Só declarar sucesso após uma inferência real na URL pública.
   Falha de upload/download/integridade interrompe a sequência; não avançar silenciosamente.

O plano contém model_hub_revision=null até existir o commit real. O SHA será fornecido
por configuração do Space, sem precisar alterar os bytes aprovados do bundle.
Nenhum SHA fictício ou recibo de aprovação do modelo real foi produzido nos testes.

## Registros operacionais privados no Space

### Configuração de cache obrigatória antes do startup

Fornecer **HF_HUB_DISABLE_SHARED_BLOBS=1 como Variable pública**, não Secret,
antes de iniciar qualquer processo Python do Space. A versão alvo huggingface-hub
2.4.0 lê a variável na importação. Defini-la depois não muda a constante já carregada.
Não é necessário desativar Xet, alterar o materializador ou aceitar links extras.

Usar cache próprio vazio no primeiro startup, por exemplo
REVIEWSENSE_CACHE_ROOT=/tmp/reviewsense-hub-no-shared-v1. Desativar compartilhamento
não converte symlinks já existentes em um cache antigo; não reaproveitar o cache
problemático. O repo e a revisão continuam os mesmos, token=False e allowlist de dez.
Na configuração do Space, REVIEWSENSE_MODEL_REVISION deve conter o SHA real já
verificado: ae89cf168ffaa64115c097204e678ed513a54265.

Essa configuração externa não altera os 17 arquivos aprovados nem o digest do plano.
A autorização operacional anterior cobre os mesmos destinos/payload, a provisão
necessária e o uso gratuito elegível; este pedido de diagnóstico também solicitou
preferir essa variável. Isso não autoriza gastos ou mudanças de arquivos/runtime
fora do escopo. Conforme a pausa solicitada, nenhum Space foi criado nesta execução.
Na retomada, confirmar elegibilidade gratuita e a matriz real; qualquer divergência
de payload ou necessidade de correção nos arquivos exige nova revisão do inventário.

### Secret dos registros operacionais

Após confirmação real e verificação do upload, fornecer como **Secret**
REVIEWSENSE_OPERATIONAL_APPROVAL_JSON um objeto com risk, destination, human e evidence.
São registros confiados ao operador, não certificados ou assinaturas digitais:

- risk: descriptor_sha256, decision=ACKNOWLEDGED_UNRESOLVED_RIGHTS,
  purpose=educational_noncommercial, third_party_permissions=NOT_ASSERTED,
  scope contendo public_weights/public_inference, evidence_file=owner_decision.md
  e evidence_sha256 dos bytes da decisão humana real.
- destination: mesmo digest, confirmed=true, repo_id exato, visibility=public e SHA
  real de 40 caracteres retornado e verificado no Hub.
- human: mesmo digest, actor=JorgeRibeiru, authorized=true, repo_id/revision/visibility
  iguais, space_id exato e ações download/public_weights/public_inference/host_zerogpu.
- evidence: texto da decisão/confirmação verdadeira, sem tokens ou documentos pessoais.

Esses registros não vão para Git, manifest público, logs ou bundle. O startup retira
esse Secret do ambiente e escreve num diretório temporário 0700, arquivos 0600,
valida os vínculos e preserva o diretório somente durante o processo. Gradio bloqueia
seu acesso. Ausência, tamanho excessivo, campos divergentes ou origem ambígua falham
com mensagem sanitizada. REVIEWSENSE_APPROVAL_ROOT continua possível em provisão
local explícita; não combinar as duas origens.

Configuração pública adicional: REVIEWSENSE_MODEL_REPO_ID, REVIEWSENSE_MODEL_REVISION,
REVIEWSENSE_CACHE_ROOT e REVIEWSENSE_PROVISION_PUBLIC_SNAPSHOT=1. SPACE_ID é fornecido
pelo runtime. O snapshot é fixado por SHA, limitado aos dez nomes e usa token=False.
As credenciais de upload ficam somente com o operador; o Space lê modelo público.

## Ponto de confirmação

Modelo público: JorgeRibeiru/reviewsense-ptbr-bertimbau. Space público:
JorgeRibeiru/reviewsense-ptbr. Exatamente dez arquivos do modelo e 17 do Space,
com nomes/tamanhos/hashes no launch_plan.json apresentado na entrega.
A confirmação é para arquivos, destinos e operações condicionadas acima; não para
uma aprovação jurídica fictícia. A preparação não executa login, escrita remota,
deploy, compra ou LinkedIn. [Plano exato](../publishing/hf_release/launch_plan.json),
[manifest](../publishing/hf_release/package_manifest.json),
[descritor v2](../publishing/hf_release/inference_descriptor.json) e
[resultados/inventários](../reports/validation_hf_publication_preparation.md).

Os campos descriptor_sha256/plan_sha256 dos vínculos privados usam o digest
canônico de JSON (canonical_digest). Inventários de arquivos usam SHA-256 dos
bytes completos, incluindo formatação. São hashes distintos, ambos rastreáveis;
não substituir um pelo outro ao preparar a confirmação operacional.
