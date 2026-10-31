# Publicação HF — execução parcial interrompida

Data UTC: 2026-10-31T01:48:39.458960+00:00. Branch feat/hf-public-launch.
[Registro estruturado](hf_publication_execution.json).

**Modelo publicado; verificação pós-download incompleta; Space não criado.**
A sequência foi interrompida na primeira falha de verificação, conforme instrução
expressa do responsável. Não se declara lançamento/deploy concluído.

## Autorização e identidade

O responsável autorizou expressamente os dez arquivos do modelo e os 17 do Space,
com visibilidade pública, sem gastos, vinculados ao plano canônico
`4c9baf15fd7a4e4a503690cbeee9d529d46785f698914188eb2590f82be3a340` e aos commits ff1c7e0/a376fce.
A confirmação verdadeira foi guardada em outputs ignorado, diretório 0700 e arquivos
0600. Não acompanha o modelo, o código do Space ou este relatório. Nenhum parecer
ou nova permissão de terceiros foi criado. A situação jurídica continua
**REQUIRES_REVIEW / NOT_ESTABLISHED**, separada da autorização operacional.

Inicialmente não havia credencial local. O usuário fez login pelo terminal; a API
confirmou a conta **JorgeRibeiru**, via OAuth. Esse fluxo não expõe o campo legado
accessToken.role; a permissão efetiva foi verificada por auth_check(write=True)
no servidor antes de enviar conteúdo. Tokens não foram impressos, copiados para
relatórios ou enviados no pacote. Nenhuma instalação ou troca de dependências.

## O que foi efetivamente publicado

- Modelo público: [JorgeRibeiru/reviewsense-ptbr-bertimbau](https://huggingface.co/JorgeRibeiru/reviewsense-ptbr-bertimbau).
- Revisão real: `ae89cf168ffaa64115c097204e678ed513a54265`.
- [Commit publicado](https://huggingface.co/JorgeRibeiru/reviewsense-ptbr-bertimbau/commit/ae89cf168ffaa64115c097204e678ed513a54265).
- Enviados exclusivamente os **dez arquivos aprovados**, por operações explícitas;
  não foi usado upload de uma pasta ampla ou do repositório de desenvolvimento.
- O Hub mantém adicionalmente .gitattributes gerido pela plataforma, previsto no
  plano; nenhum arquivo adicional do projeto foi enviado. Há 11 entradas remotas.
- O repositório permanece público nessa revisão. Não foi apagado, alterado para
  privado ou modificado após a falha; tais ações não foram executadas como rollback.

O upload foi precedido de validação local do candidato/manifest/descritor e do
plano consentido. A API confirmou propriedade/visibilidade, escrita e commit real.
A declaração LFS do servidor está no JSON; não equivale a checksum calculado de
um download local validado. O peso local aprovado permanece SHA-256
`a65f1ee302987c5f0b01cd7981722e92f17ff332358b8e19b8dc01ee05753c70`.

## Falha da verificação pós-upload

O verificador usou a revisão real acima, token=False e cache temporário vazio fora
do repositório. A consulta pública de metadata e snapshot_download retornaram.
Na etapa seguinte, materialize_hub_snapshot recusou abrir o blob dos pesos:

```text
OSError: [Errno 40] Too many levels of symbolic links
blob: a65f1ee302987c5f0b01cd7981722e92f17ff332358b8e19b8dc01ee05753c70
location: reviewsense.hub_cache._copy_approved_file
operation: os.open(... O_NOFOLLOW ...)
```

O contrato admite links folha do snapshot para blobs regulares do mesmo repo.
A recusa é compatível com um symlink na entrada de blob, mas seu destino e layout
exatos **não foram inspecionados**. O diretório temporário foi removido pelo
context manager ao propagar o erro. Não se inventa causa específica de Xet/CAS,
loop ou caminho de destino sem essa inspeção.

A falha não comprova corrupção dos pesos ou divergência de SHA: a validação integral
do download **não chegou a concluir**. Não houve nova tentativa, alternativa de
download, alteração de flags, relaxamento de LocalPackage/O_NOFOLLOW ou execução
de inferência com arquivos não validados. O erro foi preservado como impedimento,
sem convertê-lo em resultado aprovado.

**Inferência independente pós-download: não executada.** A falha ocorreu antes do
exemplo do Model Card e dos cinco casos sintéticos planejados. As equivalências
CPU/GPU e 432 testes locais anteriores continuam como evidências históricas; não
substituem esta verificação, que falhou. Não foram recalculadas métricas de ML.
A guarda do test.csv real estava instalada na tentativa; o processo que falhou
não chegou a gravar seu resumo final de auditoria. Não se reutilizam contadores
anteriores como se fossem dessa tentativa. Não houve leitura de CSV pelo fluxo;
a conferência local de preservação posterior registrou zero aberturas/bloqueios.

## Space, build e custos

- Space não criado nem enviado; **build não iniciado**.
- Nenhum Secret operacional configurado remotamente; nenhum recibo de consumo
  com SHA foi produzido antes de uma verificação bem-sucedida.
- Não há URL pública de inferência verificada, SHA de código do Space ou resultado
  de inferência ZeroGPU a apresentar.
- Matriz real Python/Torch/spaces, scheduler, VRAM, quotas e layout móvel não foram
  exercitados no provedor. Os alvos estáticos não são versões reais medidas.
- A conta foi identificada como não PRO; a listagem de hardware retornou ZeroGPU
  (enum zero-a10g) com custo unitário 0. Foi somente leitura: nenhum recurso de
  compute foi contratado/alocado, nem assinatura/crédito adquirido.

## Preservação, documentação e continuidade

Os dez arquivos locais do modelo e os 17 do Space ainda correspondem exatamente
ao inventário consentido. Nenhum código de runtime, teste, manifest ou descritor
aprovado foi alterado para tentar fazer a operação passar. Pacote 4B, checkpoint,
tokenizer, baseline, relatórios/métricas e marcadores históricos foram conferidos
por hash e continuam iguais. Os splits mantêm inode/tamanho/mtime_ns, por stat,
sem abrir CSVs nem recalcular seus fingerprints. Main e tag histórica v1.0.0
continuam no commit 98845aa, com escopo de código/documentação preservado.

README e documentos de estado receberam somente a atualização factual da execução;
os novos JSON/Markdown registram o ocorrido. O rascunho anterior de LinkedIn foi
marcado como histórico e não foi publicado. Não houve push, merge, nova tag/release,
e-mail, treino, alteração de métricas ou envio de arquivos diferentes dos aprovados.

A continuidade exige investigar o layout real do blob e preparar uma correção
localizada/verificável, sem enfraquecer os controles. Se houver mudança nos bytes
dos 17 arquivos do Space, seu novo inventário precisa de revisão antes de upload.
Nenhuma correção ou repetição foi executada nesta operação: permanece interrompida,
como solicitado. Não é necessária nova discussão jurídica para explicar esta falha técnica.
