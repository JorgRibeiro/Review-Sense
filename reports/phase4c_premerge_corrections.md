# Fase 4C — Correções cirúrgicas pré-merge

**Publicação bloqueada: REQUIRES_REVIEW; upload_authorized=false. A Fase 4C não está encerrada.**
Branch `feat/phase4c-release-readiness`, revisão sobre `8350d26`. Este registro
complementa os relatórios anteriores, que permanecem inalterados.

## Ajustes

1. `app/config.py`: configuração hosted sempre desabilita SVM, inclusive construção
   direta com enable_svm=True. Configurações locais mantêm True/False conforme solicitado.
   Testes cobrem o default, a construção direta e o seletor da UI com configuração hosted.
2. Política de direitos atualizada em `docs/FASE4C_RIGHTS_OUTREACH.md`, guia,
   plano de gates, README e plano técnico. O usuário não encaminhará solicitações
   por e-mail; o rascunho anterior não será enviado. Atribuição a dataset/base,
   finalidade educacional/não comercial, incertezas, reclamações fundamentadas e
   eventual suspensão/retirada documentadas, sem presumir autorização ou dispensa
   de obrigações. Issues do projeto é o canal definido, sujeito à confirmação de
   disponibilidade/monitoramento humano antes de lançar. Nenhuma mensagem enviada.
3. `src/reviewsense/hub_cache.py` materializa cache local; `HubSnapshot` passa a
   utilizá-lo sem snapshot_download/API/rede. `LocalPackage` não foi alterado.

## Investigação e segurança do cache

Inspeção local da instalação huggingface-hub 2.4.0, especialmente
`file_download._create_symlink` e `_snapshot_download`: snapshots por commit usam
links relativos para blobs; a implementação anterior os entregava ao validador
que corretamente rejeita symlinks. A correção separa entrada do cache e pacote
regular validado, sem relaxar esse validador.

O repo_id e SHA continuam explícitos, confirmados e sujeitos aos mesmos gates.
O layout lido é `models--<namespace>--<repo>/snapshots/<sha>/`. Só se aceitam links
folha exatos `../../blobs/<etag>` de 40/64 caracteres hexadecimais e arquivos
regulares. Diretórios são abertos por dir_fd e O_NOFOLLOW em cada ancestral;
blobs não podem ser symlinks, cadeias ou arquivos não regulares. O_NONBLOCK evita
bloqueio em FIFOs antes da conferência de tipo. Cada cópia tem limite de tamanho
e SHA-256 conforme descritor aprovado. Entradas extras do cache não são abertas
nem copiadas; no pacote materializado a allowlist é exata.

A cópia é temporária e só é promovida após validação completa, sob
`cache_root/reviewsense-materialized/<repo>/<sha>/<digest-do-descritor>/`.
Não se sobrescreve nem se repara saída existente: sua reutilização exige hashes,
configuração, metadados e status válidos. Não há download ou fallback para main.
Os testes representam cache realista com links para blobs e etags Git/SHA-256;
a integração HubSnapshot usa apenas aprovações fictícias vinculadas a modelo
minúsculo sintético. Não foram criadas aprovações para os pesos reais.

## Validação executada

- **342 testes aprovados** em 13.63 s; 22 casos adicionais sobre a suíte de
  320. Testes antigos do adaptador foram adaptados para verificar materialização
  e ausência total de chamadas Hub; os demais continuam preservados.
- Ruff, formatação dos 47 arquivos Python e pip check aprovados.
- Suíte executada com HF_HUB_OFFLINE=1 e TRANSFORMERS_OFFLINE=1 e guarda no
  test.csv real: zero leituras e zero tentativas bloqueadas.
- Testes específicos bloqueiam rede e fazem chamadas Hub/token falharem caso
  sejam acionadas. Nenhuma operação de rede foi executada nesta correção.
- 65 arquivos anteriores mantiveram hashes: relatórios históricos, métricas,
  publicações/descritor/manifest, pacote 4B, modelos selecionados e registros de
  avaliação anteriores. Os três CSVs mantiveram inode/tamanho/mtime; nenhum CSV
  foi aberto. A AST de LocalPackage permaneceu idêntica.
- Nenhum treino, alteração de pesos/tokenizer, nova métrica de ML, upload, deploy,
  autenticação, push, merge, release ou envio de mensagem.

Reprodução dos checks:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/pytest -q
.venv/bin/ruff check src tests app scripts
.venv/bin/ruff format --check src tests app scripts
.venv/bin/python -m pip check
```

## Limitações e pendências

Materialização requer POSIX com dir_fd/O_NOFOLLOW; outras plataformas falham de
forma segura. Mudanças futuras do layout Hub precisam de revisão. O cache root e
o descritor aprovado são entradas confiadas ao operador, não provas criptográficas
de autoridade ou suficiência jurídica. Não houve download/publicação ou teste em
Space real, nem repetição da equivalência com os pesos grandes: os artefatos
reais continuam byte a byte iguais e essa correção usa testes sintéticos offline.

Direitos, destino HF, elegibilidade/recursos de hospedagem, autorização humana e
confirmação/monitoramento do canal de reclamações permanecem pendentes. A política
de retirada não resolve direitos autorais, e finalidade não comercial não elimina
obrigações. Sem resolução material, não publicar pesos nem inferência pública.

Evidência estruturada: [phase4c_premerge_corrections.json](phase4c_premerge_corrections.json).
