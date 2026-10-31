# Lançamento público — preflight e decisão da Etapa A

Executado em 31/10/2026 UTC (30/10/2026, America/Recife).
Branch `feat/hf-public-launch`, criada após atualizar a main local por fast-forward
até `98845aa323508aaa79b42d3fa9e547409a28558a` (PR #8).
Somente leitura remota e alterações locais de documentação; Etapas B–D não executadas.

## Estado público confirmado

Consulta GET pelo conector GitHub à release por tag: a
[v1.0.0](https://github.com/jorgeribeiru/Review-Sense/releases/tag/v1.0.0)
existe, draft=false, prerelease=false, sem assets anexados. Publicada em
2026-10-31T00:39:42Z. A tag local obtida por fetch resolve para o SHA acima;
o texto da release declara código/documentação/resultados/demonstração local,
sem pesos, dataset ou inferência hospedada. Nada foi alterado na release/tag.
O repositório é público e a API confirma has_issues=true; monitoramento humano
das reclamações não foi presumido. Não houve consulta autenticada à conta HF,
verificação de elegibilidade ou consulta/criação dos destinos HF planejados.

## Revisão documental e fontes

Revisados os guias 4B de licença/empacotamento; Model Card, atribuições e avisos;
guias 4C de direitos, hospedagem, lançamento, preparação e ZeroGPU; templates e
runtime do Space; relatórios 4B/4C e correções; README, guia/notas/changelog e
evidências v1.0.0. Os estados históricos não foram reescritos como resultados novos.

Fontes originais reconfirmadas por consultas públicas somente de documentação e
metadata: README B2W na revisão usada, Card BERTimbau fixado, licença MIT dos autores
fixada, metadata do espelho, texto/FAQ/orientação IA da CC e lei autoral no site da
Câmara. O acesso ao Planalto foi bloqueado; não foi contado como fonte lida.
A página oficial ZeroGPU foi consultada: a oferta geral gratuita permanece
condicionada à elegibilidade; isso não confirma a situação da conta do usuário.
Links e conclusão fundamentada: [RELEASE_RIGHTS_DECISION.md](../docs/RELEASE_RIGHTS_DECISION.md).

## Verificações executadas nesta operação

| Verificação | Resultado |
| --- | --- |
| Suíte completa em modo offline | **393 passed in 14.72s**, sem falhas/skips |
| Ruff: src tests app scripts deploy/zerogpu | **All checks passed!** |
| Ruff format --check nos mesmos diretórios | **57 files already formatted** |
| pip check | **No broken requirements found.** |
| Preflight local do pacote 4B | Integridade/metadata/Card/header/descritor válidos; nove arquivos |
| Bundle ZeroGPU em diretório temporário | **17 arquivos**, allowlist/hashes validados, sem pesos ou dados |

Preflight chamado diretamente com load_preflight_inputs/preflight existentes,
sem executar o runner de inferência nem preparar novo modelo. Os 13 links externos
e duas referências locais do pacote passaram a checagem offline contra o recibo
HTTP existente de 2026-10-30T22:31:09.034437+00:00. Isso não é nova auditoria HTTP
completa do pacote; as fontes da decisão e o estado público foram consultados
separadamente. Âncoras e disponibilidade futura não são garantidas.

Gates retornados: LEGAL_REVIEW_REQUIRED=true, DESTINATION_UNCONFIRMED=true,
PUBLISH_AUTHORIZATION_REQUIRED=true, remote_execution_allowed=false.
DESTINATION_UNCONFIRMED significa ausência de recibo/candidato/revisão Hub aprovados,
apesar dos nomes planejados explícitos no pedido. TECHNICALLY_READY=false é o
valor inicial desse preflight, que só o runner promove após equivalência/inferência;
não representa uma regressão das verificações reais anteriores, que não foram repetidas.

Suíte e preflight executaram sob test_access_policy(final=False), protegendo o
test.csv real. **Ambos registraram read_opens=0 e blocked_opens=0.** Não houve
abertura de CSVs reais, nem para hash. Fixtures sintéticas continuam nos testes.
Não houve inferência com os pesos treinados, treinamento ou benchmark novo.
Ambiente existente Python 3.11.14/PyTorch 2.14.1+cu132 preservado, sem instalação
ou substituição de dependências. pip check não valida instalação limpa ou ZeroGPU real.

## Integridade e preservação

Snapshot anterior às alterações: 180 hashes de todos os arquivos versionados e
artefatos locais selecionados. **179 permanecem iguais**; README é a única alteração
intencional entre os arquivos anteriores. Três documentos novos são acrescentados.
Código, testes, métricas, notebooks, descritor, manifesto, pacote 4B de nove arquivos,
checkpoint/tokenizer, baseline e marcadores finais anteriores permanecem intactos.
Os três splits mantêm inode/tamanho/mtime_ns, conferidos por stat, sem recalcular
fingerprints dos CSVs; esse controle de metadata é complementar.

| Referência preservada | SHA-256 |
| --- | --- |
| Peso ajustado, original e pacote | a65f1ee302987c5f0b01cd7981722e92f17ff332358b8e19b8dc01ee05753c70 |
| Manifest 4B | 836f5745422f2e58dcf2f4ee8f7e5b02fedc5f1d77d95e223a1683bb6e64eb6e |
| Descritor | bb51b7f6a0f774871975d744d0c7d582fd6301a324deb72951ec906e1a4aa117 |
| Métricas finais 3C | 8f32d5fa4dccd74b47f2b6c26455372bf7d50415390548eaa2085ed9e21bf302 |

README e três documentos novos compõem o commit. git diff --check e **36 destinos
Markdown locais**, incluindo a âncora do README, foram conferidos. Busca por padrões HF/GitHub/AWS, chaves privadas,
atribuições de credenciais e caminhos pessoais, junto da revisão dos textos: sem
ocorrências. Somente documentação é adicionada; nenhum dataset, peso, cache ou
recibo de aprovação real. Essa busca não é auditoria universal de segredos.

## Resultado e alternativa preparada

Manter **REQUIRES_REVIEW; upload_authorized=false**, sem licenciar os pesos por
inferência. Não produzir novos manifestos ou recibos: os bytes do candidato não
mudaram, e não há decisão de direitos aprovada. Não basta aceitar o risco para
declarar aprovação jurídica. A próxima decisão é manter o lançamento de código
ou buscar análise especializada que fundamente eventual distribuição específica.

[Divulgação e mídia](../docs/PUBLIC_LAUNCH_COMMUNICATION.md): rascunho de LinkedIn
restrito ao código já público; duas capturas históricas locais com texto sintético,
inspecionadas visualmente e selecionadas sem modificar bytes; gráfico agregado
existente como alternativa. Nenhuma captura/gravação nova ou postagem foi realizada.
O README passa a apontar à release existente e a esses documentos, sem link HF fictício.

Sem recibos de aprovação para o candidato real, uploader novo, login/token, e-mail, criação de repo/Space,
upload, deploy, contratação, push, PR remoto, merge desta branch ou nova tag/release.
O mecanismo de primeiro upload/recibos e a validação real do Space ficam condicionados
à resolução material de direitos e às autorizações específicas do usuário.
