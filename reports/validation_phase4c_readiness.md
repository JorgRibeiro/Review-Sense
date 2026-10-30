# Fase 4C — Validação da preparação local

**Fase 4C — Engenharia de preparação concluída; publicação bloqueada por revisão de direitos.**

A Fase 4C não está encerrada. `REQUIRES_REVIEW`, `upload_authorized=false`,
destino HF não confirmado e nenhuma autorização humana de publicação registrada.
Branch `feat/phase4c-release-readiness`, criada da main atualizada `562e0fb`.
Pipeline/preflight executado no commit `77755f92fd2474de718ce33e4a74418f80c45054`; suíte final no
commit `d618d931a3efc1a10be5237f5e62ccb839a02e02`. O commit intermediário só amplia a sanitização
de configuração na CLI e seus testes; módulos de inferência/resolução/preflight
permanecem com os mesmos hashes usados na execução real. Nenhuma alteração de
modelo ou métrica final.

## Situação e entregas

- Pedido formal em português e canais institucionais: [direitos](../docs/FASE4C_RIGHTS_OUTREACH.md).
  Preparado para revisão humana; nenhuma mensagem enviada. Autoria acadêmica não
  é tomada como titularidade integral ou representação do licenciador.
- Resolução verificada local e adaptador de cache Hub opcional em
  `src/reviewsense/model_resolution.py`. Repo_id e SHA obrigatórios; gates antes
  da chamada. `local_files_only=True`, `token=False` invariantes; nenhum download.
- [Descritor público](../publishing/inference_descriptor.json) fora do pacote:
  identidade, versão, arquitetura, labels, truncamento, licença, Card e inventário.
  Não depende de relatórios internos e não contém caminhos pessoais ou dados.
- [Hospedagem](../docs/FASE4C_HOSTING_DECISION.md) pesquisada em fontes oficiais:
  CPU Basic condicionado a plano/elegibilidade, ZeroGPU a elegibilidade e adaptação.
  Sem conta consultada, recursos contratados ou preços/latências inventados.
- Preflight offline em `src/reviewsense/release_readiness.py` e
  `scripts/verify_phase4c_readiness.py`; UI/configuração em `app/app.py` e `app/config.py`.
- README e plano técnico atualizados; [guia](../docs/FASE4C_RELEASE_READINESS.md)
  documenta funções, reprodução, limites e os dois cenários de encerramento.

## Gates efetivamente observados

| Saída | Valor | Interpretação |
| --- | --- | --- |
| TECHNICALLY_READY | true | Candidato atual passou integridade e execução sintética |
| LEGAL_REVIEW_REQUIRED | true | Revisão material de direitos não resolvida |
| DESTINATION_UNCONFIRMED | true | Repo_id/revisão públicos não definidos |
| PUBLISH_AUTHORIZATION_REQUIRED | true | Autorização humana específica ausente |

Gates não são uma certificação jurídica. Recibos privados ficam vinculados ao
candidato e evidência documental; não são assinaturas digitais. Não há `--force`
ou uploader. Hospedagem é bloqueada inclusive na construção direta da UI e
na CLI, antes de servir. O caminho remoto de cache foi testado só com mocks e
artefatos pequenos sintéticos; não houve chamada real ao snapshot do Hub.

## Integridade e equivalência real

O candidato 4B continua `1.0.0-rc.1`: nove arquivos, 436.422.685 bytes.
Manifest SHA-256: `836f5745422f2e58dcf2f4ee8f7e5b02fedc5f1d77d95e223a1683bb6e64eb6e`.
Fingerprint do pacote: `dc7a58eee7919923f396eef80098e6abf1f86e4727649888b2aadec6a4fe5471`.
Descritor externo SHA-256: `bb51b7f6a0f774871975d744d0c7d582fd6301a324deb72951ec906e1a4aa117`.
Peso: `a65f1ee302987c5f0b01cd7981722e92f17ff332358b8e19b8dc01ee05753c70`.

Não houve mudança de allowlist ou cópia/reescrita do pacote original. Foram
verificados nome/tamanho/SHA, Card, header safetensors, metadados, labels,
truncamento e ausência de extras/duplicatas/caminhos inseguros.

| Comparação original/pacote | Classes | Máx. diferença logits | Máx. diferença scores | Tolerâncias atol / rtol |
| --- | --- | ---: | ---: | --- |
| CPU / FP32 | 5/5 iguais | 0.0 | 0.0 | 1e-6 / 1e-5 |
| GPU / FP16 | 5/5 iguais | 0.0 | 0.0 | 1e-3 / 1e-3 |

Cada comparação usa o mesmo dispositivo/precisão; não se afirma igualdade entre
CPU e GPU. São cinco textos sintéticos, incluindo texto longo que atinge 256 tokens,
não um dataset anotado. Logits/scores finitos, tokenizer cased e mapeamentos corretos.
O serviço novo também executou cinco entradas por modo com pré-carga e cache.
O exemplo literal do Model Card executou isolado em CPU, fora do repositório,
cache vazio, zero acesso a origens internas e nenhuma conexão de rede.

Dry-runs: zero conexões, zero leituras/tentativas de abrir test.csv, nenhuma
métrica de qualidade recalculada. PyTorch 2.14.1+cu132, Transformers
5.22.1, Gradio 6.31.0, tokenizers
0.23.3, huggingface-hub 2.4.0.
Nenhuma dependência foi instalada/substituída. Não medir startup hospedado com
estes resultados; inferência sintética valida engenharia, não generalização.

## Aplicação e testes

**320 testes aprovados** em 13.15 s:
253 anteriores preservados + 67 novos, zero falhas/skips.
Suíte final executada offline sob guarda do CSV de teste: zero aberturas e zero
bloqueios. Ruff, formatação (45 arquivos Python) e pip check aprovados.

Cobertura inclui contrato/metadados, hash/allowlist, arquivo ausente/adulterado,
symlinks/ancestrais/raiz/travessia, segredos conhecidos, conteúdo duplicado,
revisão imutável, gates independentes, recibos ausentes/adulterados/desatualizados,
cache offline mock, falha segura, ausência de treino e alterações de pesos,
comparação sintética e configuração/sanitização de erros da aplicação.

Smoke adicional em **127.0.0.1:7863, share=False**: BERT CPU e SVM via HTTP,
entrada vazia, limpeza do resultado e erros públicos seguros. Navegador Brave
headless: botão sintético e inferência, zero erros JS e requisições externas,
320/390/768/1440 px sem overflow horizontal. Servidor temporário encerrado.
Não foi executado deploy ou servidor exposto à internet. A UI hospedada foi
inspecionada apenas em teste com serviço/gate simulados, sem launch.

A primeira sondagem de navegador usava um seletor textual incompatível com a quebra
`br` do título e expirou; a sonda passou ao aguardar o elemento `.sentiment-pill`,
sem mudança no comportamento de inferência. A falha foi do seletor da verificação,
não é contada como resultado aprovado anterior.

## Links, fontes e limites

Treze URLs do Card/avisos responderam 200 na auditoria pública separada. O dry-run
usa o [recibo versionado](../publishing/readiness_links.json) e confere duas
referências locais, sem acessar rede. Âncoras e disponibilidade futura não garantidas.
As páginas oficiais de hospedagem responderam 200. Os canais institucionais RI
foram lidos pelo navegador de pesquisa, mas o cliente HTTP direto recebeu 403:
reconfirmar manualmente no envio. Não se presume que esse status prove desativação.
O conector HF de documentação retornou ferramenta indisponível; consultas somente
leitura a páginas oficiais foram o fallback, sem autenticação.

A detecção de segredos é limitada a padrões/header/metadados; não audita todos os
possíveis segredos ou memorização nos tensores. Cache padrão com symlinks é recusado;
materialização regular e provisionamento remoto dependerão de etapa autorizada.
ZeroGPU e hardware hospedado não foram exercitados. Não há latência de Space
medida nem elegibilidade de conta confirmada. Revisão resolvida exigirá um novo
candidato com licença/avisos aprovados e revisão do protocolo de pré-publicação.

## Preservação e pendências

62 arquivos anteriores mantiveram SHA-256, incluindo pacote e manifest
4B, Model Card/avisos, relatórios/resultados 3C e anteriores, marcador/resultado
local da avaliação final, pesos/tokenizer selecionados e SVM. Inode/tamanho/mtime
dos três CSVs permaneceram iguais; nenhum conteúdo de CSV foi aberto nesta fase.
Não houve treino, refit, calibração, recálculo de F1, push, merge, tag, release,
criação de repo/Space, upload, login, acesso a token, contratação ou LinkedIn.

Próximos passos humanos: revisar/enviar o pedido, verificar autoridade/alcance da
resposta e resolver direitos; confirmar namespace, revisão, conta/recursos e
orçamento; revisar candidato e obter autorização específica. Depois, em etapa
posterior, publicar e testar ambiente limpo/Gradio real, links e release.
Se direitos continuarem pendentes, manter pesos/inferência privados, disponibilizar
somente código/documentação próprios nos limites aplicáveis; gravação sintética
sujeita à revisão de direitos. Nunca contornar por hospedagem indireta.

O post definitivo de LinkedIn não foi escrito. Seus critérios de encerramento
estão preservados no guia, para depois dos materiais públicos verificados.

Evidência completa: [phase4c_readiness_verification.json](phase4c_readiness_verification.json).
