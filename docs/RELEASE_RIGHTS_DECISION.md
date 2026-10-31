# Decisão de direitos — preflight do lançamento público

Consulta em 31/10/2026 UTC (30/10/2026, America/Recife).
Branch `feat/hf-public-launch`, base `98845aa323508aaa79b42d3fa9e547409a28558a`.

**Conclusão operacional: fundamento ainda insuficiente nesta execução para liberar
a distribuição dos pesos ou a inferência pública. Manter
license_status=REQUIRES_REVIEW e upload_authorized=false.**

Esta análise documental não é parecer jurídico nem recibo de aprovação. Não conclui
que a publicação seja necessariamente ilícita ou que uma autorização individual
seja sempre exigida. A instrução atual exige parar antes de upload e apresentar
a análise ao usuário; Etapas B, C e D não foram iniciadas. A
[release v1.0.0](https://github.com/jorgeribeiru/Review-Sense/releases/tag/v1.0.0),
de código/documentação/resultados/demonstração local, permanece preservada.

## Fontes e alcance verificados

| Fonte primária | Constatação e limite |
| --- | --- |
| [B2W-Reviews01, README da revisão utilizada](https://github.com/americanas-tech/b2w-reviews01/blob/4639429ec698d7821fc99a0bc665fa213d9fcd5a/README.md) | Declara CC BY-NC-SA 4.0 e crédito a B2W Digital. Não contém uma decisão específica sobre estes pesos ajustados |
| [Model Card BERTimbau na revisão utilizada](https://huggingface.co/neuralmind/bert-base-portuguese-cased/blob/94d69c95f98f7d5b2a8700c420230ae10def0baa/README.md) | Metadata MIT do modelo base; autoria NeuralMind/Souza/Nogueira/Lotufo |
| [MIT NeuralMind, revisão do aviso incluído no pacote](https://github.com/neuralmind-ai/portuguese-bert/blob/679423c38eaa267241abccc18465be1a25c7eff0/LICENSE) | Permissões amplas com preservação do copyright e licença. Não esclarece direitos de terceiros sobre o fine-tuning |
| [Dataset card do espelho](https://huggingface.co/datasets/ruanchaves/b2w-reviews01) | Metadata CC BY 4.0 divergente. Não foi identificada evidência de relicenciamento que substitua o original |
| [CC BY-NC-SA 4.0, texto jurídico](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en) | Seções 1–4: direitos concedidos, NC, atribuição, adaptação, SA e bases de dados; não concede direitos que o licenciador não detenha |
| [Orientação CC sobre IA](https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/) e [análise detalhada, maio/2025](https://creativecommons.org/wp-content/uploads/2025/05/Using-CC-licensed-Works-for-AI-Training.pdf) | Distingue recomendações conservadoras de incidência jurídica efetiva, dependente de fatos e jurisdição |
| [FAQ CC — NonCommercial](https://creativecommons.org/faq/#does-my-use-violate-the-noncommercial-clause-of-the-licenses) | Finalidade/contexto importam; tipo de usuário e ausência de cobrança não decidem isoladamente |
| [Lei 9.610/1998, texto atualizado da Câmara](https://www2.camara.leg.br/legin/fed/lei/1998/lei-9610-19-fevereiro-1998-365399-normaatualizada-pl.html) | Arts. 7º, 8º, 29 e 46: distinguir expressão protegida, dados/ideias, atos reservados e limitações específicas; não enquadra automaticamente IA |
| [MIT do código próprio](../LICENSE) | Cobre o código ReviewSense; não foi estendida aos dados ou pesos |

Foram consultados somente documentação e metadata públicos, sem conteúdo de
datasets, download de pesos ou consulta/autenticação da conta HF do usuário.
O Planalto recusou a consulta automatizada; o texto oficial foi lido na Câmara.

## Uso pretendido e candidato

Estudo, portfólio e demonstração de classificação negativo/neutro/positivo, com
textos sintéticos ou entradas legitimamente fornecidas. Intenção não comercial:
sem venda de pesos, cobrança de inferência, publicidade ou aquisição de hardware.
Não há parceria/endosso de Americanas, NeuralMind ou iFood.

Distribuição pretendida: modelo `JorgeRibeiru/reviewsense-ptbr-bertimbau`,
eventual demonstração `JorgeRibeiru/reviewsense-ptbr`. São destinos planejados pelo
usuário, não prova de titularidade autenticada, existência ou elegibilidade.
Não se assume que a promoção profissional seja automaticamente NonCommercial.
Uma mudança de finalidade/monetização exige nova avaliação.

Candidato local: `reviewsense-ptbr-bertimbau-v1`, pacote 4B `1.0.0-rc.1`,
nove arquivos. Checkpoint selecionado antes do teste: época 1/update 5543.
Peso preservado, SHA-256:
`a65f1ee302987c5f0b01cd7981722e92f17ff332358b8e19b8dc01ee05753c70`.
Não houve alteração de tokenizer, treinamento ou recálculo de métricas.
Não redistribuir corpus, splits, IDs, estados de treinamento ou avaliações reais.

## O que poderia fundamentar uma distribuição

A licença pública original é uma concessão real nos direitos cobertos: se for
suficiente para o uso pretendido e suas condições forem atendidas, não é preciso
inventar um novo pedido de autorização para exercer esses direitos. A divergência
do espelho não cancela essa concessão nem permite ignorar NC/SA. Não se presume
que cada avaliação seja protegida ou que o publicador do corpus detenha todos os
direitos sobre os textos. A auditoria deve examinar o alcance necessário ao caso.

Há duas rotas a fundamentar: cumprir as condições incidentes sobre os direitos
licenciados ou demonstrar por que determinado ato não exige essa permissão.
A abordagem conservadora da CC sugere observar NC ao longo do uso e SA ao
compartilhar modelos baseados em material SA. A própria CC alerta que isso pode
exceder o juridicamente obrigatório; não é prova de que todo peso seja adaptação.
Tampouco o caráter numérico dos parâmetros prova ausência de obrigações.

No Brasil, a proteção da organização de uma base não elimina eventuais direitos
sobre seu conteúdo (art. 7º, § 2º); ideias/métodos como tais têm tratamento distinto
(art. 8º). A relação entre esses dispositivos, os atos de reprodução/adaptação
(art. 29) e limitações (art. 46) exige enquadramento concreto. Não foi demonstrada
uma exceção aplicável a esta distribuição, nem se importou fair use estrangeiro
como autorização brasileira ou global. Isso não afirma inexistência de exceções.

### Atribuição e eventual ShareAlike

Preservar B2W Digital/Americanas Tech, B2W-Reviews01, Real/Oshiro/Mafra, revisão e
link da fonte; indicar limpeza, deduplicação, classes por estrelas e fine-tuning.
Preservar NeuralMind, Souza/Nogueira/Lotufo, revisão base e aviso MIT completo.
Usar as [atribuições existentes](../publishing/model_card/ATTRIBUTIONS.md), sem
endosso institucional, com referências às licenças e modificações aplicáveis.

Se o compartilhamento configurar material adaptado sujeito a SA, a licença do
adaptador deve respeitar a seção 3(b) da CC BY-NC-SA, inclusive versão
ou licença compatível admitida, avisos e ausência de restrições adicionais
incompatíveis. Atribuir autores não satisfaz sozinho NC/SA. A MIT do base exige
preservar seu aviso; não determina, por si, a licença de todo o conjunto ajustado.
Não se concluiu incompatibilidade automática entre esses componentes, nem se
atribuiu MIT, CC BY ou CC BY-NC-SA aos pesos apenas para preencher metadata.

## Incertezas materiais e memorização

- Não há evidência específica, no material examinado, que resolva a extensão dos
  direitos licenciados sobre contribuições dos consumidores e outros terceiros.
- Falta um enquadramento documentado do uso profissional/educacional em NC e das
  condições efetivamente incidentes na distribuição e hospedagem pretendidas.
- Não há conclusão fundamentada sobre adaptação, direitos sobre estes pesos e
  licença aplicável, considerando base, corpus e jurisdições relevantes.
- O classificador expõe três labels/scores, não gera avaliações textuais: isso
  reduz a exposição direta pelo formulário. Pesos públicos permitem outras formas
  de inspeção/uso. Não houve auditoria exaustiva de memorização, extração ou dados
  pessoais; excluir metadados das features não garante ausência de dados pessoais
  nos textos. Não foram conduzidos novos ataques ou experimentos nesta operação.

Essas lacunas não provam infração. Também não são resolvidas por hashes corretos,
inferência sintética bem-sucedida, license: unknown ou ausência de reclamação.
A falta de auditoria de memorização não é tratada como proibição universal; seu
alcance e relevância precisam ser avaliados para este candidato. Não há exigência
de retreinar ou abrir uma nova fase de ML para concluir o portfólio de código.

## Decisão humana e escopo de autorização

**Agora:** apresentar esta análise; manter pesos e inferência hospedada indisponíveis
por este fluxo e usar o [plano alternativo](PUBLIC_LAUNCH_COMMUNICATION.md).
Recomendação: preservar a v1.0.0 de código e, se desejada a distribuição dos pesos,
obter avaliação jurídica especializada do caso. Não serão enviados e-mails ou
solicitações de permissão. A opção por avaliação especializada não contrata serviço.

Para avançar à **Etapa B**, é necessária base material documentada sobre alcance
de direitos, finalidade/NC, eventual SA, licença/avisos e risco residual, seguida
de autorização humana explícita para preparar um candidato local específico.
Aceitar risco não substitui essa base nem representa aprovação jurídica. Evidências
confidenciais permanecem fora do Git; nenhuma aprovação ou recibo real foi criado.

Antes da **Etapa C**, pedir aprovação separada do candidato final: identidade/SHA
dos pesos, manifest/descritor atualizados se necessário, allowlist e hashes finais,
conta `JorgeRibeiru`, repo_id exato e visibilidade. O escopo deverá autorizar criação
do repo, upload somente desses arquivos, leitura/download e verificação sintética
independente. A revisão Hub gerada pelo upload será registrada e fixada nas
verificações; não inventar antecipadamente esse commit ou usar main como identidade.

Antes da **Etapa D**, pedir autorização própria para criar/publicar o Space exato,
usar o modelo/revisão aprovados e executar testes públicos sintéticos. Confirmar
elegibilidade ZeroGPU gratuita, runtime real e transporte seguro dos registros
operacionais privados, sem incluí-los nos 17 arquivos. Sem compra ou fallback
para outro modelo. Nenhuma dessas ações está autorizada nesta Etapa A.

Os gates atuais exigem recibos vinculados ao digest do descritor e a uma revisão
Hub conhecida. São controles de consumo/hospedagem, não um fluxo completo de
primeiro upload. Não há uploader operacional implementado. Após resolver direitos,
a sequência da primeira publicação e do provisionamento dos recibos precisará
de revisão localizada, sem SHA fictício, relaxamento de gates ou nova arquitetura.

## Reclamações, suspensão e retirada

Aplica-se integralmente a [política existente](FASE4C_RIGHTS_OUTREACH.md): Issues
“Direitos — pedido de revisão”, material/revisão/fundamento/vínculo do requerente,
sem documentos pessoais em público. A API do GitHub confirmou Issues habilitado;
responsável e rotina de monitoramento ainda precisam de confirmação humana.
Não foi criada issue nem enviada mensagem. Suspender o acesso afetado durante
análise quando apropriado; retirar se cabível e registrar a decisão com evidências
privadas. Reativar somente após resolução material e autorização correspondente.

Um aviso de retirada não sana automaticamente direitos autorais. Retirar do Hub
também não garante apagar cópias já baixadas ou revogar licenças validamente
concedidas. Intenção não comercial, silêncio e aceite de risco não substituem
permissões/condições aplicáveis. O código próprio continua MIT; não há licença
global nova dos pesos ou dos dados.

Verificações desta execução: [preflight e preservação](../reports/validation_public_launch_preflight.md).
