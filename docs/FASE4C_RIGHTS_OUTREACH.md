# Fase 4C — Política de direitos, atribuição e reclamações

**REQUIRES_REVIEW; upload_authorized=false. Publicação continua bloqueada.**

## Decisão do responsável pelo projeto

O usuário decidiu **não encaminhar solicitações de autorização por e-mail**.
O pedido redigido na primeira execução da 4C não será enviado; deixa de ser uma
pendência operacional. Este documento substitui aquela orientação de contato.
O rascunho e a pesquisa anteriores permanecem no histórico Git e nos relatórios
da execução original, que não foram reescritos. Nenhuma mensagem foi enviada.

A decisão não concede direitos e não resolve a divergência de licenças. Não
interpretar silêncio, ausência de reclamação ou existência desta política como
consentimento. Não há envio automatizado de e-mail, formulário ou mensagem por
outro canal. A revisão material de direitos permanece um bloqueio de lançamento.

## Atribuição e transparência

- **Dataset:** atribuir B2W Digital/Americanas Tech, B2W-Reviews01, e os autores
  Livy Real, Marcio Oshiro e Alexandre Mafra. Preservar a referência ao
  [original fixado](https://github.com/americanas-tech/b2w-reviews01/tree/4639429ec698d7821fc99a0bc665fa213d9fcd5a)
  e ao [artigo](https://github.com/americanas-tech/b2w-reviews01/blob/main/b2wreviews01_stil2019.pdf).
  Autoria acadêmica, publicação institucional e titularidade das avaliações de
  consumidores são posições distintas; não se presume poder de autorização integral.
- **Modelo base:** atribuir NeuralMind e Fábio Souza, Rodrigo Nogueira e Roberto
  Lotufo, autores de BERTimbau (2020). Preservar a referência ao
  [base e revisão utilizados](https://huggingface.co/neuralmind/bert-base-portuguese-cased/blob/94d69c95f98f7d5b2a8700c420230ae10def0baa/README.md),
  a citação e o aviso MIT correspondente.
- **Licenças separadas:** a MIT do código próprio e a MIT declarada do modelo base
  não licenciam automaticamente os pesos ajustados. A fonte original do dataset
  declara CC BY-NC-SA 4.0; o [espelho](https://huggingface.co/datasets/ruanchaves/b2w-reviews01)
  declara CC BY 4.0, sem esclarecimento que resolva a divergência. Não adotar o
  espelho como permissão adicional. Ver [auditoria da 4B](FASE4B_LICENSE_REVIEW.md).
- **Alterações e limitações:** informar limpeza, rótulos por estrelas, fine-tuning,
  seleção do checkpoint e limites do corpus/classe neutra, sem redistribuir dados
  individuais. Não apresentar o projeto como parceiro/endossado por Americanas,
  NeuralMind ou iFood. Pesos mantêm `license: unknown` e `REQUIRES_REVIEW`.

## Finalidade prevista

Estudo, aprendizado e demonstração educacional, com finalidade não comercial
prevista: sem venda de pesos, cobrança de inferência ou publicidade na demonstração.
Portfólio profissional pode envolver interesses indiretos; não é classificado
automaticamente como NonCommercial. Uma futura mudança de finalidade, monetização,
patrocínio ou uso comercial exige nova análise e autorização pertinente.

A intenção não comercial não elimina atribuição, outras condições potencialmente
aplicáveis de NC/SA, direitos sobre avaliações de terceiros, privacidade ou obrigações
legais. A incidência dessas condições sobre o modelo exige análise do caso; não se
presume nem que todos os pesos sejam uma adaptação do corpus, nem que sejam livres
de direitos. Não houve auditoria exaustiva de memorização. Não redistribuir corpus,
splits, IDs, dados pessoais ou exemplos reais para divulgar o projeto.

## Canal para reclamações fundamentadas

Canal definido para a política: [Issues do ReviewSense](https://github.com/jorgeribeiru/Review-Sense/issues),
com assunto “Direitos — pedido de revisão”. Antes de qualquer lançamento, confirmar
que Issues está habilitado e que há um responsável humano por monitorar e tratar
as manifestações. A disponibilidade/monitoramento do canal ainda é uma pendência
de pré-lançamento; nenhuma issue foi criada por esta correção.

Pedir somente identificação do material/revisão/link contestados, natureza do
direito alegado, fundamento ou referência documental verificável, vínculo ou
representação do titular e medida solicitada. Não exigir nem publicar documentos
pessoais, credenciais, avaliações de consumidores ou informações confidenciais
em uma issue pública. Se forem necessárias evidências reservadas, combinar um
canal privado verificável por revisão humana antes de recebê-las; não há e-mail
privado inventado nem formulário de coleta habilitado nesta entrega.

O responsável deverá registrar datas, evidências, escopo e decisão em armazenamento
privado fora do Git; versionar, quando apropriado, somente resumo sem dados pessoais.
Verificar identidade/representação e documentos de maneira proporcional. Não prometer
prazo de resposta ou atendimento automático que não possa ser assegurado.

## Suspensão, retirada e decisão

1. Fazer triagem humana da reclamação e identificar artefatos/uso/revisões envolvidos.
2. Havendo indício fundamentado de infração ou risco relevante, suspender o acesso
   público e a inferência afetados enquanto se avaliam os fatos, quando apropriado.
   No estado atual nada está publicado: manter pesos e inferência sem acesso público.
3. Se cabível, retirar os artefatos afetados, revisar avisos/documentação e registrar
   a decisão e evidências. Medidas remotas e comunicações dependem de autorização
   específica; esta política não dispara exclusões, publicações ou respostas.
4. Resolver materialmente a pendência, com avaliação especializada quando necessária.
   Retomar uma distribuição somente após decisão documentada, condições aplicáveis
   atendidas e autorização humana para o escopo/candidato/destino exatos.

Um aviso de retirada ou o procedimento de reclamações **não resolve automaticamente
os direitos autorais**, não substitui permissões necessárias e não torna lícita uma
publicação antecipada. Não depender de uma reclamação para examinar os direitos.
A política não remove os gates: `LEGAL_REVIEW_REQUIRED`, destino e autorização
continuam pendentes; `upload_authorized` não é alterado automaticamente.

## Pendências que permanecem

Documentar uma base material suficiente para o uso/distribuição pretendidos e os
direitos de terceiros, definir condições/atribuições e licença dos pesos após essa
análise. Confirmar canal de reclamações e responsável, destino HF, infraestrutura
e autorização humana. Sem resolução suficiente, não publicar pesos nem hospedar
inferência pública, inclusive indiretamente. Manter código/documentação próprios
nos limites dos direitos aplicáveis. Não há solicitação por e-mail a realizar.
