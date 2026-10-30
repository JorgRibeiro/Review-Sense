# Fase 4B — Revisão de licenciamento

**Status dos pesos ajustados: `REQUIRES_REVIEW`. Upload autorizado: `false`.**
Escopo: preparar localmente um candidato de distribuição; não autorizar publicação.
Consulta das fontes em 30/10/2026. Esta revisão documental separa fatos verificáveis
de questões jurídicas que dependem do caso concreto e da jurisdição.

## Camadas de direitos e evidências

| Camada | Declaração encontrada | Alcance desta constatação |
| --- | --- | --- |
| Código ReviewSense | MIT, arquivo `LICENSE` do projeto; copyright 2026 contributors | O próprio arquivo limita sua cobertura ao código; não licencia os dados ou todos os pesos |
| Modelo base BERTimbau | MIT na [revisão original do Model Card](https://huggingface.co/neuralmind/bert-base-portuguese-cased/blob/94d69c95f98f7d5b2a8700c420230ae10def0baa/README.md) | Declaração do fornecedor para o modelo base; não resolve novos direitos relacionados ao fine-tuning |
| Código/artefatos NeuralMind | [MIT no repositório dos autores](https://github.com/neuralmind-ai/portuguese-bert/blob/master/LICENSE) | Preservar o aviso de copyright e o texto da licença correspondente; a cópia incluída no pacote tem hash rastreável |
| B2W-Reviews01 original | [README da fonte original](https://github.com/americanas-tech/b2w-reviews01/tree/4639429ec698d7821fc99a0bc665fa213d9fcd5a) declara CC BY-NC-SA 4.0 | Referência adotada pelo projeto para os dados de treinamento |
| Espelho no Hub | [Dataset card de ruanchaves](https://huggingface.co/datasets/ruanchaves/b2w-reviews01/blob/main/README.md) declara CC BY 4.0 | Há divergência; não foi encontrada autorização de relicenciamento que resolva a diferença |
| Pesos ReviewSense fine-tuned | Não há licença conclusiva aprovada | Não declarar automaticamente MIT nem CC BY-NC-SA; publicação permanece bloqueada |

O espelho é uma referência de catálogo do corpus, não a fonte jurídica usada para
substituir as condições do original. A execução utilizou CSV/revisão/checksum
originais registrados na Fase 1, sem usar dados do espelho como nova fonte.

## Obrigações potencialmente relevantes

Quando uma utilização depende dos direitos licenciados, a CC BY-NC-SA exige
atribuição, preservação dos avisos aplicáveis, indicação de modificações e uso
não comercial. O ShareAlike alcança material adaptado quando seus pressupostos se
aplicam. Direitos de bases de dados, privacidade, personalidade e marcas podem
exigir análise separada. O resumo não substitui o [texto jurídico, seções 1–4](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en).

As permissões da MIT do base não removem condições eventualmente incidentes sobre
o uso dos dados. Incluir o texto MIT em `BASE_MODEL_LICENSE.txt` preserva o aviso
do fornecedor; esse nome e a atribuição deixam explícito que ele não constitui
uma decisão sobre a licença integral dos pesos ajustados.

## Treinamento implica a mesma licença nos pesos?

Não é possível concluir isso apenas pelo metadata do dataset. A
[orientação específica da Creative Commons para treinamento de IA](https://creativecommons.org/wp-content/uploads/2025/05/Using-CC-licensed-Works-for-AI-Training.pdf)
reconhece variação de leis, exceções e fatos relevantes. Sua abordagem conservadora
recomenda observar NC ao treinar/compartilhar modelos e SA ao compartilhar modelos
baseados em material SA, mas declara que isso pode exceder o juridicamente exigido
e não decide quando há proteção ou adaptação. Essa orientação apoia cautela,
não uma conclusão automática de que estes pesos são uma obra derivada.

Também não há base para concluir o inverso: parâmetros numéricos não provam ausência
de material protegido, memorização ou obrigações. Não houve auditoria exaustiva de
memorização nem análise jurídica específica do ReviewSense nesta fase.

## Pendências concretas

1. Identificar os titulares relevantes e o alcance da autorização da B2W/Americanas,
   inclusive conteúdo de avaliações de terceiros e direitos eventualmente remanescentes.
2. Obter esclarecimento escrito sobre distribuição pública dos pesos ajustados e
   demonstração hospedada; esclarecer condições para exploração comercial, quando
   pretendida. Não presumir que acesso gratuito ou portfólio é automaticamente NC.
3. Avaliar, com orientação especializada quando necessário, direitos aplicáveis ao
   treinamento e à distribuição nas jurisdições relevantes. Não transportar uma
   exceção de outra jurisdição como autorização brasileira ou global.
4. Definir a licença dos pesos após essa análise, incluindo compatibilidade com os
   avisos do base, atribuições e eventuais condições NC/SA.

São perguntas em aberto; não se presume que qualquer organização, isoladamente,
possa conceder direitos que não detenha. A [FAQ da CC](https://creativecommons.org/faq/#does-my-use-violate-the-noncommercial-clause-of-the-licenses)
explica que NC depende da finalidade/contexto, não só do tipo de usuário ou de cobrança.

## Estratégias e decisão

É possível manter a demonstração local e publicar somente código/documentação
nos limites de seus direitos, sem distribuir pesos ou corpus. Outra opção é
buscar esclarecimentos/permissões suficientes antes de preparar uma licença final.
Uma alternativa futura de dados com direitos mais claros exigiria outro experimento,
fora desta fase; não se retreina este modelo para contornar a pendência.

Antes de `READY`, registrar o parecer/autorizações pertinentes, finalidade e
jurisdições cobertas, destinatários, atribuições e licença aprovada. Antes de upload,
também exigir autorização humana explícita na Fase 4C e destino confirmado.
Se houver impedimento identificado, registrar `BLOCKED`; com a incerteza material
atual, manter **`REQUIRES_REVIEW`** e bloquear publicação.

O Model Card usa `license: unknown`, identificador previsto na
[lista oficial do Hub](https://huggingface.co/docs/hub/repositories-licenses).
Esse campo comunica uma pendência, não concede direitos. Pacote tecnicamente
válido, tag de licença ou validação de YAML não são autorização jurídica.
