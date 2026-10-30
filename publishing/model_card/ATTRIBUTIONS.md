# Atribuições e escopos de licença

- Base BERTimbau: NeuralMind, Fábio Souza, Rodrigo Nogueira e Roberto Lotufo.
  Fonte: https://huggingface.co/neuralmind/bert-base-portuguese-cased,
  revisão 94d69c95f98f7d5b2a8700c420230ae10def0baa.
  O aviso MIT completo está em BASE_MODEL_LICENSE.txt, preservado de
  https://raw.githubusercontent.com/neuralmind-ai/portuguese-bert/679423c38eaa267241abccc18465be1a25c7eff0/LICENSE.
- Fine-tuning, documentação e exemplo de inferência: contribuidores ReviewSense PT-BR.
  Fonte: https://github.com/jorgeribeiru/Review-Sense.
  PROJECT_CODE_LICENSE.txt cobre o código, inclusive o exemplo de inferência;
  não concede automaticamente licença dos pesos ajustados.
- Dados: B2W Digital/Americanas Tech, B2W-Reviews01,
  https://github.com/americanas-tech/b2w-reviews01,
  revisão 4639429ec698d7821fc99a0bc665fa213d9fcd5a, janeiro–maio de 2018.
  Declaração original: CC BY-NC-SA 4.0,
  https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en.
  Foram realizadas limpeza, deduplicação, derivação de classes e divisão dos dados.
  Não são incluídos arquivos de corpus, avaliações reais, IDs de usuários ou predições
  individuais. O conteúdo dos pesos não foi auditado exaustivamente quanto a memorização.
  O espelho https://huggingface.co/datasets/ruanchaves/b2w-reviews01 declara
  CC BY 4.0; essa divergência não foi tratada como autorização de relicenciamento.

Atribuição não indica patrocínio, parceria ou endosso. Pesos ajustados continuam
REQUIRES_REVIEW; veja DISTRIBUTION_STATUS.md. As licenças de escopo específico
incluídas não equivalem a uma licença geral aprovada para todo o pacote.
