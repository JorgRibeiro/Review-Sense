# Divulgação disponível — v1.0.0 de código e demonstração local

Plano alternativo solicitado para a hipótese de direitos dos pesos ainda incertos.
**Rascunho para revisão humana; nada foi postado.** Não anuncia modelo no Hub ou
inferência hospedada. A [decisão de direitos](RELEASE_RIGHTS_DECISION.md) permanece
REQUIRES_REVIEW; upload_authorized=false. Não altera a tag ou as notas da v1.0.0.

## Texto sugerido para LinkedIn

Concluí a primeira versão pública de código e documentação do ReviewSense PT-BR,
um projeto de classificação de sentimentos em avaliações brasileiras de produtos.

Implementei o percurso completo: auditoria e splits reproduzíveis, baseline
TF-IDF + LinearSVC, fine-tuning do BERTimbau e avaliação independente. O B2W-Reviews01
ficou com 126.692 avaliações após limpeza; negativo, neutro e positivo são classes
derivadas de estrelas, não anotadas manualmente.

No teste independente, o F1 Macro foi 0,7545 no BERTimbau e 0,7231 no SVM.
O treinamento aconteceu no meu computador, com uma RTX 3050 Laptop de 6 GB.
Esses resultados descrevem o experimento, sem alegação de superioridade estatística.

Além dos modelos, desenvolvi um serviço reutilizável de inferência, demonstração
local em Gradio, empacotamento com verificação de integridade e testes offline.
Os aprendizados incluíram seleção sem vazamento do teste, retomada do treinamento,
limites de memória e separação entre engenharia pronta e publicação autorizada.

A classe neutra ainda é uma limitação. O corpus é de produtos de 2018, sem validação
específica para delivery, e os scores não são calibrados.

O que está público nesta versão: código, documentação, relatórios e figuras.
Os pesos ajustados não são distribuídos e não há demonstração hospedada.
As imagens mostram a execução local com texto sintético. A preparação ZeroGPU
está no código, mas não foi implantada. A análise de direitos dos pesos continua aberta.

Dados: B2W-Reviews01, B2W Digital/Americanas Tech, de Livy Real, Marcio Oshiro e
Alexandre Mafra. Base: BERTimbau, NeuralMind, de Fábio Souza, Rodrigo Nogueira e
Roberto Lotufo. Projeto educacional, sem parceria institucional.

Código e documentação: https://github.com/jorgeribeiru/Review-Sense

Release: https://github.com/jorgeribeiru/Review-Sense/releases/tag/v1.0.0

## Capturas selecionadas e legendas

Reaproveitar estas capturas existentes, inspecionadas nesta execução, sem modificar
seus bytes. Foram produzidas na validação histórica da Fase 4A e usam a mesma
entrada sintética. Não são capturas de uma aplicação pública ou de um novo teste.

| Arquivo | Legenda sugerida |
| --- | --- |
| [Desktop](../reports/figures/phase4a_desktop.png) | “Demonstração local ReviewSense, captura histórica com texto sintético. Pesos não distribuídos; sem inferência hospedada. Scores não calibrados.” |
| [Móvel](../reports/figures/phase4a_mobile.png) | “Layout móvel da demonstração local, captura histórica com texto sintético. A URL pública de inferência não está disponível.” |
| [F1 Macro](../reports/figures/phase3c_f1_comparison.png) | “Teste independente concluído: F1 Macro 0,7545 BERTimbau e 0,7231 SVM. Comparação descritiva, sem recálculo nesta divulgação.” |

As latências visíveis nas capturas são observações históricas individuais; não
devem ser anunciadas como benchmark novo, desempenho ZeroGPU ou SLA. Há diferenças
visuais em relação à interface atual; não apresentar essas imagens como validação
recente. Nenhum arquivo de pesos, avaliação real, identificador ou cache acompanha
as imagens. Não foi criada nova captura/gravação, nem simulada uma predição.

## Revisão final antes de divulgar

1. O usuário revisa texto e imagens e decide separadamente sobre publicação.
2. Confirmar links de repositório/release e atribuições; manter o escopo de código.
3. Usar somente as figuras selecionadas e legendas correspondentes. Se houver
   gravação futura, mostrar só a UI local com entradas sintéticas, sem terminal,
   arquivos, credenciais ou avaliações reais, e identificar sua data/condições.
4. Não anunciar pesos para download, “demo online”, resultados novos ou parceria.
   Uma futura revisão de direitos não muda retroativamente o escopo da v1.0.0.

Não há post, agendamento, upload de mídia ou nova release autorizado por este rascunho.
