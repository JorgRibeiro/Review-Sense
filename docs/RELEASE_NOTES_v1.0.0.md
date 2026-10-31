# ReviewSense PT-BR v1.0.0 — código e documentação

Primeira release do ReviewSense PT-BR com **código-fonte, documentação, resultados
experimentais e demonstração local** para classificação de sentimentos em português
brasileiro. **Não inclui pesos treinados, dataset ou inferência hospedada.**

## O que está disponível

- Pipeline B2W-Reviews01 com 126.692 avaliações após limpeza, auditoria e splits
  reproduzíveis. Classes negativo/neutro/positivo derivadas de estrelas, sem anotação manual.
- DummyClassifier, baseline TF-IDF + LinearSVC e fine-tuning BERTimbau.
- Treinamento local na RTX 3050 Laptop 6 GB, seleção por validação e retomada.
- Avaliação independente, serviço de inferência CPU/GPU, Gradio com exemplos sintéticos,
  Model Card, empacotamento por allowlist e verificação de integridade.
- Testes offline com artefatos sintéticos e preparação ZeroGPU, sem deploy.

## Resultados congelados

| Modelo | F1 Macro de teste |
| --- | ---: |
| BERTimbau | 0,7545 |
| TF-IDF + LinearSVC | 0,7231 |

Avaliação em 19.004 exemplos, após seleção pela validação. Ganho descritivo relativo
4,34%; não representa demonstração de superioridade estatística. Nenhum resultado
foi recalculado para esta release.

## Uso e limitações

O clone permite estudar código, relatórios e testes. **Inferência local exige
artefatos treinados legítimos que não acompanham o repositório**; não há download
automático ou link público de pesos ajustados. Os comandos e pré-requisitos estão
no README e no guia da versão.

Corpus de produtos de 2018, sem validação específica para delivery. A classe neutra
é limitante; notas são proxies, e os scores Softmax não são calibrados. Não há SLA
ou garantia de generalização. Testes locais não validam hardware ZeroGPU real.

## Licenças e disponibilidade

Código próprio MIT; avisos de terceiros preservados. Isso não estende a licença
aos dados ou aos pesos ajustados. Atribuições ao B2W-Reviews01/B2W Digital/Americanas
Tech e autores, e ao BERTimbau/NeuralMind e autores, constam na documentação.
**Pesos: license_status=REQUIRES_REVIEW; upload_authorized=false.**

Finalidade educacional/não comercial prevista. Política de transparência, reclamações
e eventual suspensão/retirada documentada, sem aprovação automática de direitos.
Mantida a decisão de não solicitar autorizações por e-mail. Sem parceria institucional.
Esta release de código não autoriza distribuir o modelo nem hospedar sua inferência.

Código e documentação: [jorgeribeiru/Review-Sense](https://github.com/jorgeribeiru/Review-Sense).
Consulte no repositório README.md, docs/RELEASE_v1.0.0.md e
reports/validation_release_v1.md para uso, procedimentos e verificação final.
Distribuição dos pesos e deploy ZeroGPU continuam fora desta versão.
