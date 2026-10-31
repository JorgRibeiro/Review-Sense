# Changelog

## 1.0.0

Primeira versão com escopo de código-fonte, documentação, resultados experimentais
e demonstração local. A preparação da versão não cria tag ou GitHub Release.

### Incluído

- Pipeline auditável para B2W-Reviews01: 126.692 avaliações após limpeza, splits
  reproduzíveis e validação por fingerprints, preservando acentos e negações.
- Classes derivadas de estrelas; DummyClassifier e TF-IDF + LinearSVC.
- Fine-tuning BERTimbau local na RTX 3050 Laptop 6 GB, seleção por validação e retomada.
- Avaliação independente congelada: F1 Macro de teste 0,7545 no BERT e 0,7231 no SVM.
- Inferência CPU/GPU, Gradio local, Model Card, empacotamento e integridade por SHA-256.
- Testes sintéticos/offline e adaptação ZeroGPU isolada, ainda sem deploy.
- README orientado ao produto, histórico técnico preservado e notas de release.
- Versão fonte do pacote Python alinhada a 1.0.0, sem alterações de algoritmo.

### Fora do escopo

- Distribuição de pesos ajustados, dataset ou inferência hospedada.
- Aprovação jurídica dos pesos: REQUIRES_REVIEW; upload_authorized=false.
- Novo treinamento, avaliação do teste, calibração ou benchmark nesta preparação.

[Guia e checklist](docs/RELEASE_v1.0.0.md) · [Notas de lançamento](docs/RELEASE_NOTES_v1.0.0.md)

## Desenvolvimento anterior

As entregas de dados, baseline, BERTimbau, avaliação e engenharia de inferência
precedem esta primeira release de código. Seus relatórios e commits permanecem
preservados; não se atribuem tags retroativas fictícias.
[Histórico técnico](docs/HISTORICO_TECNICO.md).
