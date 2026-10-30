# Fase 4A — Serviço de inferência e demonstração Gradio local

O serviço utiliza exclusivamente os artefatos aprovados. Não lê datasets, treina,
calibra, escolhe checkpoints ou repete a avaliação da Fase 3C. A interface não
conhece tokenização, pesos ou pós-processamento: chama `InferenceService.predict`.
Publicação e Model Card permanecem para a Fase 4B.

## Instalação e execução

Na raiz do repositório, usando a `.venv` com PyTorch CUDA já funcional:

```bash
.venv/bin/python -m pip install -r app/requirements.txt
.venv/bin/python -m app.app
```

Abra **http://127.0.0.1:7860**. O padrão é `--device auto`: GPU se disponível;
CPU se indisponível. A primeira análise carrega apenas o modelo selecionado.
Para escolher explicitamente o dispositivo:

```bash
.venv/bin/python -m app.app --device cpu
.venv/bin/python -m app.app --device cuda --port 7862
```

Execute uma instância por porta. Em `cuda`, uma falha é informada e não muda o
dispositivo explicitamente solicitado. Em `auto`, falhas CUDA reconhecidas permitem
uma única recuperação em CPU com os mesmos pesos, depois reutilizados no cache.
Logits não finitos interrompem a análise; não acionam uma tentativa de produzir
outro resultado. Não há sequência ilimitada de tentativas.

Gradio 6.31.0 e cliente 2.7.2 foram acrescentados. PyTorch 2.14.1+cu132,
Transformers 5.22.1, tokenizer e stack de ML anteriores foram preservados. O único
pacote existente substituído foi **Pillow 13.0.0 → 12.3.0**, porque Gradio requer
Pillow <13. Os locks históricos não foram modificados; não os instale junto com
os requirements da aplicação, pois eles fixam Pillow 13. O arquivo da aplicação
inclui a constraint CUDA existente e não escolhe uma variante CPU do PyTorch.

## Artefatos necessários

Pesos não estão no Git. Disponibilize cópias dos artefatos já aprovados das fases
anteriores, sem gerar novo treinamento ou substituí-los pelo checkpoint base:

```text
models/baseline/phase2-v1/
  selected_model.joblib
  results.json
  config.json
models/bertimbau/phase3b-full-v3/best_model/
  model.safetensors
  config.json
  tokenizer.json
  tokenizer_config.json
```

Também são necessários os relatórios versionados `baseline_validation.json`,
`bertimbau_phase3b_metrics.json` e `phase3c_final_metrics.json` em `reports/`.
Eles fornecem checksums de referência, não exemplos para inferência. Nenhum CSV,
split, cache de tokens ou checkpoint do otimizador é necessário para usar o app.

Os arquivos do BERT são verificados contra os hashes das Fases 3B/3C. Seu registro
de seleção deve apontar ao update 5543. O SVM confere o registro anterior e usa
`load_experiment`, que verifica também o checksum antes de desserializar o Joblib
local confiável. Artefato ausente/adulterado gera uma mensagem explícita.
`local_files_only=True` e `trust_remote_code=False` impedem substituição/download
silencioso. Não são aceitos caminhos ou repositórios arbitrários pela interface.

## API reutilizável

```python
from reviewsense.predict import InferenceService

service = InferenceService(device="auto")
prediction = service.predict("Não gostei da embalagem, mas o produto é bom.")
result = prediction.to_dict()
baseline = service.predict("O produto funciona bem.", model="svm").to_dict()
batch = service.predict_batch(["Produto ótimo.", "Não funcionou."], model="bertimbau")
```

`Prediction` é uma estrutura imutável com:

| Campo | Significado |
| --- | --- |
| `label`, `class_id` | Negativo/0, neutro/1 ou positivo/2 |
| `model` | `bertimbau` ou `svm` |
| `device`, `precision` | `cpu`/`cuda:0`; FP32/FP16 no BERT ou `sklearn_native` no SVM |
| `scores` | Scores do BERT em ordem decrescente; desempate pelo ID; vazio no SVM |
| `score_type` | `softmax_uncalibrated` no BERT; `None` no SVM |
| `inference_seconds` | Tempo da chamada de inferência, incluindo tokenização/TF-IDF, transferência e pós-processamento |
| `cold_start_seconds` | Carga sob demanda, separada; zero quando o modelo já está carregado |
| `batch_size` | Quantidade de textos da chamada; em batch o tempo pertence ao lote inteiro |
| `notes` | Mensagens públicas sobre recuperação de dispositivo, sem exceções internas |

Nenhum resultado contém o texto de entrada. Limite de **8.000 caracteres por texto**
e **16 textos por chamada de serviço**; a interface envia apenas um. O texto é
preservado: `strip` só detecta vazio. Não se removem acentos/negações, não se faz
stemming e não se acrescenta uma normalização textual diferente da já aprendida.

BERT usa tokenizer cased salvo, max_length 256, padding dinâmico ao múltiplo de 8,
microbatches de 2, `eval()` e `inference_mode()`. CUDA usa FP16 em dispositivos
compatíveis (capacidade principal ≥7); CPU usa FP32. O SVM usa o pipeline salvo
com seu TF-IDF e precisão nativa do sklearn. A CPU usa duas threads no BERT e uma
no SVM. As duas threads do BERT se referem às operações intraoperador do PyTorch,
não a afinidade exclusiva de núcleos. Não há `fit`, `fit_transform`, backward ou
otimizador no serviço.

Softmax produz **scores normalizados não calibrados**, não certeza estatística.
O SVM retorna somente a classe: suas margens não são exibidas como probabilidades.

## Arquitetura e interface

| Componente | Responsabilidade |
| --- | --- |
| `ArtifactPaths` | Resolve arquivos locais; um futuro resolvedor do Hub pode fornecer outros diretórios verificados sem alterar a interface |
| `InferenceService` | Valida entradas, carrega sob demanda, mantém um cache por instância e serializa chamadas com lock |
| `normalized_scores` | Rejeita logits inválidos e calcula Softmax estável com ordenação determinística |
| `predict` / `predict_batch` | API independente do Gradio, com resposta estruturada e latência |
| `create_app` | Constrói formulário, seleção de modelo, resultado, exemplos e links, sem carregar pesos |
| `make_analyzer` | Adapta a chamada ao serviço e apresenta erros seguros, limpando resultados anteriores |
| `launch_options` | Fixa endereço local e desativa compartilhamento, analytics, histórico, SSR, monitoramento e MCP |

A interface usa fontes do sistema, cores neutras e uma paleta discreta para os
sentimentos. Exemplos positivos, negativos, neutros e ambíguos foram escritos
para a demonstração e não vieram do dataset. O domínio de produtos de 2018,
a origem dos rótulos em estrelas e as dificuldades com neutros/ironia são visíveis.
Não há afiliação ou identidade visual do iFood.

## Privacidade, rede e erros

`share=False`, endereço padrão 127.0.0.1 e CORS estrito. A opção explícita
`--public-network` muda o bind para 0.0.0.0; nunca cria túnel de compartilhamento.
Não foi utilizada na validação local. Diretórios de dados, modelos, caches e
ambiente virtual estão bloqueados no mecanismo de arquivos do Gradio.

Não há logging de entradas, flagging, cache de exemplos ou histórico de execuções.
O Gradio é construído com analytics desligado, `run_history=False` no lançamento
e fontes locais. O histórico é desativado também no navegador. Referências:
[Blocks/launch](https://www.gradio.app/docs/gradio/blocks) e
[Examples](https://www.gradio.app/docs/gradio/examples).

`InputError`, `ArtifactError`, `DeviceError` e `InferenceError` usam mensagens
públicas. Exceções desconhecidas viram uma mensagem genérica, sem textos, caminhos
absolutos ou traceback no resultado. O serviço não escreve arquivos ou logs de
requisições. A fila é limitada a oito chamadas e o serviço controla concorrência.

## Verificação e reprodução das medições

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/pytest -q
.venv/bin/ruff check src tests app scripts
.venv/bin/ruff format --check src tests app scripts
.venv/bin/python -m pip check

# Opcional: usa os modelos originais e abre uma instância CPU temporária em 7861.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python -m scripts.verify_phase4a_local
```

O smoke grava agregados em `outputs/phase4a/local_smoke.json`, ignorado pelo Git.
Mede uma carga inicial e 12 requisições aquecidas por cenário: três passagens pelos
mesmos quatro exemplos sintéticos. Testa também entrada longa até 8.000 caracteres
e a interface CPU via HTTP. Não é avaliação do dataset ou novo teste de qualidade.

As medições são latências de requisições individuais no serviço, excluindo espera
na fila, carga, transporte HTTP e renderização; a carga inicial é separada. A
inicialização CPU acontece depois da GPU no mesmo processo, reutilizando imports.
Carga do computador e temperatura não foram isoladas. Os números dependem do texto
e não equivalem ao benchmark em lote da Fase 3C.

Os testes comuns usam checkpoints minúsculos, mocks e exemplos sintéticos, sem
downloads/GPU obrigatória. Cobrem integridade, mapas, não atualização de pesos,
recuperação CUDA limitada, textos longos, concorrência, cache, erros e construção
da interface. A configuração pytest inclui a raiz para importar o pacote `app`,
que fica fora de `src/`; os testes anteriores não foram modificados.

## Limitações e próxima etapa

A demonstração não mede generalização para restaurantes/delivery. Neutros,
sentimentos mistos, ironia, truncamento e o domínio temporal continuam limitantes.
Não há calibração ou garantia de acerto por requisição. Nenhum resultado de
inferência desta fase altera os pesos ou escolhas das fases anteriores.

Na Fase 4B serão tratados Model Card, licenciamento e preparação da publicação.
Esta entrega não publica modelo, dataset, aplicação ou Space e não implementa um
resolvedor remoto. A separação `ArtifactPaths`/serviço/UI mantém esse caminho aberto.
