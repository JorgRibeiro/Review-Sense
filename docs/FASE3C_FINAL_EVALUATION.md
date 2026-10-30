# Fase 3C — Avaliação definitiva, sem novo treinamento

O código separa preparação, execução definitiva e leitura de resultados. O
[protocolo](FASE3C_PROTOCOL.md) e `configs/phase3c_final.json` fixam os dois modelos,
métricas e condições antes de abrir o teste. A escolha de BERTimbau como candidato
principal vem da validação da Fase 3B e não muda após observar o teste.

## Operações locais

Usar a `.venv` CUDA existente na raiz do repositório; nenhuma dependência nova.
Primeiro executar os testes inteiramente sintéticos/offline:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/pytest -q
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
.venv/bin/python -m pip check
```

Antes de qualquer abertura real do teste, com código/configuração/protocolo
validados e versionados, executar o preflight:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
  -m reviewsense.final_evaluation --preflight
```

O preflight bloqueia `test.csv` por audit hook, verifica modelos/tokenizer e
hashes, usa três textos sintéticos para inferência, testa escrita atômica/esquema
em pasta temporária e mede benchmark em 1.024 avaliações de validação. Não
calcula F1 para nova seleção. Usa 32 textos de warm-up e três repetições por
modelo; fixa batch SVM 256 e BERT 2. A aprovação fica exclusivamente em
`outputs/phase3c-final-v1/preflight.json`, ignorado pelo Git.

A avaliação definitiva exige o mesmo commit, árvore limpa, versões e inventário.
Somente o argumento explícito abaixo autoriza a primeira leitura; a instrução
do usuário para executar esta fase já autoriza essa operação após todas as barreiras:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
  -m reviewsense.final_evaluation --execute-final
```

Esse comando **não é para repetir resultados já existentes**. Um registro local
exclusivo `started.json` é criado antes de abrir o teste. Ele impede nova execução
mesmo em caso de falha. Uma única leitura binária fornece SHA-256 e buffer para
parsing; a validação não modifica dados. A avaliação exige 19.004 exemplos com
esquema, IDs/ordem, fingerprint, distribuição e separação corretos. Depois há uma
passagem por modelo, agregação e escrita atômica de `results.json` local.

Após conclusão, somente esta operação é necessária:

```bash
.venv/bin/python -m reviewsense.final_evaluation --render-saved
```

Ela lê o resultado salvo, publica agregados em `reports/phase3c_final_metrics.json`
e gera `reports/validation_phase3c.md` e cinco figuras. Recusa sobrescrever JSON
público diferente; pode regenerar figuras a partir do mesmo resultado. Não abre
CSV, carrega modelos ou repete inferência. O notebook
`notebooks/04_final_model_comparison.ipynb` também só lê os agregados.

Se ocorrer falha após abrir o teste, `failure.json` guarda tipo do erro/data/contagem
de leituras, sem mensagens que exponham dados pessoais. Não retomar ou remover o
marcador automaticamente. Uma recuperação técnica exigirá documentação explícita
e não poderá alterar escolhas com base no desempenho observado.

## Funções e decisões

| Função | Propósito |
| --- | --- |
| `artifact_inventory` | Ancora pesos, tokenizer, código, manifest, configuração e referências por SHA-256, sem incluir o teste no preflight |
| `test_access_policy` | Bloqueia abertura do teste na preparação; na execução permite uma leitura e proíbe escrita/reabertura |
| `validate_manifest_metadata` | Reutiliza o carregador exclusivo de treino/validação e compara metadados do teste sem conteúdo |
| `validate_final_frame` | Confere esquema, normalização, IDs/ordem, fingerprints, classes e disjunção sem corrigir o split |
| `load_models` / `load_local_bert` | Reabre artefatos escolhidos offline, confere configurações, mapeamentos, pesos finitos e GPU |
| `predict_svm` / `predict_bert` | Recebe os mesmos textos; retorna predições vinculadas à ordem/identidade, sem ajuste |
| `timed_prediction` | Mede passagem e throughput incluindo processamento; sincroniza CUDA e amostra RSS/VRAM |
| `paired_agreement` | Conta acertos de ambos, erros de ambos e acertos exclusivos nos mesmos exemplos |
| `validate_summary` | Confere matriz/suportes/erros e recalcula P/R/F1 agregados para detectar inconsistência |
| `begin_final` / `execute_final` | Exige autorização explícita, registra execução única e recusa repetição mesmo após falha |
| `atomic_json` | Publica arquivo completo, sem sobrescrita concorrente ou resultado parcial |
| `publish_saved` / `render_report` | Reconstituem relatório/figuras sem inferência, dados individuais ou nova seleção |

As decisões estatísticas são simples: F1 Macro é a métrica principal porque dá
peso igual às três classes; métricas por classe mostram o comportamento dos neutros
que a accuracy pode ocultar. Concordância pareada explora os mesmos exemplos sem
publicá-los. Diferenças são descritivas, sem bootstrap/significância nesta fase.

O BERT usa `eval`, `inference_mode`, FP16 autocast, atenção eager, max_length 256,
padding dinâmico múltiplo de 8, batch 2 e duas threads CPU. O SVM usa seu TF-IDF
salvo e uma thread. Não há fit, backward, otimizador, calibração ou fallback
automático. Pesos e tokenizer são conferidos antes/depois por hash.

## Interpretação das medições e privacidade

O benchmark pré-teste usa a mesma amostra estratificada da validação em ambos os
modelos. Mediana de passagem, textos/s e ms/texto em batch não são latência de
requisição individual. A passagem única pelo teste tem medição separada; não é
repetida para melhorar tempos. CPU e GPU têm custos/infraestruturas diferentes.

RSS amostrado inclui ambos os modelos carregados e runtime do processo: não é
memória incremental do SVM. VRAM PyTorch refere-se ao allocator; NVML inclui
desktop/driver/outros processos e fornece pico amostrado. Temperatura, clock e
carga concorrente não são isolados. Os tempos históricos de treino têm escopos
diferentes e permanecem os registrados nas fases anteriores.

Não são versionados CSV, textos, IDs, predições individuais, pesos, caches ou
estados do otimizador. Os relatórios usam caminhos relativos ao repositório,
agregados e hashes de artefatos. A licença original dos dados continua
CC BY-NC-SA 4.0. Model Card, Hub e Gradio pertencem à próxima fase.


## Execução desta entrega

Concluída em 30/10/2026, protocolo e código congelados em `90893aa` antes de abrir
o teste. F1 Macro final: **SVM 0,723094; BERTimbau 0,754461**. Diferença absoluta
0,031367 (+4,3379% relativa). O F1 aumentou nas três classes; o neutro continua
com Recall de 38,55%. Ver contagens e análise no
[relatório final](../reports/validation_phase3c.md).

Uma leitura do CSV, uma passagem/modelo e zero falhas; a execução
`outputs/phase3c-final-v1/` está concluída. Não a repetir. O notebook foi executado
localmente sem outputs no Git; 173 testes offline, Ruff e pip check aprovados.
Código, configuração e protocolo permanecem os do congelamento, sem alterações
após observar resultados. Para verificar/visualizar a entrega, usar somente os
resultados salvos. Não há pendência de Colab nem execução de Fase 4 nesta branch.
