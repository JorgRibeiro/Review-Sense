# v1.0.0 — Código, documentação e demonstração local

Esta versão encerra o escopo publicável de código-fonte do ReviewSense PT-BR.
A preparação não criou tag ou GitHub Release. A publicação no GitHub seguirá o
checklist abaixo, após revisão humana. **Pesos e inferência hospedada não fazem
parte desta release.** A adaptação ZeroGPU permanece disponível como código preparado.

## Funcionalidades e evidências

Pipeline B2W-Reviews01, limpeza/deduplicação, splits reproduzíveis e fingerprints;
DummyClassifier, TF-IDF + LinearSVC e BERTimbau; seleção por validação, retomada de
fine-tuning local; avaliação final independente; inferência reutilizável CPU/GPU;
Gradio responsivo; validação de integridade e allowlists; testes sintéticos offline.
O treino Transformer foi executado na RTX 3050 Laptop de 6 GB.

Após limpeza: 126.692 avaliações, divididas em 88.684/19.004/19.004 para
train/val/test. Classes 0 negativo (1–2 estrelas), 1 neutro (3), 2 positivo (4–5);
são proxies por estrelas, não anotação manual. TF-IDF/modelos só foram ajustados
no treino; configurações/checkpoint escolhidos na validação antes de abrir o teste.
O BERT selecionado é época 1/update 5543. Uma leitura e passagem por modelo no
teste encerraram o experimento; nenhum refit/calibração após observar o resultado.

F1 Macro de teste: **0,7545 BERTimbau; 0,7231 SVM**, arredondados. Diferença
relativa descritiva de 4,34%, sem demonstração de superioridade estatística.
Fonte exata: [JSON congelado](../reports/phase3c_final_metrics.json) e
[relatório](../reports/validation_phase3c.md). Não executar novamente a avaliação.

## Como consultar e reproduzir conforme os recursos disponíveis

| Condição | Uso desta release |
| --- | --- |
| Somente clone | Ler código, relatórios/figuras; notebooks 03/04 inspecionam agregados sem modelos/dataset |
| Ambiente completo de testes | Executar suíte com dados e modelos sintéticos, sem GPU real/download de pesos |
| Artefatos treinados legítimos disponíveis | Executar inferência e Gradio local com integridade verificada |
| Dados legítimos, recursos e interesse em reprodução independente | Seguir guias de dados/baseline/BERT em saídas novas, preservando o experimento congelado |

Ambiente validado: Python 3.11/Linux, PyTorch 2.14.1+cu132 e dependências registradas.
Em ambiente novo, criar/ativar uma .venv e instalar primeiro uma stack PyTorch
compatível com hardware e [constraints locais](../constraints-transformer.txt).
Não substituir uma instalação CUDA funcional por build CPU para seguir este guia.
Os requirements abaixo não fornecem pesos, dados nem autorização de distribuição:

```bash
# Após disponibilizar a stack PyTorch compatível; executar na raiz do projeto.
python -m pip install -r requirements-dev.txt -r app/requirements.txt
```

Os locks históricos documentam ambientes anteriores. Não combinar
requirements-transformer-lock.txt/requirements-lock.txt (Pillow 13) com a aplicação
(Pillow 12.3). Nenhuma dependência foi instalada/trocada durante a preparação da release.
Outros SOs, builds de Torch e matriz real ZeroGPU não foram validados aqui.

Reprodução independente: [dados/histórico](HISTORICO_TECNICO.md),
[baseline](FASE2_BASELINE.md), [BERTimbau](FASE3B_BERTIMBAU.md) e notebooks em
notebooks/. Os guias conservam comandos e condições originais; não são uma sequência
a executar para verificar esta release. Não regenerar splits, retreinar modelos ou
abrir test.csv desta execução. Não usar load_artifacts sobre phase1-final: ele lê
os três splits. Treino/validação têm carregador específico em frozen_data.py.
Novas execuções independentes usam artefatos/saídas próprios e não herdam os pesos,
fingerprints ou métricas publicados. Para revisar o teste concluído, basta ler
os agregados; não rodar execute-final ou o notebook de auditoria sobre os dados reais.

## Demonstração local com artefatos legítimos

O clone não contém estes arquivos e não oferece download deles. Sem os artefatos,
a interface pode ser construída, mas não faz inferência com os modelos treinados.
Os arquivos precisam corresponder aos hashes aprovados; um novo treino não pode
substituí-los mantendo as métricas ou a identidade antiga.

```text
outputs/phase4b/model-repo-staging/   # BERT padrão: pacote local de nove arquivos
  model.safetensors
  config.json
  tokenizer.json
  tokenizer_config.json
  README.md
  BASE_MODEL_LICENSE.txt
  PROJECT_CODE_LICENSE.txt
  ATTRIBUTIONS.md
  DISTRIBUTION_STATUS.md
models/baseline/phase2-v1/           # SVM, se utilizado
  selected_model.joblib
  results.json
  config.json
```

O descritor publishing/inference_descriptor.json acompanha o código. SVM utiliza
também os registros versionados baseline_validation.json e phase3c_final_metrics.json
em reports/. Nenhum CSV é necessário. O [guia de inferência](FASE4C_RELEASE_READINESS.md)
descreve validação, diretórios externos explícitos e a compatibilidade ArtifactPaths.
Para quem dispõe somente do pacote BERT, --bert-only oculta a opção SVM.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m app.app --device auto
# Alternativa explícita: --device cpu; sem GPU não há exigência de Colab.
```

Acessar http://127.0.0.1:7860. Sem share ou exposição pública por padrão. Até 8.000
caracteres, 256 tokens no BERT, fila 8 e concorrência 1; cache/lazy load por instância.
A aplicação não grava explicitamente entradas. Evitar dados pessoais/sigilosos.
A captura do README usa texto sintético e foi reaproveitada sem alteração.

## Limitações, direitos e preparação ZeroGPU

Domínio de produtos de 2018; sem validação específica para delivery. A classe neutra
é limitante (BERT F1 0,4454, recall 0,3855). Softmax não calibrado, truncamento,
proxy por estrelas e possíveis dependências/quase duplicatas restringem a interpretação.
Sem SLA de produção ou afirmação de ausência de vieses/memorização.

Código próprio sob [MIT](../LICENSE); licenças de terceiros preservadas. Atribuir
B2W Digital/Americanas Tech e Real/Oshiro/Mafra pelo corpus, NeuralMind e
Souza/Nogueira/Lotufo pelo BERTimbau. A MIT não licencia automaticamente dados/pesos;
original do dataset CC BY-NC-SA 4.0, com divergência no espelho.
**license_status=REQUIRES_REVIEW; upload_authorized=false** continua aplicado aos
pesos. Esta release de código não resolve juridicamente sua distribuição.

Finalidade educacional/não comercial prevista, sem endosso institucional. A
[política](FASE4C_RIGHTS_OUTREACH.md) mantém atribuição/transparência, reclamações
fundamentadas e suspensão/retirada quando apropriado, sem presumir autorização.
Canal definido: Issues do projeto; não expor documentação pessoal/confidencial ali.
Não serão solicitadas autorizações por e-mail. Não há mensagens automáticas.

ZeroGPU: [entrada e arquitetura](FASE4C_ZEROGPU.md) e
[correções de contrato SDK](../reports/validation_phase4c_zerogpu_corrections.md).
Testes locais não validam o hardware/control plane reais. Direitos, revisão SHA
e destinos aprovados, autorização de hospedagem, elegibilidade e validação de runtime
são pendências **somente de uma eventual distribuição de pesos/deploy posterior**;
não integram os critérios desta release de código e documentação. Nenhum Space,
modelo público ajustado ou post definitivo de LinkedIn foi criado.

## Checklist manual — PR, integração e GitHub Release

Este procedimento está preparado, **não executado**. Não há push, PR, merge, tag ou
Release criada por esta tarefa. Publicar código no GitHub não altera os gates HF.

1. Revisar diff, [verificação final](../reports/validation_release_v1.md),
   [CHANGELOG](../CHANGELOG.md) e [notas prontas](RELEASE_NOTES_v1.0.0.md).
   Conferir que as versões fonte são 1.0.0 e não há dados/modelos/credenciais no commit.
2. Quando autorizado, enviar a branch feat/release-v1-code-docs ao remoto oficial.
   No GitHub, abrir PR para main com título “Release v1.0.0 — código e documentação”.
   Descrever explicitamente ausência de pesos/serviço hospedado e anexar o resultado
   dos checks. Conferir README/imagens na renderização do PR e canal Issues disponível.
3. Após revisão e aprovação, integrar pelo método permitido no repositório.
   Registrar o SHA definitivo da main que inclui todas as alterações. Se houver
   mudanças de código/conflitos, repetir os checks; nunca reabrir o teste de ML.
4. Com autorização para tag/release, abrir Releases → Draft a new release.
   Escolher nova tag v1.0.0 e alvo no commit revisado da main. Confirmar o SHA
   antes de criar a tag; não selecionar uma branch/commit ainda sem a preparação.
5. Usar título “ReviewSense PT-BR v1.0.0 — código e documentação” e copiar o conteúdo
   de RELEASE_NOTES_v1.0.0.md. Não anexar pesos, datasets, caches ou pastas locais.
   Os arquivos-fonte automáticos do GitHub correspondem ao conteúdo versionado.
6. Conferir notas, escopo, alvo/tag e ausência de anexos sensíveis no rascunho.
   Marcar como release estável (sem pre-release), definir como latest se apropriado
   e publicar somente após aprovação. Seguir as regras do repositório para releases.
7. Verificar página/tag, README, links/imagens e arquivos-fonte disponíveis.
   Registrar URL e SHA publicados. Não anunciar pesos ou Space inexistentes;
   LinkedIn e qualquer publicação HF continuam fora desta operação.

Referência de interface: [documentação oficial GitHub Releases](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository).
