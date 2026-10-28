# ReviewSense PT-BR

Pipeline reprodutível de preparação e auditoria de avaliações brasileiras.
Esta entrega implementa exclusivamente a **Fase 1 — Estrutura inicial e pipeline
de dados**, seguindo [o plano técnico](ReviewSense_PTBR_PLANO.md).
**Nenhum SVM ou BERTimbau foi treinado ou avaliado.**

As classes são derivadas das estrelas: **1–2 → negativo (0), 3 → neutro (1),
4–5 → positivo (2)**. Isso é uma proxy de sentimento, não uma anotação direta.
Os resultados futuros estarão inicialmente limitados a avaliações de produtos
no domínio do corpus; não comprovam generalização para delivery ou outros setores.

## Ambiente principal

O desenvolvimento e as execuções usam o computador do desenvolvedor. A Fase 1
roda em CPU; fases futuras poderão usar a GPU local após verificar hardware,
drivers e suporte das bibliotecas. CPU e GPU podem ser usadas no mesmo ambiente
virtual. Não há uso de TPU previsto. Colab ou outra infraestrutura externa ficam
como alternativas se memória ou tempo de execução se tornarem limitantes.
A validação no Colab é opcional e não bloqueia esta fase.

## Instalação e execução local

Requer Python 3.11 ou superior. Execute os comandos na raiz do repositório:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m reviewsense.data
pytest -q
ruff check src tests
ruff format --check src tests
```

A execução baixa a revisão oficial fixada, valida seu checksum, audita, limpa e
salva os splits e relatórios em `data/processed/v1/`, ignorado pelo Git. O cache
fica em `data/raw/`. Para usar um CSV local e outra saída:

```bash
python -m reviewsense.data --csv data/raw/B2W-Reviews01.csv --output-dir data/processed/v2
```

O arquivo local pode ser outro recorte: o checksum é registrado, mas a validação
contra o checksum oficial só é obrigatória no download/cache padrão. Falhas de
esquema, notas ausentes/inválidas e corrupção do cache interrompem o pipeline.
Erros não são descartados silenciosamente. Uma saída existente também causa erro,
para preservar os splits congelados. O CLI não exige GPU nem conta no Hugging Face.

As dependências ficam no `pyproject.toml`; `requirements.txt` instala o pacote e
`requirements-dev.txt` inclui testes e ferramentas de notebook. Para repetir o
ambiente **local efetivamente validado em Python 3.11/Linux**, use:

```bash
python -m pip install -r requirements-lock.txt
```

O lock local não representa validação no Colab. A instalação por intervalos de
versões é o caminho inicial para testar outros ambientes.

## Notebook local e alternativa no Google Colab

Abra [`notebooks/01_data_audit_eda.ipynb`](notebooks/01_data_audit_eda.ipynb).
Execute-o localmente com o kernel da `.venv`, após instalar as dependências de
desenvolvimento. A auditoria utiliza os módulos do pacote e não requer GPU.

Se os recursos locais se tornarem limitantes e você optar pelo Colab, o notebook
também inclui instruções de preparação. Para clonar, os commits devem estar no
remoto; alternativamente, transfira o repositório completo para a sessão.

Depois de clonar/transferir a pasta no Colab:

```python
%cd /content/Review-Sense
%pip install -r requirements-dev.txt
```

Execute as células em ordem. Se o notebook estiver em outra pasta, defina a variável
`REVIEWSENSE_REPO` com o caminho da raiz. Uma célula de exportação fica desativada
por padrão; ela usa a mesma política de não sobrescrever diretórios existentes.
Salve os artefatos locais fora de `/content` antes de encerrar a sessão. Evite
versionar notebooks com saídas que exponham avaliações ou dados pessoais.

## Protocolo de dados

1. Validar `review_text` e `overall_rating`. Notas devem ser números inteiros de 1 a 5;
   textos podem estar ausentes para que o descarte seja auditado. Valores textuais
   como `NA` e `null` são preservados. Metadados extras são auditados, não usados como entrada.
2. Auditar nulos por coluna, distribuição de notas/classes, textos vazios,
   duplicatas de linhas e textos, conflitos e comprimentos em caracteres/palavras.
3. Normalizar Unicode NFC, controles ASCII e espaços. Preservar caixa, acentos,
   pontuação, emojis e negações. Não usar stemming nem remover stopwords.
4. Remover textos vazios e **todas** as linhas de grupos de textos normalizados
   idênticos com classes conflitantes. Notas diferentes dentro da mesma classe
   são contadas na auditoria, mas não constituem conflito de classe.
5. Manter um exemplar de cada texto normalizado restante, mesmo que tenha autores
   ou produtos diferentes. A auditoria e o relatório de limpeza permitem observar
   a mudança na distribuição. Não se tenta corrigir discrepâncias semânticas texto-nota.
6. Gerar `sample_id` por SHA-256 do texto normalizado e ordenar por ele. Dividir
   por classe, aproximadamente **70/15/15**, com semente **42**, usando
   [`configs/data.json`](configs/data.json). As proporções podem variar pelo arredondamento.
   A ordem do CSV não altera os splits. Falhar se não for possível estratificar
   com todas as classes presentes em cada conjunto.
7. Verificar ausência de interseção de IDs/textos e cobertura exata do conjunto
   preparado, incluindo os rótulos. Salvar IDs, fingerprints e versões no manifest.
   Para reabrir os dados em fases futuras, usar `load_artifacts`, que verifica
   novamente os arquivos contra o manifest e preserva textos como `NA` e `null`.

```python
from reviewsense.data import load_artifacts

splits, manifest, audit = load_artifacts("data/processed/v1")
```

A garantia de disjunção cobre **textos normalizados idênticos**, não paráfrases,
textos quase iguais, produtos ou autores compartilhados. O split aleatório também
não mede generalização temporal. Essas análises ficam para etapas futuras.
O CSV preparado tem apenas `sample_id`, `review_text` e `label`; metadados e estrelas
não ficam disponíveis como features. O rótulo é exclusivamente o alvo.

## Organização

```text
configs/data.json                   # Configuração executável dos splits
configs/train.yaml                  # Reserva para fases futuras
src/reviewsense/labels.py            # Contrato de classes
src/reviewsense/data.py              # Download, validação, auditoria, limpeza, splits e CLI
src/reviewsense/{baseline,train_transformer,evaluate,predict}.py
                                    # Reservas documentadas, sem implementação de modelos
notebooks/01_data_audit_eda.ipynb     # Auditoria e gráficos usando os módulos
tests/                              # Testes offline com exemplos sintéticos
app/                                # Reserva para Gradio, sem implementação nesta fase
data/README.md                      # Proveniência, licença e artefatos locais
reports/validation_phase1.md         # Evidências da validação executada
reports/data_audit_summary.json      # Resumo agregado, sem textos ou IDs de avaliações
reports/figures/                     # Destino de figuras agregadas
```

Os notebooks 02/03 e o workflow de CI serão criados nas fases correspondentes.
A fase atual não instala PyTorch, Transformers ou Gradio.

## Fonte e licença

B2W Digital, [B2W-Reviews01](https://github.com/americanas-tech/b2w-reviews01),
Americanas.com, janeiro–maio de 2018. O original declara **CC BY-NC-SA 4.0**;
o espelho no Hugging Face informa uma licença divergente. Usamos a declaração
original como referência. Consulte [data/README.md](data/README.md) para atribuição,
revisão e checksum. A [licença MIT](LICENSE) cobre somente o código do projeto.
