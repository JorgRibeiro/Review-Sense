# Dados B2W-Reviews01

Fonte: B2W Digital, **B2W-Reviews01**, avaliações de produtos da Americanas.com
coletadas entre janeiro e maio de 2018.

- Repositório original: https://github.com/americanas-tech/b2w-reviews01
- Revisão fixada: `4639429ec698d7821fc99a0bc665fa213d9fcd5a`
- Arquivo: `B2W-Reviews01.csv`, UTF-8, delimitado por vírgulas nessa revisão.
- SHA-256: `821fb0bf9f7230b0fba4e4f9fadd75a66d1a9ff0b1657810791d33007eb2ab38`
- Espelho citado no plano: https://huggingface.co/datasets/ruanchaves/b2w-reviews01

O README original menciona ponto e vírgula, mas o arquivo fixado usa vírgula.
O carregador identifica ambos pelo cabeçalho, respeitando aspas e textos multilinha.
O download padrão usa diretamente a fonte original; não executa scripts remotos.

## Licença e atribuição

O corpus original declara **Creative Commons Attribution-NonCommercial-ShareAlike
4.0 International (CC BY-NC-SA 4.0)**:
https://creativecommons.org/licenses/by-nc-sa/4.0/.
Dê crédito à B2W Digital, registre as transformações e respeite as restrições de
uso não comercial e compartilhamento pela mesma licença nos casos aplicáveis.
O espelho informa CC BY 4.0; este projeto adota como referência a declaração
original, sem tomar a licença mais permissiva do espelho como autorização.

A licença MIT na raiz cobre o código do projeto, não o dataset. Antes de publicar
artefatos derivados ou propor uso comercial, verificar as condições aplicáveis.
Nenhum dado bruto ou processado é distribuído neste repositório. Hashes de texto
não devem ser tratados como anonimização irreversível: os manifests com IDs também
permanecem locais por padrão. Os textos podem conter dados pessoais mesmo após
excluir colunas de autor; a limpeza desta fase não realiza anonimização do conteúdo.

## Arquivos locais (ignorados pelo Git)

- `raw/B2W-Reviews01.csv`: download verificado por checksum.
- `processed/v1/train.csv`, `val.csv`, `test.csv`: somente `sample_id`, `review_text`, `label`.
- `processed/v1/manifest.json`: fonte/checksum, versões, política, configuração,
  fingerprint do conjunto preparado, IDs e fingerprints dos splits.
- `processed/v1/audit.json`: estatísticas agregadas e contagens de exclusões.

O comando `python -m reviewsense.data` cria esses artefatos. A saída existente
não é sobrescrita: para outra execução, use `--output-dir data/processed/v2`.
O armazenamento local é o padrão. Se futuramente utilizar Colab, copie os
artefatos para armazenamento persistente; `/content` é efêmero.
