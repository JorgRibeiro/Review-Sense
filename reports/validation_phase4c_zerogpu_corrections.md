# Fase 4C — Correções do contrato spaces 0.51.3

**Preparação local; publicação bloqueada. REQUIRES_REVIEW; upload_authorized=false.**
Branch feat/phase4c-zerogpu-readiness; base 2a4121d. Este relatório complementa os
anteriores sem reescrevê-los. O commit de correção é identificado pelo histórico:
`git log -1 --format=%H -- reports/validation_phase4c_zerogpu_corrections.md`.
Hashes dos módulos testados estão no JSON desta verificação.

## Variável de ambiente

Inspeção do wheel público spaces 0.51.3, SHA-256
`e588ce6b4301c566139d84bafb1c30b555967ce4ee2792be659ee5643f0f7eb1`, conferido contra metadata PyPI.
spaces/config.py implementa boolean como valor não nulo cujo lower é 1, t ou true;
não há strip, yes ou on. A checagem preliminar acompanha exatamente esse conjunto,
inclusive caixa variada. Ausência/valor inválido rejeita antes de importar o runtime.
A checagem adicional de Config.zero_gpu permanece: uma variável verdadeira não
habilita execução se o Config carregado estiver inativo. Não se consultou CUDA
real para decidir se o processo está em ZeroGPU.

## Callback, descoberta local e scheduler

O código real de GPU/_GPU registra dois objetos por identidade em decorated_cache:
o callback original associado ao handler GPU e o próprio handler associado a si.
regular_function_wrapper captura id(callback) como task_id e pede schedule antes
de criar/despachar o worker. O cache também torna GPU(função já conhecida) idempotente.
O wrapper externo antigo herdava zerogpu via wraps, mas não tinha entrada nesse
cache; os novos testes reproduzem essa discrepância com o código original.

A correção mantém o guard externo de validação/sanitização e o registra explicitamente
como alias de si mesmo no cache, depois de conferir as duas associações originais.
Também exige que GPU(guard) devolva exatamente o mesmo objeto. Os dois eventos
Gradio recebem esse guard, sem um novo wrapper alocador automático que o contorne.
Em entrada válida ele sempre chama o handler GPU original; não há caminho que
execute diretamente a inferência sem o alocador. O task_id permanece o original
usado pelo SDK. Entradas inválidas não pedem schedule; erros de alocação continuam
sendo sanitizados fora do handler. Fila 8 e concorrência compartilhada 1 preservadas.

A integração usa deliberadamente um **contrato privado do SDK 0.51.3**, fixado na
matriz existente. Registro incompatível falha na inicialização antes da UI.
Não se apresenta isso como API pública estável ou suporte a outras versões.

## O que foi realmente testado

Trechos exatos de boolean, GPU/_GPU e regular_function_wrapper (com seus helpers)
estão em tests/fixtures/spaces_0_51_3, com SHA do wheel, arquivos completos originais
e trechos, NOTICE e licença Apache-2.0. O wheel declara a licença em metadata e não
traz LICENSE separado; o texto incluído veio da Apache. Esses arquivos são apenas
fixtures e não entram nos 17 arquivos do Space. Nenhuma dependência foi instalada.

Os testes compilam funções com corpos originais e anotações postergadas, fornecendo
imports, Config, transporte e workers substituídos. Não importam o inicializador
do pacote nem executam patches CUDA, fork/hidratação ou serviços do provedor.
O transporte simulado permite conferir schedule -> worker -> release e task_id;
negação de schedule comprova que o callback não é chamado e o erro não vaza entrada
ou paths. Eventos reais do Gradio em memória compartilham o objeto registrado;
GPU(event.fn) devolve o mesmo objeto. Isso vai além de testar atributos copiados.

**Não foi comprovado o registro/descoberta no control plane externo nem uma alocação
real ZeroGPU.** A validação prova o contrato Python inspecionado com fronteiras
substituídas. Inicialização completa do SDK, serviço remoto, processos, quotas,
CUDA emulado e hardware compartilhado continuam exigindo runtime autorizado.

## Qualidade e bundle

- **393 testes aprovados em 13.97 s**, incluindo os 370 anteriores e 23 novos;
  zero falhas/skips. Ruff, formatação e pip check aprovados.
- Suíte offline sob guarda no test.csv real: zero leituras e tentativas bloqueadas.
- Novo candidato preservando os anteriores:
  outputs/phase4c-zerogpu/space-candidate-v3, 17 arquivos / 77551 bytes.
  Fingerprint do inventário: `a6d002e7be372b4ed8db1f4bb61fa6468ebf0cf160068d06bc57d3f8fbc01304`.
- Bundle validado por allowlist/checksums, sem pesos, datasets, relatórios ou fixtures
  SDK. Descritor continua idêntico e bloqueado. Execução independente -I -B, sem
  rede/acesso às origens: recusa por direitos antes de carregar spaces/torch/modelos.
- 70 arquivos históricos mantiveram SHA-256, inclusive pacote 4B, descritor,
  checkpoint/tokenizer, métricas/relatórios e resultados finais anteriores.
  O bundle ZeroGPU anterior v2 também passou com seu inventário original.
  Três splits mantiveram inode/tamanho/mtime. Nenhum CSV foi aberto.
- Sem alteração de dependências, modelo, métricas ou classificação. Sem Space,
  credenciais, upload, deploy, push, merge da branch ou mensagens/e-mails.

Arquivos de produto alterados: src/reviewsense/zerogpu.py e docs/FASE4C_ZEROGPU.md.
Testes: test_zerogpu.py, novo test_spaces_0513_contract.py, helper upstream_spaces.py
e fixtures de fonte/licença/proveniência. Evidências são novas, sem modificar as antigas.

Reprodução:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/pytest -q
.venv/bin/ruff check src tests app scripts deploy/zerogpu
.venv/bin/ruff format --check src tests app scripts deploy/zerogpu
.venv/bin/python -m pip check
```

Fonte primária: [spaces 0.51.3 no PyPI](https://pypi.org/project/spaces/0.51.3/);
proveniência completa dos trechos em
[provenance.json](../tests/fixtures/spaces_0_51_3/provenance.json).
Evidência estruturada: [phase4c_zerogpu_contract_corrections.json](phase4c_zerogpu_contract_corrections.json).
A Fase 4C não está encerrada; direitos, destinos, autorização e validação real permanecem pendentes.
