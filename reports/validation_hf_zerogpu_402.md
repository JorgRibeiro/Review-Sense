# Tentativa ZeroGPU gratuita — criação recusada (HTTP 402)

Data informada pelo responsável: **30/10/2026**, fuso **America/Recife**.
Fonte: **relato operacional do Codex fornecido pelo responsável**, não log HTTP
bruto ou nova execução do projeto. Este documento registra somente os fatos
informados; não transforma o relato em teste independente do GitHub.

## Estado confirmado no relato

- Conta: `JorgeRibeiru`, identidade confirmada, plano não PRO.
- Destino pretendido: `JorgeRibeiru/reviewsense-ptbr` (Space público ZeroGPU).
- Tentativa de criação: rejeitada pela plataforma com **HTTP 402**,
  informando que a conta não estava elegível para ZeroGPU gratuito.
- A resposta mencionou assinatura PRO e pedido de **grant comunitário**;
  nenhuma dessas ações foi executada.
- Consulta posterior: HTTP 404 para o destino; a conta permaneceu sem Spaces.
- **17 arquivos do Space:** hashes previamente aprovados conferidos novamente,
  sem alteração dos bytes; nenhum arquivo enviado.
- **Space:** não criado; nenhum build, GPU, versão de runtime hospedado ou
  inferência real no Space foi executado ou validado.
- **Custos:** nenhuma contratação, assinatura, recurso pago ou crédito adquirido.

## Modelo já publicado e validado

- Repositório: <https://huggingface.co/JorgeRibeiru/reviewsense-ptbr-bertimbau>
- Revisão fixa verificada: `ae89cf168ffaa64115c097204e678ed513a54265`.
- O relatório anterior documenta download público com `token=False`, verificação
  integral dos **10 arquivos** por SHA-256 e cinco inferências sintéticas em CPU:
  [verificação do cache](validation_hf_shared_blobs.md).
- Peso `model.safetensors`: **435.725.300 bytes**;
  SHA-256 `a65f1ee302987c5f0b01cd7981722e92f17ff332358b8e19b8dc01ee05753c70`.

## Consequências e continuidade

O bloqueio ocorreu **antes da criação do Space**. Não demonstra falha de build,
CUDA/ZeroGPU, segurança ou qualidade do modelo: nenhuma dessas etapas foi testada
na infraestrutura hospedada.

A implementação e os 17 arquivos podem permanecer preparados. Uma nova tentativa
com ZeroGPU gratuito depende de elegibilidade da conta ou concessão de grant.
Não contratar recursos pagos nem usar outro backend sem nova decisão do responsável.
Somente anunciar demonstração online após implantação e inferência pública reais.

O escopo histórico da release GitHub `v1.0.0` permanece código/documentação e
resultados; o upload dos pesos no Hub é posterior. A situação de direitos de
terceiros permanece `REQUIRES_REVIEW / NOT_ESTABLISHED`. Nenhum arquivo de corpus,
credencial, Secret, registro privado ou resultado novo de avaliação foi incluído.
