# Fase 4C — Hospedagem condicionada a direitos e elegibilidade

**Nenhum Space criado, conta autenticada ou recurso contratado.** Consulta oficial
em 2026-10-30; condições devem ser reconfirmadas no lançamento.

## Opções verificadas

| Opção | Recursos / condição | Decisão para ReviewSense |
| --- | --- | --- |
| CPU Basic | 2 vCPU, 16 GB RAM, 50 GB disco não persistente, sem cobrança por hora | Primeira opção técnica, se assinatura/elegibilidade e direitos forem confirmados |
| ZeroGPU | Gradio; contas pessoais gratuitas elegíveis (e-mail verificado, mais de 30 dias) podem hospedar até 2; PRO até 10 | Alternativa sem nova contratação se a conta for elegível; integração específica ainda necessária |
| GPU dedicada T4 small | 4 vCPU, 15 GB RAM, 16 GB VRAM, 50 GB disco, USD 0,40/h | Excesso para demonstração pequena sem evidência de demanda; não contratar |
| CPU Upgrade | 8 vCPU, 32 GB RAM, 50 GB disco, USD 0,03/h | Só avaliar após medir CPU Basic e obter orçamento |
| Static Space / GitHub | Conteúdo estático, sem execução de inferência no servidor | Alternativa de código/documentação, respeitando direitos e autorização para publicar |

A [visão geral oficial](https://huggingface.co/docs/hub/spaces-overview) diferencia
custo horário e direito de criação: Gradio/Docker em CPU Basic requer plano pago
(PRO pessoal ou Team/Enterprise institucional). Portanto, uma conta gratuita não
é garantia de criação gratuita de CPU Basic. Static Spaces são gratuitos; isso
não os transforma em hospedagem gratuita do backend Python.

A [documentação de gestão](https://huggingface.co/docs/huggingface_hub/guides/manage-spaces)
explica a exigência de assinatura no CPU Basic e de pagamento/créditos para
hardware pago. Identificador, titularidade, plano, créditos, quotas e disponibilidade
real da conta **não foram consultados**. Não há namespace confirmado, e o GitHub
`jorgeribeiru` não determina o usuário do Hugging Face.

Os [preços e especificações](https://huggingface.co/docs/hub/spaces-gpus) acima são
valores publicados em USD, sujeitos a mudança. Hardware pago é cobrado pelo tempo
em Starting/Running, mesmo sem requisições, por minuto; não apenas pelo tempo da
inferência. Pausa/suspensão precisam de política explícita. Não há estimativa de
custo mensal assumindo uso contínuo, impostos ou câmbio. Grants dependem de aprovação.

Segundo a [documentação ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu),
o recurso é compartilhado e possui filas/quotas. A integração exige `spaces.GPU`
e ciclo de vida de CUDA apropriado; não equivale à GPU local persistente. A página
lista Python 3.10/3.12 e versões PyTorch específicas: o Python 3.11 e a variante
CUDA local não estão automaticamente validados para esse runtime. O código atual
preserva o modo local; **não é um adaptador ZeroGPU testado**. Não usar créditos
adicionais automaticamente nem prometer disponibilidade/latência dessa opção.

## Capacidade e operação propostas

O pacote ocupa 436.422.685 bytes (~416,21 MiB). Isso é tamanho em disco, não RAM
máxima: imports, cópias durante carga, tensores FP32, tokenizer, Python e Gradio
consomem memória adicional. Considero 16 GB uma hipótese razoável de capacidade
para um processo BERT base em CPU; não é medição no Space. Reservar espaço para
runtime/dependências e cache; não incluir checkpoint de otimizador ou corpus.

Usar uma instância/processo, duas threads intraoperador PyTorch, uma inferência
por vez e fila máxima de oito. Limite por texto 8.000 caracteres, truncamento
256 tokens e microbatch 2. O serviço aceita até 16 itens; o formulário envia um.
Não multiplicar workers carregando cópias do modelo sem medir RAM.

Após autorização, provisionar snapshot por repo_id confirmado e SHA imutável,
conferir hashes aprovados e materializar somente a allowlist como arquivos
regulares. Não depender da branch main nem dos relatórios internos. O adaptador
da presente entrega lê apenas cache offline e não provisiona downloads.
A pré-carga opcional falha antes de servir se artefatos/gates não conferirem.

Não foi medido tempo de inicialização ou inferência em Spaces. A mediana local
histórica de CPU não estima latência de rede, fila, boot ou CPU Basic. Medir
build, download de cache vazio, verificação SHA, carga, primeira inferência e
reinício após suspensão no ambiente autorizado; registrar versões e condições.

A aplicação não grava explicitamente textos, não usa analytics/cache de exemplos
e nunca cria túnel share. A plataforma pode manter logs, telemetria, controles de
abuso e retenção próprios: revisar política/termos do provedor e informar limites,
sem prometer privacidade absoluta. Revisar acesso a arquivos, segredos, HTTPS,
CORS, abuso/rate limits e dependências no ambiente real; não enviar textos pessoais
ou sensíveis à demonstração. Scores Softmax não calibrados e ausência de validação
específica para delivery devem continuar visíveis.

## Recomendação e pendências

Priorizar CPU Basic **se já houver elegibilidade/plano autorizado**, pois o modelo
é pequeno para a RAM descrita e a concorrência inicial é baixa. Sem assinatura,
avaliar ZeroGPU elegível somente após adaptação/validação própria; custo incremental
zero depende da conta, quotas e condições confirmadas. Se nenhuma opção gratuita
for elegível, manter execução local e documentação estática; não contratar para
cumprir esta fase. Direitos pendentes bloqueiam inferência pública em todas as
opções, inclusive hospedagem indireta. Namespace, elegibilidade, orçamento,
políticas e autorização humana continuam pendentes.
