---
title: ReviewSense PT-BR
emoji: 💬
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.31.0
python_version: 3.12.12
app_file: app.py
short_description: Classificação educacional de avaliações em português
startup_duration_timeout: 30m
---

# ReviewSense PT-BR — candidato ZeroGPU, não publicado

**license_status=REQUIRES_REVIEW; upload_authorized=false.**
Este esqueleto não autoriza criar Space, publicar pesos ou hospedar inferência.
Conta informada: `JorgeRibeiru`. Destinos candidatos, sem repositórios consultados
ou criados: modelo `JorgeRibeiru/reviewsense-ptbr-bertimbau` e Space
`JorgeRibeiru/reviewsense-ptbr`. Não são defaults executáveis de provisionamento.

O diretório sozinho é um template. O preparador local do projeto copia por
allowlist o código de inferência e a UI existentes para um bundle independente.
Não copiar o repositório inteiro: datasets, relatórios, treinamento, estados e
pesos não fazem parte do Space de código. O descritor acompanha o bundle fora
do pacote de nove arquivos do modelo, que permanece intacto.

A entrada exige variáveis explícitas `REVIEWSENSE_MODEL_REPO_ID`,
`REVIEWSENSE_MODEL_REVISION` (SHA de 40 caracteres), `REVIEWSENSE_CACHE_ROOT`,
`REVIEWSENSE_APPROVAL_ROOT` e o `SPACE_ID` do runtime. Não há valores de exemplo
que concedam aprovação. Recibos privados não são enviados no bundle. O recibo
humano futuro deve vincular space_id e ação host_zerogpu ao candidato, além dos
gates de direitos/destino/autorização já existentes.

O padrão só materializa cache local autorizado. Provisionamento público futuro
exige opção separada REVIEWSENSE_PROVISION_PUBLIC_SNAPSHOT=1 e todos os gates;
usa revisão fixa, allowlist e token=False. Com o descritor atual, nem leitura de
snapshot nem download são permitidos por este caminho. Não há login ou uploader.
Não configurar preload_from_hub no metadata enquanto a publicação está bloqueada.

Runtime alvo e versões observadas estão em runtime.json. Gradio é selecionado
via sdk_version; torch, spaces e huggingface-hub são geridos pelo runtime, não
reinstalados pelo requirements. O startup recusa versões diferentes da combinação
preparada: validar/revisar a matriz em ambiente autorizado antes de uso público.
Esta matriz foi verificada por documentação/metadata, sem build real ZeroGPU.

Após autorização e provisão legítima, a inicialização importa spaces antes de
qualquer uso de torch, verifica integridade, carrega uma vez e coloca o modelo em
CUDA emulado fora do callback. O runtime atribui GPU física durante a função
GPU decorada. Não confundir esse mecanismo com CUDA local. Somente BERTimbau,
256 tokens, 8.000 caracteres, fila 8, concorrência 1, duração solicitada 10 s.
O orçamento de 10 s é provisório, não uma medição/SLA de inferência em Spaces.

A UI mantém exemplos sintéticos, scores não calibrados, limites do corpus e aviso
sobre logs da plataforma. Sem parceria institucional, cache de exemplos ou logging
explícito de textos. Nenhum modelo base substitui o fine-tuned em caso de falha.

Código: [ReviewSense no GitHub](https://github.com/jorgeribeiru/Review-Sense).
Direitos pendentes: a política de atribuição/reclamações não concede autorização;
a decisão de não solicitar permissões por e-mail permanece. Nenhum deploy foi feito.
