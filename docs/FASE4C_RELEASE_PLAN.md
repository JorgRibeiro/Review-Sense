# Fase 4C — Plano de lançamento, autorização e encerramento

Esta é a última fase planejada desta versão. Este arquivo é um plano, não uma
autorização. **Estado atual: REQUIRES_REVIEW; upload não autorizado; destino HF
não confirmado.** Não há script de publicação executável nesta entrega.

## Gate 1 — Direitos, finalidade e autorização

- Resolver materialmente as questões da auditoria 4B e documentar licença/avisos
  aprovados para os pesos e para a demonstração hospedada; finalidade/jurisdição
  e eventual exploração comercial precisam ser consideradas.
- Guardar os esclarecimentos/autorizações aplicáveis ou orientação especializada.
  Uma mudança de metadata para MIT/CC/READY, isoladamente, não satisfaz esse gate.
- Se continuar REQUIRES_REVIEW ou houver impedimento BLOCKED, não publicar pesos
  nem tratar a versão como lançada. Definir o próximo passo com o usuário.
- Confirmar o identificador HF e a titularidade do namespace com o usuário;
  não inferir conta HF a partir do nome do GitHub.
- Obter autorização humana explícita para operações remotas, escopo e destino.
  Tokens devem ficar fora de Git/relatórios e usar permissões mínimas adequadas.

## Gate 2 — Candidato exato

Revisar o Model Card e os avisos finais, aplicar a licença efetivamente aprovada
e produzir novo candidato/manifest com hashes quando houver alteração de bytes.
Reexecutar testes, equivalência CPU e GPU apropriada, carregamento independente,
inspeção da allowlist e ausência de corpus/IDs/credenciais. Não reabrir o teste,
retreinar, calibrar ou ajustar configurações usando suas métricas observadas.

O uploader futuro deve consumir a **lista exata** aprovada no manifest, a partir
do staging. Recusar extras, hashes diferentes, symlinks, destino vazio/incorreto,
autorização ausente e licença pendente. Nunca enviar models/, outputs/ ou o repo
de desenvolvimento inteiro. O manifest técnico interno não é enviado automaticamente.
SVM não faz parte deste repositório de modelo BERT; sua distribuição exigiria
decisão e revisão próprias.

Referência de API para implementação futura, após os gates:
[upload oficial do huggingface_hub](https://huggingface.co/docs/huggingface_hub/en/guides/upload).
Nenhum create_repo/upload_folder/push_to_hub/hf upload/login foi executado na 4B.

## Gate 3 — Publicação autorizada e verificação posterior

1. Criar/utilizar somente o destino explicitamente confirmado, com visibilidade aprovada.
2. Validar metadata no Hub e enviar exclusivamente os arquivos da allowlist aprovada.
3. Registrar repo_id e revisão de conteúdo exata, sem depender apenas de main.
4. Baixar em cache limpo pela revisão fixada; conferir hashes e carregar sem dados
   ou relatórios do computador de desenvolvimento.
5. Reexecutar o exemplo de inferência e equivalência sintética; conferir os labels,
   truncamento 256, scores finitos, avisos e links do Model Card.
6. Não interpretar sucesso de download ou validação YAML como aprovação jurídica.

## Gate 4 — Integração, Gradio e release

Preparar um descritor público mínimo de inferência (hashes dos quatro artefatos,
labels, max_length e versão/licença aprovada), separado do manifest interno com
histórico de desenvolvimento. Se isso acrescentar arquivo ao candidato, atualizar
allowlist/manifest e repetir sua verificação antes de upload.

Implementar um resolvedor de snapshot por repo_id/revisão que devolva diretório
verificado e descritor. Manter a API Prediction/InferenceService e a interface;
substituir o adaptador de referências locais por esse descritor, preservando o
carregador local como opção. O caminho remoto não deve precisar dos relatórios da
Fase 3C. Não habilitar essa migração na Fase 4B.

Escolher infraestrutura compatível e autorizada, sem presumir hospedagem gratuita
ou capacidade GPU. Publicar a aplicação somente com autorização; usar exemplos
sintéticos e política de privacidade compatível com hospedagem pública. Validar
CPU fallback, ausência de logging de entradas, mensagens e limites de requisição.

Conferir fluxo completo navegador → serviço → modelo → resultado, dispositivos
móveis, links GitHub/Model Card/demo, revisão fixada, licença e ausência de dados
sensíveis. Atualizar README/documentação e criar release estável com artefatos e
escopo verificáveis, sem incorporar pesos ao Git do projeto de desenvolvimento.
Push, merge, tag/release e publicação de aplicação exigem instruções correspondentes.

## Gate 5 — LinkedIn, somente no encerramento

O post será **escrito e revisado apenas na conclusão da 4C**, depois que os
artefatos públicos e links estiverem funcionando. Não há rascunho nem publicação
automática nesta fase. O usuário aprova a versão final e a publicação separadamente.

O texto deverá cobrir problema e motivação, corpus e classes por estrelas,
comparação SVM/BERT, **F1 Macro independente 0,7545** do BERT e diferença
descritiva sobre o SVM, treino local na RTX 3050 de 6 GB, serviço de inferência,
testes, Gradio, custo computacional e aprendizados de engenharia/ML/MLOps.
Incluir limitações e links testados para GitHub, Model Card e demonstração pública.

Evitar superlativos, promessa de precisão, superioridade estatística não demonstrada
e sugestão de anotação manual. Não insinuar parceria com iFood/Americanas. Escolher
um visual autorizado e verificado, como captura com texto sintético ou gráfico de
métricas reais, sem avaliações/pessoas identificáveis.

O encerramento exige uma apresentação profissional coerente com o que estiver
efetivamente público e autorizado; não anunciar publicação que não aconteceu.
