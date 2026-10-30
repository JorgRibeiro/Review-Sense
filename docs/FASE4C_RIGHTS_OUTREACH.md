# Fase 4C — Esclarecimento de direitos e solicitação para revisão humana

**REQUIRES_REVIEW; upload_authorized=false. Comunicação preparada, não enviada.**
Consulta pública em 2026-10-30. Nenhum prazo ou silêncio constitui autorização.

## Canais e autoridade

| Canal público | Evidência e finalidade | Limite |
| --- | --- | --- |
| Americanas RI: `ri@americanas.io` e formulário | [Fale com o RI](https://ri.americanas.io/informacoes-aos-investidores/fale-com-o-ri/) identifica o canal institucional | Solicitar encaminhamento ao jurídico/propriedade intelectual e responsável pelo corpus; RI não foi identificado como licenciador |
| Privacidade: `privacidade@americanas.com` | [Aviso oficial, seção 10](https://ri.americanas.io/aviso-de-privacidade-2/) publica contato para direitos dos titulares | Perguntar sobre conteúdo de terceiros e encaminhamento; DPO não é automaticamente titular dos direitos autorais |
| Repositório Americanas Tech | [B2W-Reviews01](https://github.com/americanas-tech/b2w-reviews01) está arquivado desde 31/05/2022 | Fonte histórica e licença; não depender da abertura de issue para obter resposta |
| Autores acadêmicos: Livy Real, Marcio Oshiro, Alexandre Mafra | [Artigo disponibilizado pelo repositório original](https://github.com/americanas-tech/b2w-reviews01/blob/main/b2wreviews01_stil2019.pdf) | Podem esclarecer publicação/procedência e indicar responsáveis; autoria científica não prova titularidade ou poder para autorizar todos os usos |
| Mantenedor do espelho `ruanchaves` | [Dataset card](https://huggingface.co/datasets/ruanchaves/b2w-reviews01/blob/main/README.md) declara CC BY 4.0 | Esclarecer procedência dessa declaração; não presumir poderes de relicenciamento sobre o original |

Começar com um encaminhamento institucional, sem enviar simultaneamente a vários
canais. Confirmar destinatário competente antes de anexar qualquer documento
confidencial. Não foram inventados endereços pessoais dos autores. Algumas rotas
antigas do site de RI retornaram 404; usar as páginas acima e reconfirmar no envio.
Não usar canal de denúncias para uma solicitação de licença. As páginas de RI foram
lidas pelo navegador de pesquisa em fonte oficial; consultas HTTP diretas do
verificador receberam 403. Isso limita a checagem automatizada, sem provar que o
canal foi desativado. Reconfirmar conteúdo/acesso manualmente antes de enviar.

A fonte original declara CC BY-NC-SA 4.0, distinta da CC BY 4.0 do espelho. O
[resumo da CC](https://creativecommons.org/licenses/by-nc-sa/4.0/) descreve atribuição,
NC e SA, sem resolver a situação específica dos pesos ou todos os direitos de
terceiros. Ver [auditoria anterior](FASE4B_LICENSE_REVIEW.md). Autores, organização
publicadora, atuais titulares e autores das avaliações são posições diferentes.
Uma autorização só alcança direitos que o remetente efetivamente detenha/represente.

## Pedido em português — pronto para revisão humana

**Assunto: B2W-Reviews01 — encaminhamento e esclarecimento de direitos para modelo educacional**

Prezada equipe,

Meu nome é Jorge, responsável pelo projeto independente ReviewSense PT-BR
(https://github.com/jorgeribeiru/Review-Sense). Solicito encaminhamento ao responsável
jurídico/propriedade intelectual e à equipe competente pelo corpus B2W-Reviews01,
originalmente publicado pela B2W Digital e associado ao artigo de Livy Real,
Marcio Oshiro e Alexandre Mafra.

O projeto educacional classifica avaliações em português em negativo, neutro e
positivo. Foi realizado fine-tuning do BERTimbau base, fornecido pela NeuralMind,
com textos do corpus; o código e o modelo base têm avisos MIT próprios. A fonte
original informa CC BY-NC-SA 4.0, enquanto um espelho no Hugging Face declara
CC BY 4.0. Não interpretei essa divergência como permissão adicional.

Pretendo, se autorizado, disponibilizar gratuitamente os pesos ajustados
(safetensors, aproximadamente 416 MiB com tokenizer/configuração e documentação)
no Hugging Face, permitir seu download e manter uma demonstração Gradio pública
que receba textos dos visitantes para inferência. Não pretendo redistribuir o
corpus, splits, IDs, perfis de consumidores, exemplos reais, predições individuais
ou estados de treinamento. A demonstração usaria exemplos sintéticos. Não houve
auditoria exaustiva de memorização dos pesos.

O objetivo é aprendizado e portfólio profissional, com divulgação do projeto e de
seus resultados; mesmo sem cobrança ou anúncios, reconheço que essa finalidade
pode exigir análise de NonCommercial. Não há parceria ou endosso da Americanas,
NeuralMind ou iFood. Nenhum peso ou serviço de inferência foi publicado.

Poderiam esclarecer por escrito:

1. Quem detém/representa os direitos necessários e qual é a origem da declaração
   de licença do corpus? Há autorização que explique a licença divergente do espelho?
2. A distribuição pública dos pesos ajustados e a inferência gratuita hospedada
   estão permitidas neste escopo? Caso dependam de permissão adicional, seria
   possível concedê-la expressamente e indicar seu alcance territorial e temporal?
3. Há restrições para portfólio profissional, recrutamento, anúncios, patrocínio,
   monetização, serviços comerciais ou reutilização dos pesos por terceiros?
   Que licença e condições deveriam acompanhar os pesos e a demonstração?
4. Quais atribuições, referências, avisos de alterações e obrigações SA/NC devem
   ser preservados? Há texto de aviso ou restrição de marcas a adotar?
5. A autorização alcança avaliações escritas por consumidores e eventuais direitos
   de bases de dados? Existem direitos de terceiros, consentimentos, termos ou
   medidas de privacidade/memorização que precisem de avaliação/autorização separada?
6. Quem pode confirmar formalmente esses pontos e quais documentos podem ser
   mantidos como evidência? Se não forem a equipe competente, poderiam indicar
   um canal institucional para encaminhamento?

Até resolver as pendências, manterei os pesos e a inferência restritos ao ambiente
local, sem redistribuir dados. Agradeço a orientação; posso fornecer o Model Card,
lista exata de arquivos e descrição técnica, sem encaminhar avaliações individuais.

Atenciosamente,
Jorge — responsável pelo ReviewSense PT-BR
Contato de resposta: utilizar o endereço do remetente, preenchido na revisão humana.

## Registro e decisão posterior

Revisar destinatário, assinatura, escopo e contexto antes do envio manual.
Guardar mensagem enviada, cabeçalhos, resposta integral, anexos, datas e protocolos
em diretório privado fora do Git (por exemplo `outputs/phase4c/rights-private/`).
Conferir identidade, vínculo e autoridade do respondente por canal institucional.
Registrar finalidade, territórios, duração, limites comerciais, direitos de
terceiros, condições de divulgação e atribuições. Resposta genérica ou de autor
sem representação demonstrada não resolve automaticamente a pendência.

Submeter dúvidas remanescentes a avaliação especializada pertinente. Produzir um
resumo versionável sem contatos pessoais/documentos privados e um recibo interno
com hash da evidência, vinculado ao candidato. Direitos, destino e autorização
humana são registros separados. Não mudar o status automaticamente ao receber
mensagem, nem interpretar ausência de resposta como consentimento. Se houver
recusa/limite incompatível, bloquear o lançamento pretendido.
