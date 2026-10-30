"""Interface local. Toda operação de modelos pertence a reviewsense.predict."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from html import escape
from pathlib import Path

# Definido antes de importar Gradio; sem analytics, histórico de execuções ou cache de exemplos.
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr

from app.config import AppSettings, check_hosting_authorization, service_from_settings
from reviewsense.predict import (
    ArtifactError,
    DeviceError,
    InferenceError,
    InferenceService,
    InputError,
)

ROOT = Path(__file__).resolve().parents[1]
CSS = (Path(__file__).parent / "style.css").read_text(encoding="utf-8")
EXAMPLES = [
    "O produto chegou antes do prazo e funciona muito bem. Gostei da compra!",
    "Recebi uma peça quebrada e não consegui resolver a troca. Foi frustrante.",
    "O produto tem o tamanho anunciado. Ainda estou testando no dia a dia.",
    "A qualidade é boa, mas a entrega demorou e a embalagem veio amassada.",
]

EMPTY_RESULT = """<section class="result-card empty" aria-live="polite">
  <span class="section-kicker">02 / RESULTADO</span>
  <div class="empty-symbol" aria-hidden="true">↗</div>
  <h2>Uma avaliação.<br>Três possibilidades.</h2>
  <p>Escreva um texto e analise o sentimento para ver o resultado aqui.</p>
  <div class="sentiment-key"><span>Negativo</span><span>Neutro</span><span>Positivo</span></div>
</section>"""


def result_html(result) -> str:
    """Apresenta somente campos verificados; nunca reproduz o texto de entrada."""
    sentiment = escape(result.label)
    device = "GPU · CUDA" if result.device.startswith("cuda") else "CPU"
    model = "BERTimbau" if result.model == "bertimbau" else "TF-IDF + LinearSVC"
    scores = ""
    if result.score_type == "softmax_uncalibrated":
        rows = "".join(
            f'<div class="score-row"><div><span>{escape(score.label.capitalize())}</span>'
            f'<strong>{score.value:.1%}</strong></div><div class="score-track">'
            f'<span style="width:{score.value * 100:.3f}%"></span></div></div>'
            for score in result.scores
        )
        scores = (
            '<div class="scores"><h3>Scores normalizados</h3>'
            + rows
            + (
                '<p class="score-note">Scores não calibrados. '
                "Não representam certeza estatística.</p></div>"
            )
        )
    else:
        scores = '<p class="svm-note">O SVM retorna uma classe, sem probabilidades calibradas.</p>'
    notes = "".join(f'<p class="device-note">{escape(note)}</p>' for note in result.notes)
    return f"""<section class="result-card {sentiment}" aria-live="polite">
      <span class="section-kicker">02 / RESULTADO</span>
      <div class="sentiment-pill">{sentiment.capitalize()} <span>· {result.class_id}</span></div>
      <h2>Sentimento<br>{sentiment}.</h2>
      <p class="result-caption">Classificação estimada para esta avaliação.</p>
      {scores}
      <div class="inference-meta"><span>{model}<small>{device}</small></span>
      <span>{result.inference_seconds * 1000:.1f} ms<small>Inferência</small></span></div>
      {notes}
    </section>"""


def public_error_html(message: str) -> str:
    return (
        '<section class="result-card error" role="alert">'
        '<span class="section-kicker">ANÁLISE NÃO CONCLUÍDA</span>'
        "<h2>Vamos tentar de novo.</h2>"
        f"<p>{escape(message)}</p></section>"
    )


def make_analyzer(service: InferenceService, *, enable_svm: bool = True):
    """Callback fino; erros recebem mensagens públicas, sem exceções/logs de entradas."""

    def analyze(text, model):
        try:
            return result_html(service.predict(text, model=model))
        except InputError as error:
            # Só mensagens do validador são públicas, mesmo para exceções tipadas.
            allowed = {
                "Insira uma avaliação em formato de texto.",
                "Escreva uma avaliação antes de analisar.",
                "A avaliação deve ter no máximo 8.000 caracteres.",
                "Selecione BERTimbau ou SVM.",
                "Envie de 1 a 16 avaliações por chamada.",
            }
            message = str(error) if str(error) in allowed else "Entrada inválida. Confira o texto."
        except (ArtifactError, DeviceError, InferenceError):
            message = "Modelo indisponível no momento. Confira a configuração da aplicação."
        except Exception:
            message = "Não foi possível concluir a análise. " + (
                "Tente novamente ou selecione o SVM."
                if enable_svm
                else "Tente novamente mais tarde."
            )
        return public_error_html(message)

    return analyze


def create_app(
    service: InferenceService | None = None,
    *,
    settings: AppSettings | None = None,
    callback_transform: Callable | None = None,
) -> gr.Blocks:
    """Constrói sem carregar modelos, abrir datasets ou executar exemplos."""
    settings = settings or AppSettings()
    check_hosting_authorization(settings)
    service = service if service is not None else service_from_settings(settings)
    analyze = make_analyzer(service, enable_svm=settings.enable_svm)
    if callback_transform is not None:
        analyze = callback_transform(analyze)
    badge = "Demonstração hospedada" if settings.environment == "hosted" else "Demonstração local"
    with gr.Blocks(title="ReviewSense PT-BR", analytics_enabled=False, fill_width=True) as demo:
        gr.HTML(f"""<header class="masthead"><a class="wordmark" href="#">
          <span class="brand-mark" aria-hidden="true">rs</span>
          <span>ReviewSense <b>PT-BR</b></span></a>
          <span class="local-badge"><i></i> {badge}</span></header>
          <div class="intro"><p class="eyebrow">LINGUAGEM NATURAL, EM PORTUGUÊS</p>
          <h1>Entenda o tom<br>de uma avaliação.</h1>
          <p>Uma leitura de sentimento com modelos treinados
          em avaliações de produtos brasileiros.</p>
          </div>""")
        with gr.Row(equal_height=False, elem_id="workspace"):
            with gr.Column(scale=6, min_width=300, elem_id="input-panel"):
                gr.HTML('<span class="section-kicker">01 / SUA AVALIAÇÃO</span>')
                text = gr.Textbox(
                    label="Avaliação em português",
                    placeholder="Como foi a sua experiência com o produto?",
                    lines=6,
                    max_lines=8,
                    max_length=8000,
                    elem_id="review-input",
                    buttons=[],
                )
                gr.HTML(
                    '<p class="input-hint">Até 8.000 caracteres. '
                    "BERTimbau considera até 256 tokens.</p>"
                )
                selector = gr.Radio(
                    choices=[("BERTimbau · principal", "bertimbau")]
                    + ([("SVM · leve", "svm")] if settings.enable_svm else []),
                    value="bertimbau",
                    label="Modelo de análise",
                    elem_id="model-selector",
                )
                button = gr.Button(
                    "Analisar sentimento  →", variant="primary", elem_id="analyze-button"
                )
                gr.HTML(
                    '<p class="privacy-note">A aplicação não grava explicitamente seu texto '
                    "em arquivos ou logs. Evite dados pessoais ou sigilosos. "
                    + (
                        "A plataforma de hospedagem pode manter registros e controles próprios. "
                        if settings.environment == "hosted"
                        else "O processamento é local. "
                    )
                    + "A primeira análise carrega o modelo e pode levar alguns segundos.</p>"
                )
            with gr.Column(scale=5, min_width=300, elem_id="output-panel"):
                output = gr.HTML(EMPTY_RESULT, elem_id="sentiment-result")
        with gr.Column(elem_id="examples-panel", min_width=0):
            gr.HTML(
                '<h2 class="examples-title">Experimente um exemplo sintético</h2>'
                '<p class="examples-hint">Escolha um exemplo '
                "para preencher o campo de avaliação.</p>"
            )
            with gr.Row(elem_id="example-buttons"):
                for label, example in zip(
                    ("Positiva", "Negativa", "Neutra", "Ambígua"), EXAMPLES, strict=True
                ):
                    example_button = gr.Button(label, variant="secondary", min_width=110)
                    example_button.click(
                        lambda value=example: value,
                        inputs=None,
                        outputs=text,
                        queue=False,
                        api_visibility="private",
                    )
        gr.HTML("""<section class="limitations"><h2>Uma estimativa, com contexto.</h2>
          <p>Os rótulos vêm de estrelas de avaliações de produtos de 2018. Textos neutros,
          ironia e sentimentos mistos podem ser difíceis de identificar. O treinamento
          não foi específico para restaurantes ou delivery.</p></section>
          <footer class="technical-footer"><span>Feito para explorar NLP com transparência.</span>
          <nav aria-label="Recursos técnicos">
          <a href="https://github.com/jorgeribeiru/Review-Sense"
          target="_blank" rel="noopener noreferrer">GitHub ↗</a>
          <a href="https://github.com/jorgeribeiru/Review-Sense/blob/main/reports/validation_phase3c.md"
          target="_blank" rel="noopener noreferrer">Método e resultados ↗</a></nav></footer>""")
        button.click(
            analyze,
            [text, selector],
            output,
            api_name="analyze",
            api_visibility="undocumented",
            show_progress="minimal",
            concurrency_id="inference",
            concurrency_limit=1,
        )
        text.submit(
            analyze,
            [text, selector],
            output,
            api_name="analyze_enter",
            api_visibility="undocumented",
            show_progress="minimal",
            concurrency_id="inference",
            concurrency_limit=1,
        )
    demo.queue(default_concurrency_limit=1, max_size=8, api_open=False)
    return demo


def launch_options(*, port: int = 7860, public_network: bool = False) -> dict:
    """Exposição na rede só por opção explícita; nunca cria túnel público."""
    palette = {
        "body_background_fill": "#f7f8f5",
        "body_text_color": "#172a24",
        "block_background_fill": "#ffffff",
        "block_label_background_fill": "transparent",
        "block_label_text_color": "#4b5f52",
        "input_background_fill": "#fcfdfa",
        "input_border_color": "#dbe2d9",
        "input_placeholder_color": "#637168",
        "block_border_color": "#dde3db",
        "button_secondary_background_fill": "#f5f8f1",
        "button_secondary_text_color": "#284534",
    }
    theme = gr.themes.Base(
        font=["system-ui", "sans-serif"], font_mono=["ui-monospace", "monospace"]
    ).set(**palette, **{key + "_dark": value for key, value in palette.items()})
    return {
        "server_name": "0.0.0.0" if public_network else "127.0.0.1",
        "server_port": port,
        "share": False,
        "show_error": False,
        "quiet": True,
        "inbrowser": False,
        "run_history": False,
        "enable_monitoring": False,
        "ssr_mode": False,
        "mcp_server": False,
        "pwa": False,
        "footer_links": [],
        "allowed_paths": [],
        "blocked_paths": [
            str(ROOT / name)
            for name in (
                "models",
                "data",
                "cache",
                "outputs",
                "reports",
                "publishing",
                ".git",
                ".venv",
                ".env",
            )
        ],
        "css": CSS,
        "theme": theme,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ReviewSense PT-BR — demonstração local")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--environment", choices=["local", "hosted"], default="local")
    parser.add_argument("--model-origin", choices=["local", "hub-cache"], default="local")
    parser.add_argument("--model-dir", type=Path, default=AppSettings.model_dir)
    parser.add_argument("--descriptor", type=Path, default=AppSettings.descriptor)
    parser.add_argument("--allowed-root", type=Path, default=AppSettings.allowed_root)
    parser.add_argument("--repo-id")
    parser.add_argument("--revision")
    parser.add_argument("--approval-root", type=Path)
    parser.add_argument("--cache-root", type=Path, default=AppSettings.cache_root)
    parser.add_argument("--preload", action="store_true", help="Carrega BERT antes de servir.")
    parser.add_argument("--bert-only", action="store_true", help="Oculta o baseline local.")
    parser.add_argument(
        "--public-network",
        action="store_true",
        help="Permite acesso pela rede local (0.0.0.0), sem share/túnel.",
    )
    args = parser.parse_args()
    try:
        settings = AppSettings(
            environment=args.environment,
            origin=args.model_origin,
            model_dir=args.model_dir,
            descriptor=args.descriptor,
            allowed_root=args.allowed_root,
            repo_id=args.repo_id,
            revision=args.revision,
            approval_root=args.approval_root,
            cache_root=args.cache_root,
            enable_svm=not args.bert_only and args.environment == "local",
        )
        service = service_from_settings(settings, device=args.device)
        if args.preload or args.environment == "hosted":
            service.prepare()
    except Exception:
        parser.exit(2, "Configuração/modelo indisponível ou lançamento não autorizado.\n")
    options = launch_options(port=args.port, public_network=args.public_network)
    options["blocked_paths"].extend(
        str(path.absolute())
        for path in (
            settings.model_dir,
            settings.descriptor,
            settings.cache_root,
            settings.approval_root,
        )
        if path is not None
    )
    create_app(service, settings=settings).launch(**options)


if __name__ == "__main__":
    main()
