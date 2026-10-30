"""Tabelas e figuras somente a partir de agregados; nunca abre CSV/modelos."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from reviewsense.evaluate import plot_confusion_matrix
from reviewsense.labels import LABEL_NAMES


def comparison_table(result: dict) -> list[dict]:
    return [
        {
            "modelo": name,
            **{
                key: row["metrics"][key]
                for key in ("f1_macro", "precision_macro", "recall_macro", "accuracy")
            },
        }
        for name, row in result["models"].items()
    ]


def render_report(result: dict, directory: Path, *, detailed: bool = True) -> str:
    """Pode ser executada novamente; só reconstitui figuras e texto dos agregados."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from reviewsense.final_evaluation import validate_summary

    validate_summary(result)
    figures = directory / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    names = ["svm", "bertimbau"]
    titles = ["TF-IDF + LinearSVC (CPU)", "BERTimbau (GPU)"]
    colors = ["#2878A0", "#C66535"]
    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        values = [result["models"][name]["metrics"]["f1_macro"] for name in names]
        bars = ax.bar(titles, values, color=colors, width=0.55)
        ax.bar_label(bars, fmt="%.4f", padding=4)
        ax.set(ylim=(0, 1), ylabel="F1 Macro", title="Avaliação independente — conjunto de teste")
        fig.tight_layout()
        fig.savefig(figures / "phase3c_f1_comparison.png", dpi=170)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        positions = np.arange(3)
        for i, name in enumerate(names):
            values = [
                result["models"][name]["metrics"]["per_class"][label]["f1-score"]
                for label in LABEL_NAMES.values()
            ]
            bars = ax.bar(
                positions + (i - 0.5) * 0.36, values, width=0.36, color=colors[i], label=titles[i]
            )
            ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
        ax.set(
            xticks=positions,
            xticklabels=list(LABEL_NAMES.values()),
            ylim=(0, 1.08),
            ylabel="F1",
            title="F1 por classe — teste",
        )
        ax.legend(loc="upper left", bbox_to_anchor=(0, 1.18), ncols=2, fontsize=9)
        fig.tight_layout()
        fig.savefig(figures / "phase3c_per_class_f1.png", dpi=170)
        plt.close(fig)
        for name, title in zip(names, titles, strict=True):
            fig = plot_confusion_matrix(result["models"][name]["metrics"], title=f"{title} — teste")
            fig.savefig(figures / f"phase3c_{name}_confusion.png", dpi=170)
            plt.close(fig)

        if detailed:
            benchmark = result["preflight"]["benchmark"]
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            for axis, key, label in zip(
                axes,
                ("texts_per_second", "milliseconds_per_text_amortized"),
                ("Textos/s (escala log)", "ms/texto em batch (escala log)"),
                strict=True,
            ):
                values = [benchmark[name][key] for name in names]
                bars = axis.bar(titles, values, color=colors)
                axis.bar_label(bars, fmt="%.3f", padding=4)
                axis.set(yscale="log", ylabel=label)
                axis.margins(y=0.3)
                axis.tick_params(axis="x", labelsize=9)
            fig.suptitle("Inferência pré-teste — validação, mediana de 3 passagens/1.024 textos")
            fig.tight_layout()
            fig.savefig(figures / "phase3c_inference_comparison.png", dpi=170)
            plt.close(fig)

    if not detailed:
        return "Relatório sintético validado."
    return markdown_report(result)


def markdown_report(r: dict) -> str:
    models = r["models"]
    svm, bert = models["svm"]["metrics"], models["bertimbau"]["metrics"]
    rows = "\n".join(
        f"| {entry['modelo']} | "
        + " | ".join(
            f"{entry[key]:.6f}"
            for key in ("f1_macro", "precision_macro", "recall_macro", "accuracy")
        )
        + " |"
        for entry in comparison_table(r)
    )
    class_rows = "\n".join(
        f"| {name} | {label} | {values['precision']:.6f} | {values['recall']:.6f} | "
        f"{values['f1-score']:.6f} | {int(values['support'])} |"
        for name, model in models.items()
        for label, values in model["metrics"]["per_class"].items()
    )
    neutral_rows = "\n".join(
        f"| {name} | {model['metrics']['confusion_matrix'][1][1]} | "
        f"{model['metrics']['confusion_matrix'][1][0]} | "
        f"{model['metrics']['confusion_matrix'][1][2]} |"
        for name, model in models.items()
    )
    per_class_delta = "\n".join(
        f"- {label}: ΔPrecision "
        f"{bert['per_class'][label]['precision'] - svm['per_class'][label]['precision']:+.6f}, "
        f"ΔRecall {bert['per_class'][label]['recall'] - svm['per_class'][label]['recall']:+.6f}, "
        f"ΔF1 {bert['per_class'][label]['f1-score'] - svm['per_class'][label]['f1-score']:+.6f}."
        for label in LABEL_NAMES.values()
    )
    paired = r["paired_agreement"]
    historical = r["historical_training"]
    sizes = historical["artifact_bytes"]
    benchmark = r["preflight"]["benchmark"]
    bench_rows = "\n".join(
        f"| {name} | {row['batch_size']} | {row['median_seconds']:.6f} | "
        f"{row['texts_per_second']:.2f} | {row['milliseconds_per_text_amortized']:.6f} |"
        for name, row in benchmark.items()
    )
    pass_rows = "\n".join(
        f"| {name} | {model['inference']['seconds']:.6f} | "
        f"{model['inference']['texts_per_second']:.2f} | "
        f"{model['inference']['milliseconds_per_text_amortized']:.6f} | "
        f"{model['errors']} | {model['error_fraction']:.4%} |"
        for name, model in models.items()
    )
    gpu = models["bertimbau"]["inference"]
    difference = r["comparison"]["f1_macro_difference_absolute"]
    relative = r["comparison"]["f1_macro_difference_relative"]
    relative_text = f"{relative:+.4%}" if relative is not None else "indefinida (SVM = 0)"
    inventory_rows = "\n".join(
        f"| `{name}` | `{sha}` |"
        for name, sha in r["inventory"].items()
        if not name.startswith("src/")
    )
    interpretation = (
        "A vantagem descritiva de F1 Macro observada na validação se manteve no teste."
        if difference > 0
        else "A vantagem descritiva de F1 Macro da validação não se manteve no teste."
    )
    return f"""# Fase 3C — Avaliação final independente

Execução local definitiva: {r["started_at"]} a {r["finished_at"]}.
Commit congelado `{r["git"]["commit"]}`, branch `{r["git"]["branch"]}`, árvore limpa.
O protocolo [FASE3C_PROTOCOL.md](../docs/FASE3C_PROTOCOL.md) foi versionado antes
da abertura do teste. Nenhum treinamento, refit, calibração ou seleção nesta fase.
BERTimbau já era o candidato principal; o teste não redefine essa escolha.

## Procedimento e dados

B2W-Reviews01 da Americanas, avaliações de produtos de 2018. Fonte/revisão e
fingerprints ancorados na Fase 1. Licença original **CC BY-NC-SA 4.0**; sem
redistribuição ou presunção de autorização comercial. Estrelas são proxy de
sentimento: negativo 0 (1–2), neutro 1 (3), positivo 2 (4–5).

Teste congelado: **{r["test"]["rows"]}** exemplos. Ambos receberam os mesmos textos,
rótulos e ordem. Conferidos esquema, nulos, normalização existente, IDs, unicidade,
ordem do manifest, fingerprints/distribuição, disjunção e fingerprint global.
Somente review_text como feature. Os dados não foram corrigidos nem regenerados.

- Preflight: **zero leituras/tentativas bloqueadas** do teste; guarda de abertura ativa.
- Preparação da execução definitiva: **{r["test"]["preparation_read_opens"]} leituras** do teste.
- Final: **{r["test"]["read_opens"]} abertura de leitura**, {r["test"]["blocked_opens"]} bloqueadas;
  hash e parsing a partir do mesmo buffer. Uma passagem de inferência por modelo.
- Registro exclusivo criado antes da leitura; falhas não permitem repetição automática.
- SHA do CSV: `{r["test"]["csv_sha256"]}`.
- Fingerprint preparado: `{r["test"]["fingerprint"]}`; sobreposição de IDs: zero.
- Arquivos de entrada/modelos com hashes preservados; stat do teste inalterado.

SVM: pipeline da Fase 2, C=0,1, ngrams=(1,2), class_weight=balanced, 80.000 features.
BERTimbau: época 1/update 5543 da Fase 3B, best_model igual a checkpoint-5543;
origem neuralmind/bert-base-portuguese-cased, revisão
94d69c95f98f7d5b2a8700c420230ae10def0baa. Inferência offline/cuda:0, eval,
inference_mode, FP16, max_length=256 e padding dinâmico/múltiplo de 8.

## Resultados reais de teste

| Modelo | F1 Macro | Precision Macro | Recall Macro | Accuracy |
| --- | ---: | ---: | ---: | ---: |
{rows}

ΔF1 Macro BERT−SVM: **{difference:+.6f}** ({difference * 100:+.4f} pontos percentuais),
variação relativa **{relative_text}** sobre o SVM. {interpretation}
Comparação descritiva; sem bootstrap, intervalo de confiança ou alegação de
superioridade estatística. Valores de validação anteriores permanecem inalterados
(SVM 0,727137; BERT 0,755522).

| Modelo | Classe | Precision | Recall | F1 | Suporte |
| --- | --- | ---: | ---: | ---: | ---: |
{class_rows}

Diferenças por classe (BERT−SVM):

{per_class_delta}

![F1 Macro](figures/phase3c_f1_comparison.png)
![F1 por classe](figures/phase3c_per_class_f1.png)

## Neutros e concordância pareada

| Modelo | Neutros corretos | Neutros → negativos | Neutros → positivos |
| --- | ---: | ---: | ---: |
{neutral_rows}

As diferenças de Precision e Recall acima mostram os ganhos/perdas separadamente;
as matrizes são evidência de contagens, não explicação causal dos erros.

- Ambos acertaram: **{paired["both_correct"]}**.
- Ambos erraram: **{paired["both_wrong"]}**.
- Somente SVM acertou: **{paired["only_svm_correct"]}**.
- Somente BERTimbau acertou: **{paired["only_bertimbau_correct"]}**.
- Soma: **{paired["samples"]}**; isso não identifica correlação entre autores/produtos.

Matrizes com ordem negativo/neutro/positivo, linhas reais e colunas previstas:

![SVM](figures/phase3c_svm_confusion.png)
![BERTimbau](figures/phase3c_bertimbau_confusion.png)

## Desempenho computacional

CPU: {historical["cpu_model"]}, {historical["logical_cpus"]} CPUs lógicas;
SVM limitado a uma thread. GPU: {r["gpu"]["name"]}; BERT usa duas threads CPU e
FP16. Software: Python {r["software"]["python"]}, Torch {r["software"]["torch"]},
Transformers {r["software"]["transformers"]}, sklearn {r["software"]["scikit-learn"]}.
CUDA {r["gpu"]["cuda_build"]}. Versões e snapshot do driver estão no JSON.

Benchmark **antes de abrir o teste**: mesmos 1.024 textos de validação, amostragem
estratificada seed 42, índices ordenados; warm-up 32 textos/modelo e três repetições.
Mediana por passagem, sincronização CUDA, inclui tokenização/TF-IDF, transferências
e predição; exclui carga, aquecimento, I/O e cálculo de métricas.

| Modelo | Batch | Passagem mediana (s) | Textos/s | ms/texto amortizado |
| --- | ---: | ---: | ---: | ---: |
{bench_rows}

![Benchmark](figures/phase3c_inference_comparison.png)

Passagem **única** pelo teste, sem repetir ou ajustar configurações:

| Modelo | Passagem 19.004 (s) | Textos/s | ms/texto amortizado | Erros | Fração de erros |
| --- | ---: | ---: | ---: | ---: | ---: |
{pass_rows}

Throughput amortizado em batch não é latência individual (não medida). CPU e GPU
são infraestruturas diferentes; temperatura, frequência e outros processos não
foram isolados. O tempo do teste também inclui condições iniciais dessa passagem.

- RSS pico amostrado do processo na passagem SVM:
  {models["svm"]["inference"]["process_rss_peak_sampled_bytes"] / 2**20:.2f} MiB.
  Inclui ambos os modelos carregados e runtime; **não** é memória exclusiva do SVM.
  Amostragem Linux de 10 ms, não um pico contínuo exato.
- Inferência BERT/teste: PyTorch pico alocado
  **{gpu["pytorch"]["peak_allocated_bytes"] / 2**30:.3f} GiB**,
  reservado **{gpu["pytorch"]["peak_reserved_bytes"] / 2**30:.3f} GiB**;
  driver pico amostrado **{gpu["driver"]["peak_used_bytes_sampled"] / 2**30:.3f} GiB**,
  livre mínimo **{gpu["driver"]["min_free_bytes"] / 2**30:.3f} GiB** (NVML 50 ms,
  inclui desktop/driver/outros processos).
- Artefato SVM: {sizes["svm_joblib"]:,} bytes joblib comprimido.
  BERT pesos: {sizes["bert_weights"]:,} bytes; tokenizer: {sizes["bert_tokenizer"]:,} bytes.
- Treino histórico SVM selecionado: {historical["svm_fit_seconds"]:.6f} s;
  busca+validação completa: {historical["svm_search_validation_seconds"]:.6f} s.
  BERT: {historical["bert_three_epochs_validation_checkpoints_seconds"]:.3f} s para
  três épocas com validações/checkpoints. Escopos diferentes, sem novo treino.

## Auditoria e limitações

Hashes dos artefatos e referências (hashes do código também constam no JSON):

| Arquivo relativo ao repositório | SHA-256 |
| --- | --- |
{inventory_rows}

Resultados estruturados: [phase3c_final_metrics.json](phase3c_final_metrics.json).
Guia e comandos: [FASE3C_FINAL_EVALUATION.md](../docs/FASE3C_FINAL_EVALUATION.md).
As figuras/notebook só reabrem agregados, sem ler teste ou inferir novamente.

Não há explicação causal dos erros, probabilidades calibradas ou evidência de
generalização para delivery, outros domínios ou períodos. Os rótulos são proxy;
há truncamento e possíveis quase duplicatas/autores/produtos compartilhados.
Disjunção de textos idênticos não garante independência desses fatores.
O teste foi reservado até este protocolo; depois desta execução deixa de ser
informação não observada e não deve guiar novos ajustes neste experimento.
Publicação, Model Card e Gradio ficam para a Fase 4.
"""


def publish_saved(config: dict) -> None:
    """Só lê resultados locais; resultados públicos existentes devem ser idênticos."""
    from reviewsense.final_evaluation import atomic_json, read_json, validate_summary

    result = read_json(Path(config["run_dir"]) / "results.json")
    validate_summary(result)
    if result.get("status") != "completed":
        raise ValueError("Execução final não concluída.")
    destination = Path("reports/phase3c_final_metrics.json")
    if destination.exists():
        if read_json(destination) != result:
            raise FileExistsError("Recusada substituição de métricas finais diferentes.")
    else:
        atomic_json(destination, result)
    text = render_report(result, Path("reports"))
    Path("reports/validation_phase3c.md").write_text(text, encoding="utf-8")
