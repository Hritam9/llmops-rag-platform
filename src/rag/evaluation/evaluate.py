"""Evaluation pipeline: runs the RAG chain against a fixed question set,
scores it with ragas metrics, logs results to MLflow, and exits non-zero
if quality drops below the configured threshold (used as a CI gate).

Usage:
    python -m src.rag.evaluation.evaluate

Transparency: every question, its generated answer, and its per-metric
scores are printed to the console AND written as a Markdown table to
GitHub Actions' job summary (visible in the Summary tab of the run,
no digging through raw logs required) when running in CI.
"""
import json
import math
import os
import sys

import mlflow
from datasets import Dataset
from langchain_groq import ChatGroq
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, faithfulness

from src.config import load_config
from src.rag.generation.rag_chain import RAGChain
from src.rag.ingestion.embedder import get_embedding_model

METRIC_MAP = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "context_precision": context_precision,
}

# ragas changed its dataset column schema between versions:
#   old (<=0.1.x): question / answer / contexts / ground_truth
#   new (>=0.2.x): user_input / response / retrieved_contexts / reference
# We build the dataset with the new (current) schema below, but keep this
# resolver so reporting code never hard-codes one naming scheme and can't
# be broken again by a future ragas version renaming columns.
QUESTION_KEYS = ("user_input", "question")
ANSWER_KEYS = ("response", "answer")


def _col(row_or_df, keys, default=""):
    """Return the first present column/key from `keys`, or `default`."""
    for k in keys:
        try:
            if k in row_or_df:
                return row_or_df[k]
        except TypeError:
            continue
    return default


def _build_ragas_judge(config: dict):
    """Ragas needs its own LLM + embeddings to grade answers — this is
    separate from the LLM your RAG chain uses to *generate* answers.
    By default ragas.evaluate() falls back to OpenAI, which fails with no
    OPENAI_API_KEY. We point it at Groq (free tier) instead.

    IMPORTANT: some ragas metrics (answer_relevancy in particular) ask the
    judge LLM to produce strict structured output. Small/fast models are
    unreliable at this and silently return NaN for those rows. We use a
    separate, more capable judge model (configs/config.yaml ->
    evaluation.judge_model_name) rather than reusing the fast generation
    model, specifically to avoid that failure mode.
    """
    judge_model_name = config["evaluation"].get(
        "judge_model_name", config["generation"]["model_name"]
    )
    judge_llm = ChatGroq(
        model=judge_model_name,
        temperature=0,  # deterministic grading
    )
    judge_embeddings = get_embedding_model(
        config["embedding"]["model_name"],
        config["embedding"]["device"],
    )
    return LangchainLLMWrapper(judge_llm), LangchainEmbeddingsWrapper(judge_embeddings)


def load_eval_set(path: str) -> list[dict]:
    with open(path, "r") as f:
        return json.load(f)


def _fmt_score(value) -> str:
    """Render a score for display, making NaN/failed-grading explicit
    instead of letting it silently disappear into an average."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A (grading failed)"
    return f"{value:.3f}"


def _print_per_question_report(scores_df, metric_names: list[str]) -> None:
    """Console-friendly, fully transparent breakdown: for every question,
    show the question, the generated answer, and each metric's score."""
    print("\n" + "=" * 100)
    print("[evaluation] PER-QUESTION BREAKDOWN")
    print("=" * 100)
    for i, row in scores_df.iterrows():
        print(f"\n--- Question {i + 1}/{len(scores_df)} ---")
        print(f"  Q: {_col(row, QUESTION_KEYS)}")
        print(f"  A: {_col(row, ANSWER_KEYS)}")
        for m in metric_names:
            if m in row:
                print(f"  {m}: {_fmt_score(row[m])}")
    print("\n" + "=" * 100)


def _write_github_step_summary(
    scores_df, metric_names: list[str], avg_scores: dict, overall_avg, threshold: float, passed: bool
) -> None:
    """Write a Markdown table to the GitHub Actions job summary, if running
    in GitHub Actions (GITHUB_STEP_SUMMARY env var is set by the runner).
    This shows up directly in the Summary tab of the workflow run — no need
    to scroll through raw step logs to see what was asked, generated, or scored.
    """
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return  # not running in GitHub Actions - skip silently

    lines = ["## RAG Evaluation Report\n"]
    status_emoji = "PASSED ✅" if passed else "FAILED ❌"
    lines.append(f"**Result:** {status_emoji}  ")
    lines.append(f"**Overall average:** {_fmt_score(overall_avg)} (threshold: {threshold})\n")

    lines.append("### Per-metric averages\n")
    lines.append("| Metric | Average |")
    lines.append("|---|---|")
    for m in metric_names:
        lines.append(f"| {m} | {_fmt_score(avg_scores.get(m))} |")

    lines.append("\n### Per-question detail\n")
    header = "| # | Question | Answer | " + " | ".join(metric_names) + " |"
    sep = "|---|---|---|" + "---|" * len(metric_names)
    lines.append(header)
    lines.append(sep)
    for i, row in scores_df.iterrows():
        q = str(_col(row, QUESTION_KEYS)).replace("|", "\\|").replace("\n", " ")
        a = str(_col(row, ANSWER_KEYS)).replace("|", "\\|").replace("\n", " ")
        if len(a) > 200:
            a = a[:200] + "…"
        score_cells = " | ".join(_fmt_score(row.get(m)) for m in metric_names)
        lines.append(f"| {i + 1} | {q} | {a} | {score_cells} |")

    with open(summary_path, "a") as f:
        f.write("\n".join(lines) + "\n")


def run_evaluation(config: dict):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    eval_set = load_eval_set(config["evaluation"]["test_set_path"])
    chain = RAGChain(config)

    questions, answers, contexts, ground_truths = [], [], [], []

    print(f"[evaluation] Running {len(eval_set)} question(s) through the RAG chain...\n")
    for idx, item in enumerate(eval_set):
        result = chain.query(item["question"], log_to_mlflow=False)
        questions.append(item["question"])
        answers.append(result["answer"])
        contexts.append([c["text"] for c in result["retrieved_chunks"]])
        ground_truths.append(item["ground_truth"])
        print(f"[evaluation] ({idx + 1}/{len(eval_set)}) Q: {item['question']}")
        print(f"[evaluation]        A: {result['answer']}\n")

    # ragas >=0.2 expects this schema: user_input / response / retrieved_contexts / reference
    dataset = Dataset.from_dict({
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": ground_truths,
    })

    metric_names = config["evaluation"]["metrics"]
    metrics = [METRIC_MAP[m] for m in metric_names]
    judge_llm, judge_embeddings = _build_ragas_judge(config)

    judge_name = config["evaluation"].get("judge_model_name", config["generation"]["model_name"])
    print(f"[evaluation] Grading with judge model: {judge_name}")
    scores = evaluate(
        dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
    )
    scores_df = scores.to_pandas()

    # Full transparency: print every question, answer, and per-metric score.
    _print_per_question_report(scores_df, metric_names)

    # Per-metric averages, with an explicit NaN check so failed gradings are
    # visible rather than silently skipped by pandas' default skipna mean.
    avg_scores = {}
    for m in metric_names:
        if m not in scores_df.columns:
            continue
        col = scores_df[m]
        nan_count = int(col.isna().sum())
        if nan_count:
            print(f"[evaluation] WARNING: {m} failed to grade on {nan_count}/{len(col)} "
                  f"question(s) (judge LLM did not return a parseable score). "
                  f"Marking this metric as failed rather than averaging around it.")
            avg_scores[m] = float("nan")
        else:
            avg_scores[m] = float(col.mean())

    valid_scores = [v for v in avg_scores.values() if not math.isnan(v)]
    if len(valid_scores) < len(avg_scores):
        overall_avg = float("nan")
    else:
        overall_avg = sum(valid_scores) / len(valid_scores) if valid_scores else float("nan")

    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_params({
            "num_eval_questions": len(eval_set),
            "judge_model_name": judge_name,
        })
        # MLflow rejects NaN metric values, so log a -1 sentinel when grading failed
        # (never a silently-dropped metric, and never a value that could look like
        # a real passing score).
        loggable_metrics = {k: (v if not math.isnan(v) else -1.0) for k, v in avg_scores.items()}
        mlflow.log_metrics(loggable_metrics)
        mlflow.log_metric("overall_avg_score", overall_avg if not math.isnan(overall_avg) else -1.0)

    print("\n[evaluation] Per-metric averages:")
    for k, v in avg_scores.items():
        print(f"  {k}: {_fmt_score(v)}")
    print(f"[evaluation] Overall average: {_fmt_score(overall_avg)}")

    threshold = config["evaluation"]["regression_threshold"]
    # NaN-safe pass/fail. Plain `nan < threshold` is False in Python, which
    # is exactly the bug that let a broken run silently report PASSED before.
    passed = (not math.isnan(overall_avg)) and (overall_avg >= threshold)

    _write_github_step_summary(scores_df, metric_names, avg_scores, overall_avg, threshold, passed)

    return overall_avg, passed


if __name__ == "__main__":
    cfg = load_config()
    avg, passed = run_evaluation(cfg)
    threshold = cfg["evaluation"]["regression_threshold"]

    if not passed:
        if math.isnan(avg):
            print("[evaluation] FAILED: one or more metrics could not be graded (NaN) - "
                  "treating as a failure rather than silently passing.")
        else:
            print(f"[evaluation] FAILED: avg score {avg:.3f} is below threshold {threshold}")
        sys.exit(1)
    else:
        print(f"[evaluation] PASSED: avg score {avg:.3f} meets threshold {threshold}")
        sys.exit(0)
