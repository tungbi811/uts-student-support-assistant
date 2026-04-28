"""
Automated evaluation pipeline for the UTS Student Support Assistant.

Metrics (all automated, no human needed):
  - Faithfulness        : answer is grounded in retrieved context (no hallucination)
  - Answer Relevancy    : answer actually addresses the question
  - Context Precision   : retrieved chunks are relevant (not noisy)
  - Context Recall      : retrieved chunks contain the info needed to answer
  - Source Hit Rate     : expected source URL appears in retrieved chunks
  - Correctness         : answer matches reference answer (LLM-as-judge)

Usage:
    python src/evaluate.py
    python src/evaluate.py --test-set data/test_set.json --output data/eval_results.json
"""
import json
import sys
import argparse
import yaml

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

sys.path.insert(0, "src")
from rag import load_config, load_vectorstore, build_rag_chain


CORRECTNESS_PROMPT = PromptTemplate.from_template(
    """You are an impartial evaluator. Score how well the system answer matches the reference answer.

Question: {question}
Reference answer: {reference}
System answer: {answer}

Score from 0.0 to 1.0:
  1.0 = fully correct and complete
  0.5 = partially correct
  0.0 = incorrect or missing key information

Reply with a single number only."""
)


def parse_score(text):
    try:
        return max(0.0, min(1.0, float(text.strip())))
    except ValueError:
        return 0.0


def run_evaluation(test_set_path="data/test_set.json", output_path="data/eval_results.json"):
    config = load_config()
    vectorstore = load_vectorstore(config)
    chain, _ = build_rag_chain(vectorstore, config)

    judge = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    correctness_chain = CORRECTNESS_PROMPT | judge | StrOutputParser()

    with open(test_set_path) as f:
        test_set = json.load(f)

    print(f"Evaluating {len(test_set)} questions...\n")

    # ── Collect RAG outputs ────────────────────────────────────────────────────
    questions, answers, contexts, ground_truths = [], [], [], []
    source_hits, correctness_scores = [], []
    per_item = []

    for i, item in enumerate(test_set):
        question   = item["question"]
        reference  = item["reference"]
        expected   = item.get("expected_source", "")

        print(f"[{i+1}/{len(test_set)}] {question}")

        answer, docs = chain({"question": question, "chat_history": []})
        chunk_texts  = [doc.page_content for doc in docs]
        source_urls  = [doc.metadata["url"] for doc in docs]

        # Source hit rate
        hit = int(any(expected in url for url in source_urls)) if expected else 0

        # Correctness (LLM-as-judge)
        correctness = parse_score(correctness_chain.invoke({
            "question": question,
            "reference": reference,
            "answer": answer,
        }))

        print(f"  correctness={correctness:.2f}  source_hit={hit}")

        questions.append(question)
        answers.append(answer)
        contexts.append(chunk_texts)
        ground_truths.append(reference)
        source_hits.append(hit)
        correctness_scores.append(correctness)

        per_item.append({
            "question":    question,
            "answer":      answer,
            "reference":   reference,
            "sources":     source_urls,
            "source_hit":  hit,
            "correctness": correctness,
        })

    # ── RAGAS evaluation ───────────────────────────────────────────────────────
    print("\nRunning RAGAS metrics...")
    dataset = Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths,
    })

    ragas_result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    ragas_scores = ragas_result.to_pandas()

    # Attach RAGAS per-item scores
    for i, item in enumerate(per_item):
        item["faithfulness"]       = float(ragas_scores["faithfulness"].iloc[i])
        item["answer_relevancy"]   = float(ragas_scores["answer_relevancy"].iloc[i])
        item["context_precision"]  = float(ragas_scores["context_precision"].iloc[i])
        item["context_recall"]     = float(ragas_scores["context_recall"].iloc[i])

    # ── Summary ────────────────────────────────────────────────────────────────
    n = len(per_item)
    summary = {
        "n":                  n,
        "correctness":        sum(r["correctness"] for r in per_item) / n,
        "source_hit_rate":    sum(r["source_hit"] for r in per_item) / n,
        "faithfulness":       float(ragas_scores["faithfulness"].mean()),
        "answer_relevancy":   float(ragas_scores["answer_relevancy"].mean()),
        "context_precision":  float(ragas_scores["context_precision"].mean()),
        "context_recall":     float(ragas_scores["context_recall"].mean()),
    }

    print("\n" + "=" * 55)
    print("EVALUATION SUMMARY")
    print("=" * 55)
    print(f"Questions evaluated  : {summary['n']}")
    print(f"Correctness          : {summary['correctness']:.2f}")
    print(f"Source hit rate      : {summary['source_hit_rate']:.2f}")
    print(f"Faithfulness         : {summary['faithfulness']:.2f}  (no hallucination)")
    print(f"Answer Relevancy     : {summary['answer_relevancy']:.2f}  (addresses the question)")
    print(f"Context Precision    : {summary['context_precision']:.2f}  (retrieved chunks are relevant)")
    print(f"Context Recall       : {summary['context_recall']:.2f}  (chunks cover the answer)")
    print("=" * 55)

    output = {"summary": summary, "results": per_item}
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-set", default="data/test_set.json")
    parser.add_argument("--output",   default="data/eval_results.json")
    args = parser.parse_args()
    run_evaluation(test_set_path=args.test_set, output_path=args.output)
