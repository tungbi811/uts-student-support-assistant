"""
Evaluation pipeline for the UTS Student Support Assistant.

Metrics:
  - Correctness      (0-1): does the answer match the reference? (LLM-as-judge)
  - Faithfulness     (0-1): is the answer grounded in retrieved context? (LLM-as-judge)
  - Answer Relevancy (0-1): does the answer directly address the question? (LLM-as-judge)
  - Source hit rate  (0/1): did the retriever return the expected source URL?

Usage:
    # Default (semantic chunks, RAG enabled)
    python src/evaluate.py

    # No-retrieval baseline (vanilla LLM, no FAISS)
    python src/evaluate.py --no-retrieval --output data/eval_results_no_rag.json

    # Alternative chunk index
    python src/evaluate.py --index-path data/faiss_index_fixed500 --output data/eval_results_fixed500.json
"""
import json
import sys
import argparse
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

sys.path.insert(0, "src")
from rag import load_config, load_vectorstore, build_rag_chain


CORRECTNESS_PROMPT = """You are an impartial evaluator. Score how well the system answer matches the reference answer.

Question: {question}
Reference answer: {reference}
System answer: {answer}

Score from 0.0 to 1.0 where:
  1.0 = fully correct and complete
  0.5 = partially correct
  0.0 = incorrect or missing key information

Reply with a single number only."""


FAITHFULNESS_PROMPT = """You are an impartial evaluator. Score whether the system answer is grounded in the provided context and does not hallucinate information.

Context: {context}
System answer: {answer}

Score from 0.0 to 1.0 where:
  1.0 = fully grounded in context, no hallucination
  0.5 = mostly grounded with minor unsupported claims
  0.0 = contains significant information not found in context

Reply with a single number only."""


RELEVANCY_PROMPT = """You are an impartial evaluator. Score whether the system answer directly addresses the question asked.

Question: {question}
System answer: {answer}

Score from 0.0 to 1.0 where:
  1.0 = directly and fully addresses the question
  0.5 = partially addresses the question or includes significant irrelevant content
  0.0 = does not address the question at all

Reply with a single number only."""


def parse_score(text):
    try:
        return max(0.0, min(1.0, float(text.strip())))
    except ValueError:
        return 0.0


NO_RETRIEVAL_PROMPT = """You are a helpful university assistant. Answer the following question about UTS (University of Technology Sydney) as best you can.

Question: {question}

Answer:"""


def run_evaluation(config_path="config/config.yaml", test_set_path="data/test_set.json",
                   output_path="data/eval_results.json", no_retrieval=False, index_path=None):
    config = load_config(config_path)

    if index_path:
        config["vectorstore"]["index_path"] = index_path

    judge = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
    correctness_chain  = PromptTemplate.from_template(CORRECTNESS_PROMPT)  | judge | StrOutputParser()
    faithfulness_chain = PromptTemplate.from_template(FAITHFULNESS_PROMPT) | judge | StrOutputParser()
    relevancy_chain    = PromptTemplate.from_template(RELEVANCY_PROMPT)    | judge | StrOutputParser()

    if no_retrieval:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=config["llm"]["model"], temperature=0)
        no_retrieval_chain = PromptTemplate.from_template(NO_RETRIEVAL_PROMPT) | llm | StrOutputParser()
        print("Mode: NO-RETRIEVAL baseline (vanilla LLM)\n")
    else:
        vectorstore = load_vectorstore(config)
        chain, _ = build_rag_chain(vectorstore, config)
        print(f"Mode: RAG  index={config['vectorstore']['index_path']}\n")

    with open(test_set_path) as f:
        test_set = json.load(f)

    results = []

    for i, item in enumerate(test_set):
        question        = item["question"]
        reference       = item["reference"]
        expected_source = item.get("expected_source", "")

        print(f"[{i+1}/{len(test_set)}] {question}")

        if no_retrieval:
            answer   = no_retrieval_chain.invoke({"question": question})
            sources  = []
            context  = ""
            source_hit = 0
        else:
            answer, docs = chain({"question": question, "chat_history": []})
            sources  = [doc.metadata["url"] for doc in docs]
            context  = "\n\n".join(doc.page_content for doc in docs)
            source_hit = int(any(expected_source in url for url in sources)) if expected_source else 0

        correctness = parse_score(correctness_chain.invoke(
            {"question": question, "reference": reference, "answer": answer}
        ))
        faithfulness = parse_score(faithfulness_chain.invoke(
            {"context": context if context else "No context provided.", "answer": answer}
        ))
        relevancy = parse_score(relevancy_chain.invoke(
            {"question": question, "answer": answer}
        ))

        print(f"  correctness={correctness:.2f}  faithfulness={faithfulness:.2f}  "
              f"relevancy={relevancy:.2f}  source_hit={source_hit}")

        results.append({
            "question":         question,
            "answer":           answer,
            "reference":        reference,
            "sources":          sources,
            "source_hit":       source_hit,
            "correctness":      correctness,
            "faithfulness":     faithfulness,
            "answer_relevancy": relevancy,
        })

    n = len(results)
    summary = {
        "n":                n,
        "correctness":      sum(r["correctness"]      for r in results) / n,
        "faithfulness":     sum(r["faithfulness"]     for r in results) / n,
        "answer_relevancy": sum(r["answer_relevancy"] for r in results) / n,
        "source_hit_rate":  sum(r["source_hit"]       for r in results) / n,
    }

    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Questions evaluated : {summary['n']}")
    print(f"Correctness         : {summary['correctness']:.3f}")
    print(f"Faithfulness        : {summary['faithfulness']:.3f}")
    print(f"Answer Relevancy    : {summary['answer_relevancy']:.3f}")
    print(f"Source hit rate     : {summary['source_hit_rate']:.3f}")
    print("=" * 50)

    with open(output_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-set",     default="data/test_set.json")
    parser.add_argument("--output",       default="data/eval_results.json")
    parser.add_argument("--no-retrieval", action="store_true",
                        help="Vanilla LLM baseline — skip FAISS retrieval entirely")
    parser.add_argument("--index-path",   default=None,
                        help="Override FAISS index path (e.g. data/faiss_index_fixed500)")
    args = parser.parse_args()
    run_evaluation(
        test_set_path=args.test_set,
        output_path=args.output,
        no_retrieval=args.no_retrieval,
        index_path=args.index_path,
    )
