"""
Evaluation pipeline for the UTS Student Support Assistant.

Metrics:
  - Correctness  (0-1): does the answer match the reference? (LLM-as-judge)
  - Faithfulness (0-1): is the answer grounded in retrieved context? (LLM-as-judge)
  - Source hit rate (0/1): did the retriever return the expected source URL?
"""
import json
import sys
import yaml
from langchain_openai import ChatOpenAI
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


def parse_score(text):
    try:
        return max(0.0, min(1.0, float(text.strip())))
    except ValueError:
        return 0.0


def evaluate(config_path="config/config.yaml", test_set_path="data/test_set.json"):
    config = load_config(config_path)
    vectorstore = load_vectorstore(config)
    chain, retriever = build_rag_chain(vectorstore, config)

    judge = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    correctness_chain = PromptTemplate.from_template(CORRECTNESS_PROMPT) | judge | StrOutputParser()
    faithfulness_chain = PromptTemplate.from_template(FAITHFULNESS_PROMPT) | judge | StrOutputParser()

    with open(test_set_path) as f:
        test_set = json.load(f)

    results = []

    for i, item in enumerate(test_set):
        question = item["question"]
        reference = item["reference"]
        expected_source = item["expected_source"]

        print(f"[{i+1}/{len(test_set)}] {question}")

        # Get answer and retrieved docs
        answer = chain({"question": question, "chat_history": []})
        docs = retriever.invoke(question)
        sources = [doc.metadata["url"] for doc in docs]
        context = "\n\n".join(doc.page_content for doc in docs)

        # Source hit rate
        source_hit = int(any(expected_source in url for url in sources))

        # Correctness
        correctness = parse_score(correctness_chain.invoke({
            "question": question,
            "reference": reference,
            "answer": answer
        }))

        # Faithfulness
        faithfulness = parse_score(faithfulness_chain.invoke({
            "context": context,
            "answer": answer
        }))

        results.append({
            "question": question,
            "answer": answer,
            "reference": reference,
            "sources": sources,
            "source_hit": source_hit,
            "correctness": correctness,
            "faithfulness": faithfulness,
        })

        print(f"  correctness={correctness:.2f}  faithfulness={faithfulness:.2f}  source_hit={source_hit}")

    # Summary
    n = len(results)
    avg_correctness = sum(r["correctness"] for r in results) / n
    avg_faithfulness = sum(r["faithfulness"] for r in results) / n
    avg_source_hit = sum(r["source_hit"] for r in results) / n

    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Questions evaluated : {n}")
    print(f"Correctness         : {avg_correctness:.2f}")
    print(f"Faithfulness        : {avg_faithfulness:.2f}")
    print(f"Source hit rate     : {avg_source_hit:.2f}")
    print("=" * 50)

    output = {
        "summary": {
            "n": n,
            "correctness": avg_correctness,
            "faithfulness": avg_faithfulness,
            "source_hit_rate": avg_source_hit,
        },
        "results": results
    }

    with open("data/eval_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Saved to data/eval_results.json")


if __name__ == "__main__":
    evaluate()
