"""
Coverage test — checks how the system handles questions it cannot answer.

Tests two failure modes:
  1. Out-of-scope  : question has nothing to do with UTS (should refuse)
  2. In-scope miss : question is about UTS but not in crawled pages (should refuse or say uncertain)

Usage:
    python src/coverage_test.py
"""
import json
import sys
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

sys.path.insert(0, "src")
from rag import load_config, load_vectorstore, build_rag_chain


# ── Test questions ─────────────────────────────────────────────────────────────

OUT_OF_SCOPE = [
    "What is the meaning of life?",
    "Who won the FIFA World Cup in 2022?",
    "How do I cook pasta?",
    "What is the stock price of Apple?",
    "Tell me a joke.",
]

IN_SCOPE_MISS = [
    "Who is the current UTS Vice Chancellor?",         # likely not in /for-students
    "What is the UTS annual budget?",                  # not student-facing
    "How many staff does UTS employ?",                 # not student-facing
    "What research grants did UTS receive this year?", # not student-facing
    "What is the UTS endowment fund size?",            # not student-facing
]


# ── Judge prompt ───────────────────────────────────────────────────────────────

JUDGE_PROMPT = PromptTemplate.from_template(
    """You are evaluating a student support chatbot. Determine whether the answer appropriately refuses or admits it cannot help, versus hallucinating or making up an answer.

Question: {question}
Answer: {answer}

Does the chatbot correctly refuse or express uncertainty (rather than hallucinating)?
Reply with one of:
  CORRECT_REFUSAL   - the chatbot said it couldn't find the answer or redirected appropriately
  HALLUCINATED      - the chatbot gave a confident answer that appears fabricated
  PARTIAL           - the chatbot gave a vague or hedged answer but didn't fully refuse

Reply with one word only."""
)


def run_coverage_test(output_path="data/coverage_results.json"):
    config = load_config()
    vectorstore = load_vectorstore(config)
    chain, _ = build_rag_chain(vectorstore, config)

    judge = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    judge_chain = JUDGE_PROMPT | judge | StrOutputParser()

    results = {"out_of_scope": [], "in_scope_miss": []}

    for category, questions in [("out_of_scope", OUT_OF_SCOPE), ("in_scope_miss", IN_SCOPE_MISS)]:
        print(f"\n{'='*55}")
        print(f"Category: {category.replace('_', ' ').upper()}")
        print("="*55)

        for q in questions:
            print(f"\nQ: {q}")
            answer, docs = chain({"question": q, "chat_history": []})
            verdict = judge_chain.invoke({"question": q, "answer": answer}).strip()

            print(f"A: {answer[:200]}{'...' if len(answer) > 200 else ''}")
            print(f"Verdict: {verdict}")

            results[category].append({
                "question": q,
                "answer": answer,
                "verdict": verdict,
                "sources": [doc.metadata["url"] for doc in docs],
            })

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("COVERAGE TEST SUMMARY")
    print("="*55)

    for category, items in results.items():
        total = len(items)
        correct = sum(1 for r in items if r["verdict"] == "CORRECT_REFUSAL")
        partial = sum(1 for r in items if r["verdict"] == "PARTIAL")
        hallucinated = sum(1 for r in items if r["verdict"] == "HALLUCINATED")
        print(f"\n{category.replace('_', ' ').title()} ({total} questions):")
        print(f"  Correct refusals : {correct}/{total}")
        print(f"  Partial          : {partial}/{total}")
        print(f"  Hallucinated     : {hallucinated}/{total}  ← bad")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    run_coverage_test()
