"""Script to evaluate RAG system performance using a ground truth dataset."""
import json
import os
from typing import Dict, List, Tuple

import numpy as np
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer, util

from rag_chain import build_chain


def load_evaluation_dataset(filepath: str = "evaluation_dataset.json") -> Dict:
    """Load the evaluation dataset from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_precision_recall(
    retrieved_docs: List[Document],
    relevant_keywords: List[str],
) -> Tuple[float, float, float]:
    if not retrieved_docs:
        return 0.0, 0.0, 0.0
    
    relevant_retrieved = 0
    for doc in retrieved_docs:
        content_lower = doc.page_content.lower()

        if any(keyword.lower() in content_lower for keyword in relevant_keywords):
            relevant_retrieved += 1
    

    precision = relevant_retrieved / len(retrieved_docs) if retrieved_docs else 0.0

    keywords_found = 0
    for keyword in relevant_keywords:
        keyword_lower = keyword.lower()
        if any(keyword_lower in doc.page_content.lower() for doc in retrieved_docs):
            keywords_found += 1
    
    recall = keywords_found / len(relevant_keywords) if relevant_keywords else 0.0
    
    # F1 Score
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    
    return precision, recall, f1


def calculate_answer_similarity(
    generated_answer: str,
    ground_truth_answer: str,
    model: SentenceTransformer,
) -> float:
    """
    Calculate semantic similarity between generated and ground truth answers.
    Uses cosine similarity of sentence embeddings.
    """
    if not generated_answer or not ground_truth_answer:
        return 0.0
    
    # Generate embeddings
    embedding1 = model.encode(generated_answer, convert_to_tensor=True)
    embedding2 = model.encode(ground_truth_answer, convert_to_tensor=True)
    
    # Calculate cosine similarity
    similarity = util.cos_sim(embedding1, embedding2).item()
    
    return similarity


def calculate_mrr(
    retrieved_docs: List[Document],
    relevant_keywords: List[str],
) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    Returns the reciprocal of the rank of the first relevant document.
    """
    if not retrieved_docs or not relevant_keywords:
        return 0.0
    
    for rank, doc in enumerate(retrieved_docs, start=1):
        content_lower = doc.page_content.lower()
        if any(keyword.lower() in content_lower for keyword in relevant_keywords):
            return 1.0 / rank
    
    return 0.0


def check_sources_match(
    retrieved_docs: List[Document],
    expected_sources: List[str],
) -> bool:
    """Check if retrieved docs come from expected sources."""
    if not retrieved_docs:
        return False
    
    retrieved_sources = {doc.metadata.get("source", "") for doc in retrieved_docs}
    return any(source in retrieved_sources for source in expected_sources)


def evaluate_single_case(
    test_case: Dict,
    qa_chain,
    similarity_model: SentenceTransformer,
) -> Dict:
    """Evaluate a single test case and return metrics."""
    question = test_case["question"]
    ground_truth_answer = test_case["ground_truth_answer"]
    relevant_keywords = test_case.get("relevant_keywords", [])
    expected_sources = test_case.get("relevant_sources", [])
    
    # Run RAG
    result = qa_chain.invoke({"query": question})
    generated_answer = result.get("result", "")
    retrieved_docs = result.get("source_documents", [])
    
    # Calculate metrics
    precision, recall, f1 = calculate_precision_recall(retrieved_docs, relevant_keywords)
    answer_similarity = calculate_answer_similarity(
        generated_answer,
        ground_truth_answer,
        similarity_model,
    )
    mrr = calculate_mrr(retrieved_docs, relevant_keywords)
    sources_correct = check_sources_match(retrieved_docs, expected_sources)
    
    return {
        "question": question,
        "generated_answer": generated_answer,
        "ground_truth_answer": ground_truth_answer,
        "num_retrieved_docs": len(retrieved_docs),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "answer_similarity": answer_similarity,
        "mrr": mrr,
        "sources_correct": sources_correct,
    }


def print_summary_statistics(results: List[Dict]) -> None:
    """Print summary statistics of all evaluation results."""
    if not results:
        print("No results to summarize.")
        return
    
    metrics = ["precision", "recall", "f1_score", "answer_similarity", "mrr"]
    
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total test cases: {len(results)}")
    print(f"Sources correct: {sum(r['sources_correct'] for r in results)}/{len(results)}")
    print()
    
    for metric in metrics:
        values = [r[metric] for r in results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        
        print(f"{metric.upper()}")
        print(f"  Mean: {mean_val:.4f}")
        print(f"  Std:  {std_val:.4f}")
        print(f"  Min:  {min_val:.4f}")
        print(f"  Max:  {max_val:.4f}")
        print()


def save_detailed_results(results: List[Dict], output_file: str = "evaluation_results.json") -> None:
    """Save detailed evaluation results to JSON file."""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Detailed results saved to: {output_file}")


def main() -> None:
    """Main evaluation pipeline."""
    print("Loading evaluation dataset...")
    dataset = load_evaluation_dataset()
    test_cases = dataset.get("test_cases", [])
    
    if not test_cases:
        print("No test cases found in dataset.")
        return
    
    print(f"Found {len(test_cases)} test cases.")
    print("\nBuilding RAG chain...")
    qa_chain = build_chain()
    
    print("Loading similarity model...")
    similarity_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    print("\nEvaluating test cases...\n")
    results = []
    
    for i, test_case in enumerate(test_cases, start=1):
        print(f"[{i}/{len(test_cases)}] Evaluating: {test_case['question'][:60]}...")
        
        result = evaluate_single_case(test_case, qa_chain, similarity_model)
        results.append(result)
        
        # Print individual result
        print(f"  Precision: {result['precision']:.3f} | "
              f"Recall: {result['recall']:.3f} | "
              f"F1: {result['f1_score']:.3f} | "
              f"Similarity: {result['answer_similarity']:.3f}")
        print()
    
    # Print summary
    print_summary_statistics(results)
    
    # Save detailed results
    save_detailed_results(results)


if __name__ == "__main__":
    main()