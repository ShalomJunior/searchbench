import os
import sys
import json
import argparse
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag import RAGGenerator

def main():
    parser = argparse.ArgumentParser(description="Run RAG Generation")
    parser.add_argument("--dataset", type=str, default="scifact")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="HuggingFace model name")
    parser.add_argument("--quantize", action="store_true", help="Enable 4-bit quantization (requires bitsandbytes)")
    args = parser.parse_args()

    print(f"=== Running RAG Generation ({args.dataset.upper()}) ===")
    
    # 1. Load the retrieval cache
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_path = os.path.join(project_root, "data", f"rag_retrieval_cache_{args.dataset}.json")
    
    if not os.path.exists(cache_path):
        print(f"Error: Retrieval cache not found at {cache_path}")
        print("Please run generate_rag_cache.py first.")
        sys.exit(1)
        
    with open(cache_path, "r", encoding="utf-8") as f:
        cache_data = json.load(f)
        
    print(f"Loaded {len(cache_data)} queries from cache.")
    
    # 2. Initialize RAG Generator
    rag = RAGGenerator(model_name=args.model, quantize=args.quantize)
    
    # 3. Generate answers
    results = []
    
    print("\nGenerating answers (this may take a while depending on your hardware)...")
    start_time = time.perf_counter()
    
    for i, item in enumerate(cache_data):
        q_id = item["query_id"]
        query = item["query"]
        documents = item["documents"]
        
        answer = rag.generate(query, documents)
        
        results.append({
            "query_id": q_id,
            "query": query,
            "answer": answer,
            "retrieved_docs": [doc["doc_id"] for doc in documents]
        })
        
        print(f"[{i+1}/{len(cache_data)}] Query: {query[:50]}... -> Done")
        
    total_time = time.perf_counter() - start_time
    print(f"\nGeneration completed in {total_time:.2f} seconds ({total_time/len(cache_data):.2f}s per query).")
    
    # 4. Save results
    out_dir = os.path.join(project_root, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"rag_answers_{args.dataset}.json")
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"Saved {len(results)} answers to {out_path}")

if __name__ == "__main__":
    main()
