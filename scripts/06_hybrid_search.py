# scripts/06_hybrid_search.py
import os
import math
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from typing import List, Tuple

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 10   # беремо ширше, щоб RRF міг переранжувати
K_RRF = 60

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet").reset_index(drop=True)

# From lesson 7
# Реалізувати Reciprocal Rank Fusion (RRF) 
def reciprocal_rank_fusion(
    rankings: List[List[int]],
    k: int = K_RRF
) -> List[Tuple[int, float]]:
    """
    Об'єднує кілька ранжованих списків через RRF.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# 4. Реалізувати функції пошуку: BM25
def bm25_search(bm25_idx, query: str, top_k: int = TOP_K) -> List[int]:
    tokenized_query = query.lower().split()
    bm25_scores = bm25_idx.get_scores(tokenized_query)
    bm25_ranking = np.argsort(bm25_scores)[::-1][:top_k]

    return bm25_ranking

def embed_query(query: str) -> list:
    return model.encode(query, normalize_embeddings=True).tolist()

def convert_to_global_idx(doc_id: str) -> int:
    """
    Перетворює doc_id у форматі "paper_<номер>" на глобальний індекс.
    """
    return int(doc_id.split("_")[1])
# 4. Реалізувати функції пошуку: векторний (Pinecone)
def vector_search(vector_idx, query: str, top_k: int = TOP_K) -> List[str]:
    query_vec = embed_query(query)
    results = vector_idx.query(
        vector=query_vec,
        top_k=top_k,
        include_metadata=False
    )

    return [convert_to_global_idx(match['id']) for match in results['matches']]
# 4. гібридний (BM25 + векторний через RRF).
def hybrid_search(
    query: str,
    bm25_idx: BM25Okapi,
    vector_idx: Pinecone,
    vector_search,
    top_k: int = TOP_K,
) -> List[Tuple[int, float]]:
    # --- Векторний пошук ---
    vector_ranking = vector_search(vector_idx, query, top_k=top_k)
    # --- BM25 пошук ---
    bm25_ranking = bm25_search(bm25_idx, query, top_k=top_k)
    # --- RRF об'єднання ---
    fused = reciprocal_rank_fusion([vector_ranking, bm25_ranking])

    return [(doc_id, score) for doc_id, score in fused[:top_k]]

def print_search_results(results, df: pd.DataFrame):
    if len(results) == 0:
        print("No matches found.")
        print("\n" + "="*50 + "\n")
        return

    for doc_id in results:
        title = df.loc[doc_id, "title"]
        # abstract = df.loc[doc_id, "abstract"]
        print(f" - {title} (ID: {doc_id})")
        # print(f"   Abstract: {abstract[:200]}...")  # Displaying first 200 characters of abstract
        
    print("\n" + "="*50 + "\n")

def print_search_sresults_with_scores(results, df: pd.DataFrame):
    if len(results) == 0:
        print("No matches found.")
        print("\n" + "="*50 + "\n")
        return

    for doc_id, score in results:
        title = df.loc[doc_id, "title"]
        print(f" - {title} (ID: {doc_id}, RRF Score: {score:.4f})")
        
    print("\n" + "="*50 + "\n")

def multiple_search(query: str, bm25_idx, vector_idx, top_k: int = TOP_K):
    print(f"Query: {query}")
    bm25_results = bm25_search(bm25_idx, query, top_k=top_k)
    print("\nBM25 Search Results:")
    print_search_results(bm25_results[:5], df)
    vector_results = vector_search(vector_idx, query, top_k=top_k)
    print("\nVector Search Results:")
    print_search_results(vector_results[:5], df)
    hybrid_results = hybrid_search(query, bm25_idx, vector_idx, vector_search, top_k=top_k)
    print("\nHybrid Search Results:")
    print_search_sresults_with_scores(hybrid_results[:5], df)

def main():
    # 1. Побудувати локальний BM25-індекс за заголовками і анотаціями всіх статей.
    tokenized_corpus = (df["title"] + " " + df["abstract"]).str.lower().str.split().tolist()
    bm25 = BM25Okapi(tokenized_corpus)

    # 5. Для демонстрації виконати три запити:
    queries = [
        "BERT fine-tuning",
        "Yann LeCun convolutional networks",
        "making computers understand human emotions from text"
    ]

    for q in queries:
        # 6. Вивести результати для кожного методу і порівняти:
        multiple_search(q, bm25, index, top_k=TOP_K)


if __name__ == "__main__":
    main()
