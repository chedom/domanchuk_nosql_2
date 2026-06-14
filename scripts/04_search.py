# scripts/04_search.py
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from datetime import datetime
from sentence_transformers import SentenceTransformer

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")  # для отримання повного abstract

# 2. Реалізувати функцію кодування запиту в ембеддинг.
def embed_query(query: str) -> list:
    return model.encode(query, normalize_embeddings=True).tolist()

def print_search_results(results):
    print("Search results:")
   
    if len(results['matches']) == 0:
        print("No matches found.")
        print("\n" + "="*50 + "\n")
        return

    for i, result in enumerate(results['matches']):
        metadata = result['metadata']
        print(f"{i+1}. {metadata['title']} ({metadata['year']}) - {metadata['abstract'][:150]}...")
        
    print("\n" + "="*50 + "\n")

def metrics_comparison():
    # 5. Порівняти різні метрики схожості на локальних ембеддингах:
    #  - cosine (косинусна схожість)
    #  - euclidean (евклідова відстань)
    #  - dot_product (скалярний добуток)
    local_embeddings = np.load("embeddings/embeddings.npy")
    metrics = ["cosine", "euclidean", "dot_product"]
    query_vec = embed_query("teaching machines to recognize objects in pictures")
    # Cosine Similarity
    cosine_scores = np.dot(local_embeddings, query_vec) / (np.linalg.norm(local_embeddings, axis=1) * np.linalg.norm(query_vec))
    # Dot Product
    dot_scores = np.dot(local_embeddings, query_vec)
    # L2-distance
    l2_distances = np.linalg.norm(local_embeddings - query_vec, axis=1)
    # Find top-K indices for each metric
    top_cosine_idx = np.argsort(cosine_scores)[::-1][:TOP_K]
    top_dot_idx = np.argsort(dot_scores)[::-1][:TOP_K]
    top_l2_idx = np.argsort(l2_distances)[:TOP_K]

    # Виводимо результати для порівняння за global_idx
    print("\nLocal embedding metrics comparison:")
    print(f"Top-5 Cosine: {top_cosine_idx.tolist()}")
    print(f"Top-5 Dot Product: {top_dot_idx.tolist()}")
    print(f"Top-5 L2 Distance: {top_l2_idx.tolist()}")
    # Top-5 Cosine: [378, 3350, 4115, 610, 3181]
    # Top-5 Dot Product: [378, 3350, 4115, 610, 3181]
    # Top-5 L2 Distance: [378, 3350, 4115, 610, 3181]
        

def main():
    print(f"Using model: {MODEL_NAME}")
    query = "teaching machines to recognize objects in pictures"
    # 3. Виконати чистий семантичний пошук:
    results = index.query(
        vector=embed_query(query),
        top_k=TOP_K,
        include_metadata=True,
    )
    print_search_results(results)
    # 4. Виконати пошук з фільтрацією
    #  приклад A: статті по reinforcement learning за останні 5 років і категорія cs.LG
    print("Filtered search example A: reinforcement learning, last 5 years, category cs.LG")
    query2 = "reinforcement learning"
    results_a = index.query(
        vector=embed_query(query2),
        top_k=TOP_K,
        include_metadata=True,
        filter={
            "category": {"$eq": "cs.LG"},
            "year": {"$gte": datetime.now().year - 5},
        }
    )
    print_search_results(results_a)
    # No matches found.
    # - приклад B: більш старі статті (до 2015 року), будь-яка категорія;
    print("Filtered search example B: older articles (up to 2015), any category")
    results_b = index.query(
        vector=embed_query(query2),
        top_k=TOP_K,
        include_metadata=True,
        filter={
            "year": {"$lte": 2015}
        }
    )
    print_search_results(results_b)
    #     Filtered search example B: older articles (up to 2015), any category
    # Search results:
    # 1. Multi-Agent Modeling Using Intelligent Agents in the Game of Lerpa (2007.0) - Game theory has many limitations implicit in its application. By utilizing
    # multiagent modeling, it is possible to solve a number of problems that are
    # ...
    # 2. Introduction to Phase Transitions in Random Optimization Problems (2007.0) - Notes of the lectures delivered in Les Houches during the Summer School on
    # Complex Systems (July 2006)....
    # 3. Architecture for Pseudo Acausal Evolvable Embedded Systems (2007.0) - Advances in semiconductor technology are contributing to the increasing
    # complexity in the design of embedded systems. Architectures with novel
    # techniq...
    # 4. Why only few are so successful ? (2007.0) - In many professons employees are rewarded according to their relative
    # performance. Corresponding economy can be modeled by taking $N$ independent
    # agen...
    # 5. Opinion Dynamics and Sociophysics (2007.0) - No abstract given. Contents:
    #   I. Definition and Introduction
    #   II. Schelling Model
    #   III. Opinion Dynamics
    #   IV. Languages, Hierarchies and Football
    # ...

    # ==================================================
    # Як бачимо, пошук по прикладу А не повернув жодного результату, на відмінну від прикладу B, це пов'язано з тим, що відпрацювала 
    #  фільтрація по категорії cs.LG, яка є досить вузькою, і за останні 5 років не було знайдено релевантних статей. 
    # У прикладі B, де фільтрація була лише за роком (до 2015), було знайдено кілька релевантних статей, що свідчить про те, 
    #  що в базі даних є старіші статті, які відповідають запиту. Це демонструє важливість вибору правильних фільтрів для отримання р
    #  елевантних результатів у семантичному пошуку.

    # 5. Порівняти різні метрики схожості на локальних ембеддингах:
    metrics_comparison()

if __name__ == "__main__":
    main()