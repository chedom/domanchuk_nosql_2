# scripts/05_chunking.py
import os
import re
from time import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")

INDEX_FIXED = "arxiv-chunks-fixed"
INDEX_SEMANTIC = "arxiv-chunks-semantic"
BATCH_SIZE = 200   # Pinecone рекомендує батчі до 200 векторів


def create_index(index_name: str):
    # Створюємо індекс (якщо не існує)
    if index_name not in pc.list_indexes().names():
        print(f"Creating index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=VECTOR_DIM,        # повинна збігатися з розмірністю моделі
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1", # single available region for my free
            ),
        )

        # Чекаємо ініціалізації індексу
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)

        print(f"Index '{index_name}' created and ready to use.")

        return pc.Index(index_name)
    else:
        print(f"Index '{index_name}' already exists.")  
        return pc.Index(index_name)

def find_largest_abstractions(df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """
    Знаходимо топ-N статей із найбільшими абстрактами.
    """
    df["abstract_length"] = df["abstract"].apply(len)
    return df.nlargest(top_n, "abstract_length").copy()
# Function from lecture 7
def fixed_size_chunking(text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\\n\\n", "\\n", ". ", " ", ""],
    )

    return splitter.split_text(text.strip())


def semantic_chunking(
    text: str,
    model: SentenceTransformer,
    threshold: float = 0.7,
    min_chunk_size: int = 50,
    max_chunk_size: int = 500,
) -> List[str]:
    """
    Ділить текст на семантично зв'язні блоки.
    Новий chunk починається, коли косинусна схожість
    між сусідніми реченнями падає нижче threshold.
    """
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    if not sentences:
        return []
    if len(sentences) < 2:
        return [sentences[0] + "."]

    embeddings = model.encode(sentences, normalize_embeddings=True)

    similarities = [
        float(np.dot(embeddings[i], embeddings[i + 1]))
        for i in range(len(embeddings) - 1)
    ]

    chunks, current_chunk = [], [sentences[0]]
    for i, sim in enumerate(similarities):
        next_sentence = sentences[i + 1]
        potential_chunk_text = ". ".join(current_chunk + [next_sentence]) + "."
        
        by_threshold = (sim < threshold and len(". ".join(current_chunk) + ".") >= min_chunk_size)
        by_max_size = (len(potential_chunk_text) > max_chunk_size)
        
        if by_threshold or by_max_size:
            chunks.append(". ".join(current_chunk) + ".")
            current_chunk = [next_sentence]
        else:
            current_chunk.append(next_sentence)

    if current_chunk:
        chunks.append(". ".join(current_chunk) + ".")

    return chunks

def prepare_point_objects(top_30_data_list, chunking_strategy_name: str):
    all_vectors_to_upsert = []
    
    for row in top_30_data_list:
        arxiv_id = row["id"]
        title = row["title"]
        abstract = row["abstract"]
        year_val = row["year"]
        category = row["category"]

        # select chunking strategy
        if chunking_strategy_name == "fixed":
            chunks = fixed_size_chunking(abstract, chunk_size=200, chunk_overlap=50)
        else:
            chunks = semantic_chunking(abstract, model=model, threshold=0.7, min_chunk_size=50, max_chunk_size=200)

        for chunk_num, chunk_text in enumerate(chunks):
            chunk_id = f"{chunking_strategy_name}_chunk_{arxiv_id}_{chunk_num}"            
            full_input_text = f"{title} [SEP] {chunk_text}"
            embedding = model.encode(full_input_text, normalize_embeddings=True).tolist()
            
            metadata = {
                "arxiv_id": arxiv_id,
                "title": title,
                "text": chunk_text,
                "chunk_num": chunk_num,
                "year": year_val,
                "category": category
            }
            
            all_vectors_to_upsert.append((chunk_id, embedding, metadata))

    return all_vectors_to_upsert

def upsert_chunks_to_pinecone(index, vectors, index_name: str, top_k: int = 5):
    for i in tqdm(range(0, len(vectors), BATCH_SIZE), desc=f"Upserting chunks to {index_name}"):
        batch = vectors[i:i + BATCH_SIZE]
        index.upsert(vectors=batch)

def search_chunks(query_text, idx_fixed_client, idx_semantic_client, top_k: int = 5):  
    query_vector = model.encode(query_text, normalize_embeddings=True).tolist()
    
    # Fixed-size chunk searchs
    res_fixed = idx_fixed_client.query(vector=query_vector, top_k=top_k, include_metadata=True)
    print("\n Fixed-size chunk search results:")
    for m in res_fixed.get("matches", []):
        meta = m["metadata"]
        print(f"Score: {m['score']:.4f} | Title: {meta['title'][:60]}... (Chunk {meta['chunk_num']})")
        print(f"Chunk text: {meta['text'][:150]}...\n")

    # Semantic chunk search
    res_semantic = idx_semantic_client.query(vector=query_vector, top_k=top_k, include_metadata=True)
    print("\n Semantic chunk search results:")
    for m in res_semantic.get("matches", []):
        meta = m["metadata"]
        print(f"Score: {m['score']:.4f} | Title: {meta['title'][:60]}... (Chunk {meta['chunk_num']})")
        print(f"Chunk text: {meta['text'][:150]}...\n")

def main():
    # 1. Вибрати 30 статей із найдовшими анотаціями.
    top_30_abstracts = find_largest_abstractions(df, top_n=30).to_dict(orient="records")
    print(top_30_abstracts)
    # 3. Створити окремі індекси в Pinecone для кожного типу чанків (arxiv-chunks-fixed і arxiv-chunks-semantic).
    ixed_client = create_index(INDEX_FIXED)
    semantic_client = create_index(INDEX_SEMANTIC)
    # 4. Для кожного чанка створити ембеддинг та сформувати об’єкти
    fixed_vectors = prepare_point_objects(top_30_abstracts, chunking_strategy_name="fixed")
    semantic_vectors = prepare_point_objects(top_30_abstracts, chunking_strategy_name="semantic")
    # 5. Завантажувати чанки в Pinecone батчами і відображати прогрес.
    upsert_chunks_to_pinecone(ixed_client, fixed_vectors, index_name=INDEX_FIXED)
    upsert_chunks_to_pinecone(semantic_client, semantic_vectors, index_name=INDEX_SEMANTIC)
    # 6. Реалізувати функцію пошуку по чанках:
    queries = [
        "methods for improving machine learning model performance",
        "advances in natural language processing techniques",
        "applications of deep learning in computer vision"
    ]
    for query in queries:
        print(f"\nSearching for query: '{query}'")
        search_chunks(query, ixed_client, semantic_client)

if __name__ == "__main__":
    main()  