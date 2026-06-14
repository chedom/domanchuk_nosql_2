# scripts/03_load_to_pinecone.py
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INPUT_PARQUET = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"
INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 768
BATCH_SIZE = 200   # Pinecone рекомендує батчі до 200 векторів

# Ініціалізація клієнта
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

def create_index():
    # Створюємо індекс (якщо не існує)
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=VECTOR_DIM,        # повинна збігатися з розмірністю моделі
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1", # single available region for my free
            ),
        )

        # Чекаємо ініціалізації індексу
        while not pc.describe_index(INDEX_NAME).status['ready']:
            time.sleep(1)

        print(f"Index '{INDEX_NAME}' created and ready to use.")
    else:
        print(f"Index '{INDEX_NAME}' already exists.")  

def prepare_and_inser_data(index, df: pd.DataFrame, embeddings: np.ndarray):
    # для кожного запису сформувати об’єкт із:
    # унікальним id вигляду "paper_<номер>";
    # ембеддингами;
    # метаданими: arxiv_id, title, abstract (до 500 символів), authors (до 200 символів), year, category.
   for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Uploading to Pinecone"):
        batch_df = df.iloc[i:i+BATCH_SIZE]
        batch_embeddings = embeddings[i:i+BATCH_SIZE]

        vectors = []
        for idx, (global_idx, row) in enumerate(batch_df.iterrows()):
            vector_id = f"paper_{global_idx}"
            metadata = {
                "arxiv_id": row["id"],
                "title": row["title"],
                "abstract": row["abstract"][:500],
                "authors": row["authors"][:200],
                "year": int(row["year"]),
                "category": row["category"]
            }
            vectors.append((vector_id, batch_embeddings[idx].tolist(), metadata))

        # Завантажуємо батч в Pinecone
        index.upsert(vectors=vectors)


# Створюємо індекс (якщо не існує)
def main():
    # 1. Створити індекс arxiv-papers у Pinecone, якщо він ще не існує, і підключитися до нього.
    create_index()
    index = pc.Index(INDEX_NAME)
    # 2. Завантажити дані:
    # прочитати датасет із файлу data/arxiv_subset.parquet;
    # завантажити ембеддинги з файлу embeddings/embeddings.npy
    df = pd.read_parquet(INPUT_PARQUET)
    embeddings = np.load(INPUT_EMBEDDINGS)
    # 3. Підготувати дані для завантаження:
    # 4. Завантажити дані в Pinecone батчами і показувати прогрес.
    prepare_and_inser_data(index, df, embeddings)
    # 5. Після завершення завантаження вивести в консоль загальну кількість векторів в індексі.
    print("Data upload completed.")
    time.sleep(3)  # Allow some time for Pinecone to process the upserted vectors
    index_stats = index.describe_index_stats()
    total_vectors = index_stats['total_vector_count']
    print(f"Total vectors in the index: {total_vectors}")


if __name__ == "__main__":
    main()
