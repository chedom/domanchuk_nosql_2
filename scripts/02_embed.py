import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def main():
    # 1. Завантажити датасет із файлу data/arxiv_subset.parquet з використанням бібліотеки pandas.
    df = pd.read_parquet("data/arxiv_subset.parquet")
    # 2. Підготувати тексти для кодування:
    # для кожного запису об’єднати поля title і abstract в один рядок у форматі:title + " [SEP] " + abstract
    texts = (df["title"] + " [SEP] " + df["abstract"]).tolist()
    # 3. Згенерувати ембеддинги текстів за допомогою моделі allenai/specter2_base з бібліотеки sentence-transformers.
    model = SentenceTransformer("allenai/specter2_base")
    # 4. Закодувати всі тексти в ембеддинги з урахуванням таких вимог:
    embeddings = model.encode(
        sentences=texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    # 5. Вивести в консоль: 
    # загальну кількість оброблених текстів;
    print(f"Загальна кількість оброблених текстів: {len(texts)}")
    # розмірність ембеддингів (очікується 768);
    print(f"Розмірність ембеддингів: {embeddings.shape[1]}")
    # норму першого ембеддингу (повинна бути близька до 1.0).
    print(f"Норма першого ембеддингу: {np.linalg.norm(embeddings[0])}")
    # 6. Зберегти отримані ембеддинги у файл embeddings/embeddings.npy у форматі NumPy.
    os.makedirs("embeddings", exist_ok=True)
    np.save("embeddings/embeddings.npy", embeddings)


if __name__ == "__main__":
    main()
