import pandas as pd

from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer

def embed_text_chunks(pages_and_chunks : list[dict]) -> list[dict]:
    embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device="cpu")

    # Create embeddings one by one on the GPU
    for item in tqdm(pages_and_chunks):
        item["embedding"] = embedding_model.encode(item["sentence_chunk"])

    return pages_and_chunks

def save_embeddings(embeddings : list[dict], save_path : str):
    text_chunks_and_embeddings_df = pd.DataFrame(embeddings)
    text_chunks_and_embeddings_df.to_csv(save_path, index=False)
