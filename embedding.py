import pandas as pd

from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer

def embed_text_chunks(pages_and_texts : list[dict]) -> list[list]:
    embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device="cpu")

    # Create embeddings one by one on the GPU
    #for item in tqdm(pages_and_texts):
    #    item["embedding"] = embedding_model.encode(item["sentence_chunk"])

    text_chunks = [item["sentence_chunk"] for item in pages_and_texts]

    text_chunk_embeddings = embedding_model.encode(text_chunks,
                                               batch_size=32, # you can use different batch sizes here for speed/performance, I found 32 works well for this use case
                                               convert_to_tensor=True)

    return text_chunk_embeddings

def save_embeddings(embeddings : list[dict], save_path : str):
    text_chunks_and_embeddings_df = pd.DataFrame(embeddings)
    text_chunks_and_embeddings_df.to_csv(save_path, index=False)
