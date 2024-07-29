import os
import requests
import pandas as pd

from text_handler import split_all_text_into_chunks
from embedding import embed_text_chunks, save_embeddings
from tqdm.auto import tqdm

if __name__ == "__main__":
    pdf_path = "poker.pdf"

    if not os.path.exists(pdf_path):
        print("No File to Read :(")
    else:
        pages_and_chunks = split_all_text_into_chunks(pdf_path)

        text_chunks_embeddings = embed_text_chunks(pages_and_chunks)

        embeddings_df_save_path = "text_chunks_embeddings_df.csv"
        save_embeddings(text_chunks_embeddings, embeddings_df_save_path)

        text_chunks_embeddings_df_load = pd.read_csv(embeddings_df_save_path)
        print(text_chunks_embeddings_df_load.head())
