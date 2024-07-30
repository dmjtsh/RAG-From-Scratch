import os

import torch
import numpy as np
import pandas as pd

import textwrap

from text_handler import split_all_text_into_chunks
from embedding import embed_text_chunks, save_embeddings
from tqdm.auto import tqdm

from sentence_transformers import util, SentenceTransformer

def print_wrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)

if __name__ == "__main__":
    pdf_path = "poker.pdf"

    if not os.path.exists(pdf_path):
        print("No File to Read :(")
    else:
        embeddings_df_save_path = "text_chunks_and_embeddings_df.csv"
        '''
        pages_and_chunks = split_all_text_into_chunks(pdf_path)

        text_chunks_and_embeddings = embed_text_chunks(pages_and_chunks)

        text_chunks_and_embeddings_df = pd.DataFrame(text_chunks_and_embeddings)

        save_embeddings(text_chunks_and_embeddings_df, embeddings_df_save_path)
        '''

        device = "cpu"

        text_chunks_and_embeddings_df = pd.read_csv(embeddings_df_save_path)

        # Convert embedding column back to np.array (it got converted to string when it got saved to CSV)
        text_chunks_and_embeddings_df["embedding"] = text_chunks_and_embeddings_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
        # Convert texts and embedding df to list of dicts
        pages_and_chunks = text_chunks_and_embeddings_df.to_dict(orient="records")

        # Convert embeddings to torch tensor and send to device (note: NumPy arrays are float64, torch tensors are float32 by default)
        embeddings = torch.tensor(np.array(text_chunks_and_embeddings_df["embedding"].tolist()), dtype=torch.float32).to(device)
        embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2",
                                      device=device) # choose the device to load the model to

        # OUR QUERY

        query = "flush"
        print(f"Query: {query}")

        query_embedding = embedding_model.encode(query, convert_to_tensor=True)

        # search
        dot_scores = util.dot_score(a=query_embedding, b=embeddings)[0]
        top_results_dot_product = torch.topk(dot_scores, k=5)

        most_relevant_answer = pages_and_chunks[top_results_dot_product[1][0]]["sentence_chunk"]

        print("\n", most_relevant_answer)
