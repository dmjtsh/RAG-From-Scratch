import os
import requests
import pandas as pd

from text_handler import *
from tqdm.auto import tqdm

if __name__ == "__main__":
    pdf_path = "poker.pdf"

    if not os.path.exists(pdf_path):
        print("No File to Read :(")
    else:
        pages_and_texts = open_and_read_pdf(pdf_path)

        pages_and_texts = divide_text_into_sentences(pages_and_texts)
        pages_and_texts = split_text_into_chunks(pages_and_texts, 5)
        pages_and_texts = split_chunks_into_item(pages_and_texts)

        df = pd.DataFrame(pages_and_texts)
        print(df.describe())

        print(pages_and_texts[1]["sentence_chunk"])
