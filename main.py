import os
import requests
import pandas as pd

from text_handler import text_formatter, open_and_read_pdf
from tqdm.auto import tqdm

from spacy.lang.en import English

if __name__ == "__main__":
    pdf_path = "poker.pdf"

    if not os.path.exists(pdf_path):
        print("No File to Read :(")
    else:
        pages_and_texts = open_and_read_pdf(pdf_path)

        df = pd.DataFrame(pages_and_texts)
        print(df.describe())

        # Divide into sentences + chunking
        nlp = English()
        nlp.add_pipe("sentencizer") # <--- For Dividing on Sentences

        for item in tqdm(pages_and_texts):
            item["sentences"] = list(nlp(item["text"]).sents)
            # Make sure all sentences are strings
            # item["sentences"] = [str(sentence) for sentence in item["sentences"]]

            # Count the sentences
            item["page_sentence_length"] = len(item["sentences"])




