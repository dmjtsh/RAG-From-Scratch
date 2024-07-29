import fitz
import re

from tqdm.auto import tqdm
from spacy.lang.en import English

def text_formatter(text: str) -> str:
    cleaned_text = text.replace("\n", " ").strip()

    return cleaned_text

def open_and_read_pdf(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)  # open a document
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc)):  # iterate the document pages
        text = page.get_text()  # get plain text encoded as UTF-8
        text = text_formatter(text)
        pages_and_texts.append({"page_number": page_number,  # adjust page numbers since our PDF starts on page 42
                                "page_char_count": len(text),
                                "page_word_count": len(text.split(" ")),
                                "page_sentence_count_raw": len(text.split(". ")),
                                "page_token_count": len(text) / 4,  # 1 token = ~4 chars
                                "text": text})
    return pages_and_texts

def divide_text_into_sentences(pages_and_texts : list[dict]) -> list[dict]:
        # Divide into sentences
        nlp = English()
        nlp.add_pipe("sentencizer") # <--- For Dividing on Sentences

        for item in tqdm(pages_and_texts):
            item["sentences"] = list(nlp(item["text"]).sents)

            # Make sure all sentences are strings !!!
            item["sentences"] = [str(sentence) for sentence in item["sentences"]]

            # Count the sentences
            item["page_sentence_length"] = len(item["sentences"])

        return pages_and_texts

# Recursively splits a list into desired sizes
def split_list(input_list: list, split_size: int) -> list[list[str]]:
    splitted_list = []

    for i in range(0, len(input_list), split_size):
        splitted_list.append(input_list[i:i + split_size])

    return splitted_list

# Split all sentences on chunks
def split_text_into_chunks(pages_and_texts : list[dict], chunk_size) -> list[dict]:
    # Loop through pages and texts and split sentences into chunks
    for item in tqdm(pages_and_texts):
        item["sentence_chunks"] = split_list(input_list=item["sentences"],
                                            split_size=chunk_size)
        item["num_chunks"] = len(item["sentence_chunks"])

    return pages_and_texts

# Split each chunk into its own item
def split_chunks_into_item(pages_and_texts : list[dict]) -> list[dict]:
    pages_and_chunks = []

    for item in tqdm(pages_and_texts):
        for sentence_chunk in item["sentence_chunks"]:
            chunk_dict = {}
            chunk_dict["page_number"] = item["page_number"]

            # Join the sentences together into a paragraph-like structure, aka a chunk (so they are a single string)
            joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
            joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk) # ".A" -> ". A" for any full-stop/capital letter combo
            chunk_dict["sentence_chunk"] = joined_sentence_chunk

            # Get stats about the chunk
            chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
            chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
            chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4 # 1 token = ~4 characters

            pages_and_chunks.append(chunk_dict)

    return pages_and_chunks

def split_all_text_into_chunks(pdf_path : str) -> list[dict]:
    pages_and_texts = open_and_read_pdf(pdf_path)

    pages_and_texts  = divide_text_into_sentences(pages_and_texts)
    pages_and_texts  = split_text_into_chunks(pages_and_texts, 5)
    pages_and_chunks = split_chunks_into_item(pages_and_texts)

    return pages_and_chunks
