"""Chunk data into smaller semantic chunks."""

import json

from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings


def semantic_chunk(document, max_length=600):
    """
    Breaks down documents into smaller semantic chunks if they exceed a specified maximum length.

    :param document: List of sections and their text.
    :param max_length: Length threshold for semantic chunking.
    :return: A list of dictionaries with the section title and the semantic chunked text.
    """
    embed_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")
    semantic_chunker = SemanticChunker(embed_model, breakpoint_threshold_type="percentile")

    semantic_chunks = []
    for doc in document:
        len_text = doc['text'].split()
        if len(len_text) > max_length:
            chunks = semantic_chunker.create_documents([doc['text']])
            for ind, chunk in enumerate(chunks):
                semantic_chunks.append({"section": f"{doc['section']} - {ind + 1}", "text": chunk.page_content})
        else:
            semantic_chunks.append({"section": doc['section'], "text": doc['text']})

    return semantic_chunks


if __name__ == "__main__":
    with open('../data/document.json', 'r') as f:
        document = json.load(f)

    semantic_chunks = semantic_chunk(document)
    with open('../data/document_chunks.json', 'w', encoding='utf-8') as f:
        json.dump(semantic_chunks, f)
