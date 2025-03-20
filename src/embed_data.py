import json

from sentence_transformers import SentenceTransformer


def create_embeddings(documents):
    """
    Creates embeddings for a list of documents using a pre-trained SentenceTransformer model.

    :param documents: List of sections and their text.
    :return: list along with the embeddings of text.
    """
    model = SentenceTransformer('bert-base-german-dbmdz-uncased')
    document_embeddings = []
    for i, doc in enumerate(documents):
        doc["text_vector"] = model.encode(doc["text"]).tolist()
        document_embeddings.append(doc)

    return document_embeddings


if __name__ == "__main__":
    with open('../data/document_chunks.json', 'r', encoding="utf-8") as f:
        documents = json.load(f)

    embeddings = create_embeddings(documents)
    with open('../data/document_embeddings.json', 'w', encoding="utf-8") as f:
        json.dump(embeddings, f)
