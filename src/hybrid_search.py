import json

from elasticsearch import Elasticsearch
from langchain_elasticsearch import ElasticsearchRetriever
# from langchain.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer

es_url = "http://localhost:9200"
es_client = Elasticsearch(es_url)


def hybrid_query(search_query: str) -> dict:
    """
    This function generate a vector representation for the given search query needed for hybrid search.
    :param search_query: Search query
    :type search_query: str
    :return: A dictionary representing the hybrid search query combining both traditional
        search parameters and vector search configuration for passing to Elasticsearch.
    :rtype: dict
    """
    model = SentenceTransformer('bert-base-german-dbmdz-uncased')
    vector = model.encode(search_query)
    return {
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": search_query,
                        "fields": ["text", "section"],
                        "type": "best_fields",
                        "boost": 0.5,
                    }
                }
            }
        },
        "knn": {
            "field": "question_text_vector",
            "query_vector": vector,
            "k": 5,
            "num_candidates": 100,
            "boost": 0.5
        },
        "size": 5,
    }


def hybrid_search(search_query: str, size: int = 1):
    """
    Executes a hybrid search using Elasticsearch with a combination of dense vectors and
    text-based similarity.

    :param search_query: The search query as a string used to retrieve matching documents.
    :type search_query: str
    :param size: Number of results to return. Defaults to 1.
    :return: List of search results retrieved based on the given query.
    :rtype: list
    """

    index_settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "section": {"type": "text"},
                "text": {"type": "text"},
                "section_vector": {"type": "dense_vector", "dims": 768, "index": True, "similarity": "cosine"},
                "text_vector": {"type": "dense_vector", "dims": 768, "index": True, "similarity": "cosine"}
            }
        }
    }

    index_name = "hybrid_search"

    es_client.indices.delete(index=index_name, ignore_unavailable=True)
    es_client.indices.create(index=index_name, body=index_settings)

    with open('../data/document_embeddings.json', 'r', encoding="utf-8") as f:
        documents = json.load(f)

    for doc in documents:
        es_client.index(index=index_name, document=doc)

    hybrid_retriever = ElasticsearchRetriever.from_es_params(
        index_name=index_name,
        body_func=hybrid_query,
        content_field='text',
        url=es_url
    )

    hybrid_results = hybrid_retriever.invoke(search_query, size=size)

    return hybrid_results


if __name__ == "__main__":
    search_query = (
        "Welche sind die wichtigsten physischen und psychischen Gesundheitsrisiken, denen Landwirte aufgrund "
        "ihrer Arbeitsbedingungen ausgesetzt sind, und wie geht die Arbeitsmedizin mit diesen Herausforderungen um?")
    result = hybrid_search(search_query)
    print(result)
