import json
import os

from elasticsearch import Elasticsearch
from langchain_elasticsearch import ElasticsearchRetriever
# from langchain.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer

es_url = "http://localhost:9200"
es_client = Elasticsearch(es_url)
model = SentenceTransformer('bert-base-german-dbmdz-uncased')

base_dir = os.path.dirname(os.path.abspath(__file__))  # This gets the directory of the current file
embedding_file_path = os.path.join(base_dir, '../data/document_embeddings.json')


def hybrid_query(search_query, **kwargs) -> dict:
    size = kwargs.get("size", 1)

    """
    This function generate a vector representation for the given search query needed for hybrid search.
    :param search_query: Search query
    :type search_query: str
    :return: A dictionary representing the hybrid search query combining both traditional
        search parameters and vector search configuration for passing to Elasticsearch.
    :rtype: dict
    """

    vector = model.encode(search_query)
    return {"query": {"bool": {"must": {
        "multi_match": {"query": search_query, "fields": ["text", "section"], "type": "best_fields", "boost": 0.5, }}}},
        "knn": {"field": "question_text_vector", "query_vector": vector, "k": 5, "num_candidates": 100, "boost": 0.5},
        "size": size, }


def init_elastic_search():
    """
    Initialize Elasticsearch
    :return: elasticsearch retriever
    """

    index_settings = {"settings": {"number_of_shards": 1, "number_of_replicas": 0}, "mappings": {
        "properties": {"section": {"type": "text"}, "text": {"type": "text"},
                       "section_vector": {"type": "dense_vector", "dims": 768, "index": True, "similarity": "cosine"},
                       "text_vector": {"type": "dense_vector", "dims": 768, "index": True, "similarity": "cosine"}}}}

    index_name = "hybrid_search"

    es_client.indices.delete(index=index_name, ignore_unavailable=True)
    # es_client.indices.create(index=index_name, body=index_settings)
    es_client.indices.create(index=index_name, settings=index_settings['settings'], mappings=index_settings['mappings'])

    with open(embedding_file_path, 'r', encoding="utf-8") as f:
        documents = json.load(f)

    for doc in documents:
        es_client.index(index=index_name, document=doc)

    hybrid_retriever = ElasticsearchRetriever.from_es_params(index_name=index_name, body_func=hybrid_query,
                                                             content_field='text', url=es_url)

    return hybrid_retriever


def hybrid_search(hybrid_retriever, search_query, size):
    """
    Do hybrid search

    :param hybrid_retriever: elasticsearch retriever
    :param search_query: search query
    :param size: number of results to return
    :return:
    """
    hybrid_results = hybrid_retriever.invoke(input=search_query, config=None, size=size)
    return hybrid_results


if __name__ == "__main__":
    search_query = (
        "Welche sind die wichtigsten physischen und psychischen Gesundheitsrisiken, denen Landwirte aufgrund "
        "ihrer Arbeitsbedingungen ausgesetzt sind, und wie geht die Arbeitsmedizin mit diesen Herausforderungen um?")
    retriever = init_elastic_search()
    results = hybrid_search(retriever, search_query, size=3)
    print(results)
