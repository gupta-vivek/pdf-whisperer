import lancedb
from lancedb.rerankers import CrossEncoderReranker


def create_lance_db(uri, table_name, schema, data):
    """
    Creates Lance DB
    :param uri: URI
    :param table_name: Table name
    :param schema: Schema
    :param data: Data
    :return:
    """
    db = lancedb.connect(uri)
    table = db.create_table(table_name, schema=schema, exist_ok=True)
    table.add(data)
    table.create_fts_index("text")


def init_lance_db(uri, table_name):
    """
    Initializes Lance DB
    :param uri: URI
    :param table_name: Table name
    :return: lance db table
    """
    db = lancedb.connect(uri)
    table = db.open_table(table_name)

    return table


def get_reranker():
    """
    Gets re-ranker
    :return: reranker
    """
    return CrossEncoderReranker("cross-encoder/msmarco-MiniLM-L12-en-de-v1")


def hybrid_search_lance(table, query, reranker, limit=1):
    """
    Hybrid search with Lance DB
    :param table: table
    :param query: search query
    :param reranker: reranker
    :param limit: number of results
    :return:
    """
    return table.search(query, query_type="hybrid").rerank(reranker).limit(limit).to_pandas()['text'].to_list()


if __name__ == "__main__":
    URI = "../data/lance_db"
    TABLE_NAME = "documents"
    # Create Lance DB
    # # Load documents
    # with open("../data/document_chunks.json", "r") as f:
    #     documents = json.load(f)
    #
    # model = get_registry().get("sentence-transformers").create(name="bert-base-german-dbmdz-uncased", device="cpu")
    # # Schema
    # class Words(LanceModel):
    #     text: str = model.SourceField()
    #     vector: Vector(model.ndims()) = model.VectorField()
    #
    # # Create LanceDB
    # data = [{'text': doc['text']} for doc in documents] # Modified for lancedb format
    # create_lance_db(URI, TABLE_NAME, Words, data)

    db = lancedb.connect(URI)
    table = db.open_table(TABLE_NAME)
    query = "Welche sind die wichtigsten physischen und psychischen Gesundheitsrisiken, denen Landwirte aufgrund ihrer Arbeitsbedingungen ausgesetzt sind, und wie geht die Arbeitsmedizin mit diesen Herausforderungen um?"
    reranker = get_reranker()
    results = table.search(query, query_type="hybrid").rerank(reranker).limit(1).to_pandas()['text'].to_list()
