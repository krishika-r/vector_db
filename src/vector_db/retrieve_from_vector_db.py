import numpy as np
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

class VectorSearch:
    """
    VectorSearch class provides methods to retreive data from the db
    """
    def __init__(self, data_config, model_config, user_config, embeddings,logger):
        # user config
        self.user_config = user_config
        # data config
        self.data_config = data_config
        # model config
        self.model_config = model_config
        self.embeddings = embeddings
        self.logger = logger
        #collection initialization
        self.collection = Collection(model_config['milvus_params']['collection']['name'])
        self.collection.load()
        self.logger.info(f"Collection: {model_config['milvus_params']['collection']['name']} loaded")
        # search vector params
        search_params = self.model_config['search_vector_params']
        self.anns_field = search_params["anns_field"]
        self.limit = search_params["limit"]
        self.expr = search_params["expr"]
        self.output_fields = search_params["output_fields"]
        self.consistency_level =search_params["consistency_level"]

    def perform_search(self, search_query, anns_field, search_params, limit, expr, output_fields, consistency_level):
        """
        Perform a vector-based search using the provided search query.

        Parameters:
        - search_query (str): The query used for searching in the vector database.
        - anns_field (str): The field used for annotations in the vector database.
        - search_params (str): Additional parameters for the search operation.
        - limit (int): The maximum number of results to retrieve.
        - expr (str): An expression to further filter the search results.
        - output_fields (str): The names of the fields to retrieve from the search results.
        - consistency_level (str): The consistency level for the search operation.
        - embedding_type (str): Type of embedding to use.
        - embedding_model (str): embedding model name.

        Returns:
        - search_output: The retrieved data from the vector database.

        """

        search_vector=self.search_query_embedding(search_query)
        self.logger.info(f"Generated the embeddings of the search query {search_query}")
        # Update vector search params
        self.anns_field = anns_field
        self.search_params = search_params
        self.limit = limit
        self.expr = expr
        self.output_fields = output_fields
        self.consistency_level = consistency_level

        search_output = self.collection.search(
            data=[search_vector],
            anns_field=self.anns_field,
            param=self.search_params,
            limit=self.limit,
            expr=self.expr,
            output_fields=self.output_fields,
            consistency_level=self.consistency_level
        )
        self.logger.debug(f"Search result:\n {search_output}")
        return search_output

    def search_query_embedding(self,query):

        """
        Embed the search query using pre-trained embeddings.

        Parameters:
        - query (str): The search query to be embedded.

        Returns:
        - search_vector: The embedded representation of the search query.

        """
        search_vector = self.embeddings.embed_documents([query])
        search_vector = np.array(search_vector[0]).reshape(-1,)

        return search_vector