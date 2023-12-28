from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Milvus
import numpy as np
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
from tqdm import tqdm

class EmbeddingExtractor:
    """
    EmbeddingExtractor class provides methods to create embeddings and store them in a db
    """
    def __init__(self, data_config, model_config, user_config,logger):
        # user config
        self.user_config = user_config
        # data config
        self.data_config = data_config
        # model config
        self.model_config = model_config
        self.logger = logger

       # Extract embedding params
        embedding_params = self.model_config["embedding_params"]
        self.embedding_type = embedding_params["embedding_type"]
        self.embedding_model = embedding_params["embedding_model"]
        self.score_factor = embedding_params["score_factor"]
        # Extract open api params
        self.connection_param_dict = user_config["connection_params"]
        self.milvus_params =  self.model_config["milvus_params"]


    def _initialize_embedding(self,embedding_type,embedding_model):
        """
        Initializing different embedding models

        Parameters:
        - embedding_type (str): The type of embedding model to initialize.
        - embedding_model (str): The specific model within the chosen type.

        Returns:
        - embeddings: The initialized embedding model.

        """
        # Update embedding type and model
        self.embedding_type = embedding_type
        self.embedding_model = embedding_model

        self.score_factor = self.score_factor
        if self.embedding_type == "HuggingFaceEmbeddings":
            self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)

        elif self.embedding_type == "HuggingFaceInstructEmbeddings":
            self.embeddings = HuggingFaceInstructEmbeddings(model_name=self.embedding_model)

        elif self.embedding_type == "openai":
            self.embeddings = OpenAIEmbeddings(
                deployment=self.connection_param_dict["deployment"],
                model=self.connection_param_dict["model"],
                openai_api_base=self.connection_param_dict["api_base"],
                openai_api_type=self.connection_param_dict["api_type"],
                openai_api_key= self.connection_param_dict["api_key"],
                chunk_size=self.connection_param_dict["chunk_size"],
                max_retries=self.connection_param_dict["max_retries"],
                show_progress_bar=True
            )
        elif self.embedding_type == "SentenceTransformer":
            self.embeddings = SentenceTransformer(self.embedding_model)
        else:
            raise ValueError("Unsupported embedding model: {}".format(self.embedding_type))

        return self.embeddings



    def vector_storage(self,parsed_df):
        """
        Method to connect with vector DB and store the chunk_dataframe
        
        Parameters:
            - parsed_df(pd.DataFrame) : chunk_dataframe which includes the embeddings
                                        and unique identifier to each rows

        Returns: 
            - None
        """
        
        try:
            connections.connect(alias= self.milvus_params['connection']['alias'],
                            host= self.milvus_params['connection']['host'],
                            port= self.milvus_params['connection']['port'],
                            user = self.milvus_params['connection']['user'],
                            password= self.milvus_params['connection']['password'])
        except:
            self.logger.error("Unable to connect to vector database")
            raise Exception("Connection Failed")
        
        DIMENSION = len(parsed_df.iloc[0].embeddings)
        COLLECTION_NAME = self.milvus_params['collection']['name']

        if utility.has_collection(COLLECTION_NAME):
            utility.drop_collection(COLLECTION_NAME)

        fields = [
        FieldSchema(**{"name":"id", "dtype": DataType.INT64, "descrition":"Ids", "is_primary":True, "auto_id":False}),
        FieldSchema(**{"name":"page_content", "dtype": DataType.VARCHAR, "description":"texts inside the chunk", "max_length":5000}),
        FieldSchema(**{"name":"metadata", "dtype":DataType.JSON, "description":"chunk metadata", "max_length":200}),
        FieldSchema(**{"name":"embeddings", "dtype": DataType.FLOAT_VECTOR, "description":"chunk metadata", "dim": DIMENSION})
        ]
        schema = CollectionSchema(fields=fields, description=self.milvus_params['collection']['description'])
        collection = Collection(name=COLLECTION_NAME, schema=schema)

        index_params = self.milvus_params['index_params']
        collection.create_index(field_name="embeddings", index_params=index_params)

        data = [
            [], #ID
            [], #txt
            [], #metadata
            [] #embedding
        ]

        for i in tqdm(range(0, parsed_df.shape[0])):
            data[0].append(parsed_df['ID'][i])
            data[1].append(parsed_df['page_content'][i])
            data[2].append(parsed_df['metadata'][i])
            data[3].append(parsed_df['embeddings'][i])
            collection.insert(data)
            data = [[],[],[],[]]
        
        Collection(COLLECTION_NAME).flush()
        self.logger.info(f"Successfully saved the data to the collection: {COLLECTION_NAME}")
        return 



    def process_and_store_embeddings(self, chunk_df, embedding_type,embedding_model):
        """
        Process the input DataFrame and store the embeddings in a database.

        Parameters:
        - chunk_df (pd.DataFrame): chunk_df DataFrame.
        - embedding_type (str): Type of embedding to use.
        - embedding_model (str): The specific model within the chosen type.

        Returns:
            - processed_df (pd.DataFrame): DataFrame with added 'embeddings' column.

        """
        # Update embedding type and model
        self.embedding_type = embedding_type
        self.embedding_model = embedding_model
        self.logger.debug(f"Embedding Type: {self.embedding_type}")
        self.logger.debug(f"Embedding Model: {self.embedding_model}")
        # Initialize embedding model
        self._initialize_embedding(self.embedding_type,self.embedding_model)


        chunk_df['embeddings'] = chunk_df['page_content'].apply(lambda x: self.embeddings.embed_documents([x])).apply(lambda x : x[0]).apply(np.array)

        # Store embeddings in the database
        #vector_db = self.vector_storage(chunk_df)
        chunk_df = chunk_df.reset_index(drop=True) 
        chunk_df['ID'] = range(1,chunk_df.shape[0]+1)
        self.logger.info("Generated the embeddings of the chunks")

        self.vector_storage(chunk_df)
        return chunk_df