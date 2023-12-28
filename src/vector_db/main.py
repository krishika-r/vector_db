from text_extraction import DocumentExtractor
from extract_embeddings import EmbeddingExtractor
from chunking import LangChunkExtractor
from retrieve_from_vector_db import VectorSearch
from  utils import check_input_path
import openai
from pymilvus import exceptions
from logger import logger
import pandas as pd 
import sys 

class VectorDB:
    """
    Main class that controls individual steps outputs.
    """

    def __init__(
        self,
        user_config: dict,
        data_config: dict,
        model_config: dict,
    ) -> None:
        """Class constructor"""

        self.user_config = user_config
        self.data_config = data_config
        self.model_config = model_config

    def pdf_to_text_extraction(
        self, input_file_path: str = None, parser : str = None) -> str:

        """
        Extract text from PDF and return processed document dataframe.

        Parameter:
        - input_file_path (str, optional): Path to the input PDF file. If not provided, uses the default path from data config.
        - parser (str, optional): Parser to be used. If not provided, uses the default parser from model config.

        Returns:
        - all_processed_docs_df (dataframe): Processed document dataframe.
        """
        logger.info("STEP1: Text extraction from pdf started")
        try:
            if input_file_path is None:
                # If input_file_path is not provided by the user, get the default path from data config
                input_file_path = self.data_config['input_data_path']

            if parser is None:
                # If parser is not provided by the user, get the default parser from model config
                parser = self.model_config['extract_params']['parser']

            # Check if the input path is valid (either a folder containing PDFs or a PDF file)
            check_input_path(input_file_path)

            self.input_file_path = input_file_path

            doc = DocumentExtractor(
            user_config=self.user_config,
            model_config=self.model_config,
            data_config=self.data_config,
            logger=logger
            )

            all_processed_doc_df = doc.extract_text_from_folder_multiprocessing(folder_path=input_file_path,parser=parser)
            #logger.debug("DataFrame:\n%s", all_processed_doc_df.to_string(index=False))
            logger.info("STEP1 Completed")
            return all_processed_doc_df
            
        except FileNotFoundError as e:
            logger.error(f"Error: File not found at {input_file_path}.")
            raise e
            
        except IsADirectoryError as e:
            logger.error(f"Error: {input_file_path} is a directory. Please provide a valid PDF file path.")
            raise e
            
        except ValueError as e:
            logger.error(f"Error: {e} ")
            raise e
            
        except Exception as e:
            logger.error(f"Error: General Exception type: {type(e).__name__}, Message: {e}")
            raise e


    def generate_chunks(self,parsed_df:pd.DataFrame , chunk_size:int= None, chunk_overlap:int= None, splitter:str= None ):
        """
        Generate chunks for each documents based on splitter, chunk size and chunk_overlap

        Reads the dataframe generated in previous step and generate
        chunk for each document. 

        Parameters:
            - parsed_df (pd.Dataframe): Output dataframe from the text extraction step.
            - chunk_size (int): size of each chunk.
            - chunk_overlap (int): amount of overlap between each chunk.
            - splitter (str): Method to split the documents into chunks.

        Returns:
            - Chunk Dataframe(pd.DataFrame)
        """
        logger.info("STEP2: Chunk generation started")
        if 'loaded_docs' not in parsed_df.columns:
            raise Exception("Required column name 'loaded_docs' is missing from the parsed_df dataframe")
        
        chunk_extractor =  LangChunkExtractor(user_config = self.user_config,
                    data_config = self.data_config,
                    model_config = self.model_config,
                    logger = logger)
        

        chunk_df = chunk_extractor.doc_chunk(parsed_df=parsed_df, 
                                  chunk_overlap=chunk_overlap, 
                                  chunk_size=chunk_size, 
                                  splitter=splitter)
        
        logger.info("STEP2 Completed")
        return chunk_df
    
    
    

    def process_embedding_and_store_db(self, chunk_data, embedding_type :str = None,embedding_model:str = None):

        """
        Process the chunk DataFrame and store the embeddings in a database.

        Parameters:
        - chunk_data (pd.DataFrame): Input DataFrame.
        - embedding_type (str, optional): Type of embedding to use.
        - embedding_model (str, optional): embedding model name.
        """
        logger.info("STEP3: Embedding generation and storage step started")
        try:
            # If embedding_type,model is not provided by the user, get the default embedding_type/model from model config
            embedding_type = embedding_type or self.model_config['embedding_params']['embedding_type']
            embedding_model = embedding_model or self.model_config['embedding_params']['embedding_model']

        
            # Initialize EmbeddingExtractor
            embedding_extractor = EmbeddingExtractor(
                user_config=self.user_config,
                model_config=self.model_config,
                data_config=self.data_config,
                logger=logger
                )

            # Perform vector storage
            vector_db = embedding_extractor.process_and_store_embeddings(chunk_data,embedding_type,embedding_model)
            logger.info("STEP3 Completed")
            return vector_db


        except openai.error.APIError as e:
            logger.error(f"Something went wrong on the OpenAI side. Please pass the df again.\nError:{e}")
            raise e

        except openai.error.Timeout as e:
            logger.error(f"Request to GPT timed out.Please pass the df again..\nError:{e}")
            raise e

        except openai.error.RateLimitError as e:
            logger.error(f"Ratelimit exceeded. Please pass the df again.\nError:{e}")
            raise e 

        except openai.error.APIConnectionError as e:
            logger.error(f"Could not establish connection to OpenAI's services. Please check with your IT team.\nError:{e}")
            raise e

        except openai.error.AuthenticationError as e:
            logger.error(f"The API key may have been expired. Please check with your IT team.\nError:{e}")
            raise e  

        except openai.error.ServiceUnavailableError as e:
            logger.error(f"OpenAI's services are not available at the moment. Please pass the df again. If problem still persists, please check with your IT team.\nError:{e}")
            raise e   

        except Exception as e:
            logger.error(f"Error: General Exception type: {type(e).__name__}, Message: {e}")
            raise e


    def retreive_from_vector_db(self, search_query, threshold:float =None,
     anns_field:str = None, search_params:str = None, limit:int = None, 
     expr:str = None, output_fields:str = None, consistency_level:str = None,
     embedding_type :str = None,embedding_model:str= None):

        """
        Retrieve data from a vector database based on specified parameters.

        Parameters:
        - search_query (str): The query used for searching in the vector database.
        - anns_field (str, optional): The field used for annotations in the vector database.
        - search_params (str, optional): Additional parameters for the search operation.
        - limit (int, optional): The maximum number of results to retrieve.
        - expr (str, optional): An expression to further filter the search results.
        - output_fields (str , optional): The names of the fields to retrieve from the search results.
        - consistency_level (str, optional): The consistency level for the search operation.
        - embedding_type (str, optional): Type of embedding to use.
        - embedding_model (str, optional): embedding model name.
        - threshold (float, optional): The threshold value for filtering results


        Returns:
        - vector_db: The retrieved data from the vector database.
        """
        logger.info("STEP4: Querying from the vectorDB")
        try:
            # Set default values if None
            anns_field = anns_field or self.model_config['search_vector_params']['anns_field']
            search_params = search_params or self.model_config['search_vector_params']['search_params']
            limit = limit or self.model_config['search_vector_params']['limit']
            expr = expr or self.model_config['search_vector_params']['expr']
            output_fields = output_fields or self.model_config['search_vector_params']['output_fields']
            consistency_level = consistency_level or self.model_config['search_vector_params']['consistency_level']
            embedding_type = embedding_type or self.model_config['embedding_params']['embedding_type']
            embedding_model = embedding_model or self.model_config['embedding_params']['embedding_model']
            threshold = threshold or self.model_config['search_vector_params']['threshold']

                
            # Initialize EmbeddingExtractor
            embedding_extractor = EmbeddingExtractor(
                user_config=self.user_config,
                model_config=self.model_config,
                data_config=self.data_config,
                logger=logger
                )

            embeddings = embedding_extractor._initialize_embedding(embedding_type,embedding_model)

            # Initialize VectorSearch
            vector_extractor = VectorSearch(
                user_config=self.user_config,
                model_config=self.model_config,
                data_config=self.data_config,
                embeddings= embeddings,
                logger=logger
                )
            # Perform vector storage
            vector_db = vector_extractor.perform_search(search_query, anns_field, search_params, limit, expr, output_fields, consistency_level)
            
            #filtered based on threshold
            filtered_vector_db = [result for result in vector_db[0] if result.distance > threshold]
            filtered_vector_db

            logger.info("STEP4 Completed")
            return filtered_vector_db

        except openai.error.APIError as e:
            logger.error(f"Something went wrong on the OpenAI side. Please pass the df again.\nError:{e}")
            raise e
            
        except openai.error.Timeout as e:
            logger.error(f"Request to GPT timed out.Please pass the df again..\nError:{e}")
            raise e

        except openai.error.RateLimitError as e:
            logger.error(f"Ratelimit exceeded. Please pass the df again.\nError:{e}")
            raise e

        except openai.error.APIConnectionError as e:
            logger.error(f"Could not establish connection to OpenAI's services. Please check with your IT team.\nError:{e}")
            raise e
        
        except openai.error.AuthenticationError as e:
            logger.error(f"The API key may have been expired. Please check with your IT team.\nError:{e}")
            raise e  

        except openai.error.ServiceUnavailableError as e:
            logger.error(f"OpenAI's services are not available at the moment. Please pass the df again. If problem still persists, please check with your IT team.\nError:{e}")
            raise e   

        except exceptions.ConnectionNotExistException as e:
            logger.error(f"Connection not exist exception occurred : {e}")
            raise e
            
        except Exception as e:
            logger.error(f"Error: General Exception type: {type(e).__name__}, Message: {e}")
            raise e

