import pandas as pd
import os
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    TokenTextSplitter,
)
from langchain.docstore.document import Document
from docx import Document as DOC

class LangChunkExtractor:
    def __init__(self, data_config, model_config, user_config,logger):
        """
        This class extracts the chunks from the documents
        """
        self.user_config = user_config
        self.data_config = data_config
        self.model_config = model_config
        self.output_path = self.data_config['path']["logger"]["path"]
        self.logger =  logger

    def doc_chunk(self, parsed_df : pd.DataFrame , chunk_size=None, chunk_overlap=None, splitter=None):
        """
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
        # Read values from the data_config if not provided by the user
        chunk_size = chunk_size if chunk_size is not None else self.model_config['chunk_params']['chunk_size']
        chunk_overlap = chunk_overlap if chunk_overlap is not None else self.model_config['chunk_params'].get('chunk_overlap',
                                                                                                              0)
        splitter = splitter if splitter is not None else self.model_config['chunk_params'].get('splitter', 
                                                                                'RecursiveCharacterTextSplitter')

        if not isinstance(chunk_size, int) or not isinstance(chunk_overlap, int) or chunk_size <= 0 or chunk_overlap < 0:
            self.logger.error("Invalid chunk_size or chunk_overlap values. They should be positive integers.")
            raise ValueError("Invalid chunk_size or chunk_overlap values. They should be positive integers.")
        
        if splitter not in ["RecursiveCharacterTextSplitter", "SentenceTransformersTokenTextSplitter", "TokenTextSplitter"]:
            self.logger.error("Invalid splitter. Please provide a valid splitter.")
            raise ValueError("Invalid splitter. Please provide a valid splitter.")
        
        self.splitter =  splitter 
        self.logger.info(f"Splitter Method: {self.splitter}")

        try:
            text_splitter = self._get_text_splitter(splitter, chunk_size, chunk_overlap)
            parsed_df["splits"] = parsed_df.loaded_docs.apply(lambda x: text_splitter.split_documents(x))
            

            if self.model_config["chunk_params"]["save_summary"]:
                parsed_df["splits"].apply(self._write_list_to_file)
                self._export_document_summary(parsed_df=parsed_df)


        except Exception as e:
            self.logger.error(f"Error: General Exception type: {type(e).__name__}, Message: {e}")
        
        #return parsed_df
        res1 = parsed_df.explode('splits')
        chunks= res1['splits'].to_list()
        # Create a DataFrame from the list of objects
        data = {'page_content': [obj.page_content for obj in chunks],
        'metadata': [obj.metadata for obj in chunks]}
        chunk_df  = pd.DataFrame(data)
        return chunk_df
    
    def _export_document_summary(self, parsed_df):
        """
        Method to export the document summary

        Parameter:
            - parsed_df(pd.DataFrame): dataframe which contains the page_count,
                                        chunk_splits 
        
        Returns:
            - None
        """
        summary_df =  pd.DataFrame()
        summary_df["file_path" ] =  parsed_df['splits'].apply(
                                                    lambda x : 
                                                    x[0].metadata['source'] if len(x) > 0 and isinstance(x[0],Document) and "source" in x[0].metadata else None)
        summary_df["page_count"] = parsed_df['page_count']
        summary_df['chunk_counts'] =  parsed_df["splits"].apply(lambda x : len(x))
        summary_df['chunk_method'] =  self.splitter
        output_path = os.path.join(self.output_path, "doc_summary")
        os.makedirs(output_path, exist_ok=True)
        file_name =  self.model_config["chunk_params"]["summary_file_name"]
        summary_df.to_csv(os.path.join(output_path, file_name), index=False)
        self.logger.info(f"Exported the summary dataframe as {file_name} to path {output_path}")

    
    def _write_list_to_file(self,chunk_list):
        """
        Method to dump chunks for each document into the docx file
        
        Parameters:
            - chunk_list(list): list of chunks for the specific document
        Returns:
            - None
        """


        # Define the output file path (modify as needed)
        
        output_path = os.path.join(self.output_path, "chunks")
        os.makedirs(output_path, exist_ok=True)

        file_name = os.path.basename(chunk_list[0].metadata.get('source'))
        file_name = file_name.replace(".pdf", ".docx")
        output_file_path = os.path.join(output_path, f'{file_name}')
        
        #print(chunk_list)
        doc = DOC()
        
        for idx, document in enumerate(chunk_list):
            page_num  =  document.metadata.get('page')
            cleaned_text = document.page_content.strip()
            separator = f"\n{'*' * 40} Chunk {idx + 1} {'*' * 40} Page: { page_num }\n"
            formatted_text = f"{separator}{cleaned_text}\n"

            # Add formatted text to the DOCX document
            doc.add_paragraph(formatted_text)
        doc.save(output_file_path)
        self.logger.info(f"Saved the chunk to file {output_file_path}")


    def _get_text_splitter(self, splitter, chunk_size, chunk_overlap):
        """
        Method to select the splitter

        Parameters:
            - splitter (str): Method to split the documents into chunks.
            - chunk_size (int): size of each chunk.
            - chunk_overlap (int): amount of overlap between each chunk.

        Returns:
            - splitter Object (langchain.text_splitter)            


        """
        if splitter == "RecursiveCharacterTextSplitter":
            return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif splitter == "SentenceTransformersTokenTextSplitter":
            return SentenceTransformersTokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif splitter == "TokenTextSplitter":
            return TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
