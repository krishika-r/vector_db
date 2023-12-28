import pandas as pd
import os
from docx import Document as DOC 
import multiprocessing
import glob
import time
# from tigernlp.doc_parsing.api import PdfMiner,PdfPlumber,PyMuPdf,PyPdf2,PyPDFium
from utils import clean_text
from langchain.document_loaders import PyPDFLoader, PyMuPDFLoader, UnstructuredPDFLoader, PyPDFium2Loader, PDFMinerLoader, PDFPlumberLoader, PDFMinerPDFasHTMLLoader,UnstructuredFileLoader

class DocumentExtractor:
    """
    Tigernlp document parser provides methods to extract pdf to text
    """

    def __init__(self, data_config, model_config, user_config,logger):
        self.user_config = user_config 
        self.data_config = data_config
        self.model_config = model_config
        # pdf parser method
        self.parser = model_config['extract_params']['parser']
        # Check if doc_extraction_output_flag is set in the config
        self.logger =  logger
        self.doc_extraction_output_flag = data_config['path']['doc_extraction_output_flag']
        self.output_path = data_config['path']['logger']['path']

    def _save_outputs(self, text, file_path):
        """
        Save the extracted text outputs in the respective folders for each file if output_flag is set to True.
        """
        if not self.doc_extraction_output_flag:
            return  # Skip saving if output_flag is False

        else:
            output_path = os.path.join(self.output_path, "extracted_text")
            os.makedirs(output_path, exist_ok=True)

            # Extract the file name from the path
            file_name, ext = os.path.splitext(os.path.basename(file_path))
            output_file_path = os.path.join(output_path,f'{file_name}_output.docx')
            # Save the formatted text to the output file
            doc = DOC()
            for idx, document in enumerate(text):
                cleaned_text = document.page_content.strip()
                separator = f"\n{'*' * 40} Page {idx + 1} {'*' * 40}\n"
                formatted_text = f"{separator}{cleaned_text}\n"

                # Add formatted text to the DOCX document
                doc.add_paragraph(formatted_text)

            # Save the DOCX document
            doc.save(output_file_path)

            self.logger.info(f"Extracted text for {file_name} saved to {output_file_path}")

        return


    def doc_extraction(self, file_path, parser=None):
        """
        Extract text from the PDF files using the specified parser.

        Parameters:
        - file_path (str): Path to the PDF files.
        - parser (str): Name of the parser to be used.

        Returns:
        - loaded_docs (list): List of loaded documents, where each document is represented as a string.
        - page_count (int): Number of pages in the PDF file.
        - parse_time (float): Time taken to parse the PDF file, in seconds.
        """

        if parser is None:
            parser = self.parser
   

        if 'UnstructuredPDFLoader' in parser:
            loader = UnstructuredPDFLoader(file_path, mode="elements")
        elif 'PyPDFLoader' in parser:
            loader = PyPDFLoader(file_path)
        elif "PyPDFium2Loader" in parser:
            loader = PyPDFium2Loader(file_path)
        elif 'PDFMinerLoader' in parser:
            loader = PDFMinerLoader(file_path)
        elif "PyMuPDFLoader" in parser:
            loader = PyMuPDFLoader(file_path)
        elif 'UnstructuredFileLoader' in parser:
            loader = UnstructuredFileLoader(file_path, mode="elements")
        else:
            raise ValueError("Unsupported parser: {}".format(parser))


        start_time = time.time()
        loaded_docs = loader.load()
        end_time = time.time()

        parse_time = end_time - start_time
        page_count = int(len(loaded_docs))

        if self.doc_extraction_output_flag:
            # Save extracted text outputs for each pdf file in output folder
            self._save_outputs(loaded_docs, file_path)

        return loaded_docs, page_count, parse_time
      
    def _process_single_file(self, row, parser):
        """
        Helper function to process a single PDF file.
        """
        file_path = row['file_path']
        return pd.Series(self.doc_extraction(file_path, parser))

    def extract_text_from_folder_multiprocessing(self, folder_path, parser=None, num_processes=None):
        """
        Process all PDF files in a folder using multiprocessing.

        Parameters:
        - folder_path (str): Path to the folder containing PDF files.
        - parser (str): Name of the parser to be used.
        - num_processes (int): Number of parallel processes to use.

        Returns:
        - all_processed_docs_df (dataframe): Processed document dataframe.
        """
        pdf_files = glob.glob(os.path.join(folder_path, '*.pdf'))

        # Set the number of processes to the specified value or the available CPU cores
        if num_processes is None:
            num_processes = multiprocessing.cpu_count()

        # Use multiprocessing.Pool to parallelize document extraction
        with multiprocessing.Pool(processes=num_processes) as pool:
            all_processed_docs_df = pd.DataFrame(pdf_files, columns=['file_path']).apply(
                lambda row: self._process_single_file(row, parser),  # Pass parser as an argument
                axis=1
            )

        all_processed_docs_df.columns = ['loaded_docs', 'page_count', 'parse_time']
        all_processed_docs_df['cleaned_text'] = all_processed_docs_df['loaded_docs'].apply(clean_text)

        # Save the DataFrame to a CSV file
        csv_path = os.path.join(self.output_path, 'extracted_text','pdf_to_text_df.csv')
        all_processed_docs_df.to_csv(csv_path, index=False, encoding='utf-8')
        self.logger.info(f"Dataframe saved to {self.output_path}")

        return all_processed_docs_df
