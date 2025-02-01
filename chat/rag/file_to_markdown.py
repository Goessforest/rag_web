import glob
import nest_asyncio
import shutil
from llama_parse import LlamaParse
from llama_index.core.schema import Document, MediaResource
from typing import *
import pdfplumber
import os, re

from pydantic_core import ValidationError
from queue import Queue
from ..open_ai.openai_query import OpenAIQuery
import logging

DEFAULT_PROMPT = """Please name the document. 
        Striclty return only a string in the format first Author ( + et al if at least 2) Year /keywords
        the keywords should have maximum 3 key words or 60 characters.
        example: "Smith et al 2021 climate_change_impact environment"
        or "Smith 2021 clouds environment"
        """ 

MAX_OPENAI_CHARACTERS = 4096 * 2
class FileToMarkdown:
    def __init__(self ):
        """This class converts files to markdown. This takes about 30 seconds per page"""
        
        # Initialize Openai Query
        self._openai_query = OpenAIQuery()


    def parseFile(self, path:str) -> Tuple[str, str]:
        '''Processes all files from a path and name them using OpenAI
        Returns a tuple of the file content and the new name'''
        assert os.path.exists(path), "File does not exist"
        original_file_names = os.path.basename(path)
        try:
            # use llamaParse to parse the file in a high quality way
            nest_asyncio.apply()
            llama_parser =  LlamaParse(
                api_key=os.environ.get('llama_cloud_api'),  # can also be set in your env as LLAMA_CLOUD_API_KEY
                result_type="markdown",  # "markdown" and "text" are available
                num_workers=4,  # if multiple files passed, split in `num_workers` API calls
                verbose=True,
                # language="en",  # Optionally you can define a language, default=en
            )
            pages = llama_parser.load_data(path)
        except ValidationError as e:
            # API KEY MISSING -> USE PUMMER INSTAED
            logging.error(f"API KEY MISSING -> USE PUMMER INSTAED: {e}")
            pages = [Document(text_resource=MediaResource(text="NO_CONTENT_HERE"))]
        

        # Flatten the pages and use plummer to extract text if parsing failed
        file_content = ""
        for index, page in enumerate(pages):
            file_content += f"# Page {index} of File: %%FILE_NAME%%\n\n"

            page_content = page.text_resource.text

            # if parsing Failed with plummer
            if page_content == "NO_CONTENT_HERE":
                logging.warning(f"---------Plummer triggerd for page {index} of file {path}---------")
                page_content = self._extractPdfPageWithPlummer(path, pageNumber=index)
            
            file_content += page_content + "\n\n"

        # get new name
        new_name = self.get_file_name(file_content, original_file_names)

        # replace the placeholder with the new name
        file_content = file_content.replace("%%FILE_NAME%%", new_name)
        
        return file_content, new_name



    def get_file_name(self, md_file:str, file_name:str) -> str:
        '''Returns the file name from a markdown file using OpenAI'''
        try:   
            # shorten the file if it is too long
            file_charaters = len(md_file)
            if file_charaters > MAX_OPENAI_CHARACTERS:
                limit = int(MAX_OPENAI_CHARACTERS/2)
                start = md_file[:limit]
                end = md_file[-limit:]
                md_file = start + "\n ...(section shortend)... \n" + end

            # generate response
            response = self._openai_query.query(prompt=md_file, instructions=DEFAULT_PROMPT, max_tokens=100, messages=[])
            ai_file_name = response.content
            if bool(re.match(r"^[A-Z][a-zA-Z]+( et al)? \d{4} .+$", ai_file_name)):
                return ai_file_name[:80]
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            logging.error(f"Error in getting file name: {e}")

            
        return file_name
        


    @staticmethod
    def _extractPdfPageWithPlummer(inputPdfPath:str, pageNumber:int=0) -> str:
        '''Extracts text from a pdf file'''
        # Open the original PDF in read-binary mode
        with pdfplumber.open(inputPdfPath) as pdf:
            # Extract the page
            first_page = pdf.pages[pageNumber]
            
            # Extract text from the first page
            text = first_page.extract_text()
            
            return str(text)
        






