from llama_index.llms.ollama import Ollama
import ast
import os
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
# lib lamaparse for loading unstructured , semi-structured files like pdf file in this example
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.agent import ReActAgent 
from prompts import context,code_parser_template
from codereader import code_reader
from pydantic import BaseModel
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.query_pipeline import QueryPipeline

llm= Ollama(model="mistral",request_timeout=3600.0)
# we want to parse our file specially in case of unstructured data like pdf files so we can better query it 
parser =LlamaParse(result_type="markdown")

# dictionary that holds the parser for different file types
file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader("./data",file_extractor=file_extractor).load_data()

# get access to a local model by default it will used chatgpt but we want to use our own local model for embedding
embed_model =  resolve_embed_model("local:BAAI/bge-m3")
# create the vector index from the document we want to be using as embdding, this index becomes like a question and answer bot that then return some result based on the pdf file and then use the llm to generate final result
vector_index= VectorStoreIndex.from_documents(documents,embed_model=embed_model)
query_engine= vector_index.as_query_engine(llm=llm)
tools = [
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="api_documentation",
            description="this gives documentation about code for an API. Use this for reading docs for the API",
        ),
    ),
    code_reader,
]
# codellama is a model that is designed for code generation instead of just question and naswer
code_llm = Ollama(model="codellama",request_timeout=3600.0)
agent = ReActAgent.from_tools(tools, llm=code_llm, verbose=True, context=context)
class CodeOutPut(BaseModel):
    code: str
    description : str
    filename : str

parser = PydanticOutputParser(CodeOutPut)
json_prompt_str = parser.format(code_parser_template)
json_prompt_temp= PromptTemplate(json_prompt_str)
output_pipeline = QueryPipeline(chain=[json_prompt_temp,llm])
while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    retries = 0

    while retries < 3:
        try:
            result = agent.query(prompt)
            print("Agent response:", result)
            next_result = output_pipeline.run(response=result)
            cleaned_json = ast.literal_eval(str(next_result).replace("assistant:", ""))
            break
        except (ValueError, SyntaxError, AttributeError) as e:
            retries += 1
            print(f"Error occurred, retry #{retries}: {e}")
            if retries >= 3:
                raise RuntimeError("Maximum retries reached") from e

    if retries >= 3:
        print("Unable to process request, try again...")
        continue

    print("Code generated")
    print(cleaned_json["code"])
    print("\n\nDesciption:", cleaned_json["description"])

    filename = cleaned_json["filename"]

    try:
        with open(os.path.join("output", filename), "w") as f:
            f.write(cleaned_json["code"])
        print("Saved file", filename)
    except:
        print("Error saving file...")