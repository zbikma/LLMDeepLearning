from llama_index.llms.ollama import Ollama
# lib lamaparse for loading unstructured , semi-structured files like pdf file in this example
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model
llm= Ollama(model="mistral",request_timeout=30.0)

result = llm.complete("hello world")
print(result)
