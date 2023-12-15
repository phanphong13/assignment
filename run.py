from gpt_utils.models import get_gpt_llm

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import json
from pydantic import BaseModel, Field
from typing import List
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import HumanMessagePromptTemplate
from langchain.prompts import SystemMessagePromptTemplate
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseOutputParser
from langchain.chains.base import Chain
from langchain.chains.question_answering import load_qa_chain


import torch


DB_FAISS_PATH = "vectorstores/db_faiss/"

llm = get_gpt_llm()

modelPath = "sentence-transformers/all-MiniLM-l6-v2"

model_kwargs = {'device':'cpu'}

encode_kwargs = {'normalize_embeddings': False}

# Initialize an instance of HuggingFaceEmbeddings with the specified parameters
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,     # Provide the pre-trained model's path
    model_kwargs=model_kwargs, # Pass the model configuration options
    encode_kwargs=encode_kwargs # Pass the encoding options
)

def build_qa_chain(llm: type[BaseChatModel]) -> type[Chain]:

    # -- Create system prompt template
    sys_tpl = "You are a helpful assistant with expertise in security issues"
    sys_msg_pt = SystemMessagePromptTemplate.from_template(sys_tpl)

    usr_pt = PromptTemplate(template="{context}\n{question}\n",
                            input_variables=["context", "question"])
    usr_msg_pt = HumanMessagePromptTemplate(prompt=usr_pt)

    # -- Combine (system, user) into a chat prompt template
    prompt = ChatPromptTemplate.from_messages([sys_msg_pt, usr_msg_pt])

    # Create chain for QA and pass the prompt instead of plain text query
    chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)
    return chain


def run_qa_chain(chain, query, vec_db) -> str:
    # Lookup
    docs = vec_db.similarity_search(query, k=10, include_metadata=True)
    res = chain({"input_documents": docs, "question": query})
    return res["output_text"]

vec_db = FAISS.load_local(DB_FAISS_PATH,embeddings)
chain = build_qa_chain(llm)
query = "What kind of user information is collected during the use of services?"

test = run_qa_chain(chain, query, vec_db)

print(test)