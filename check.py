import streamlit as st
from streamlit_chat import message

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv
from langchain_upstage import UpstageEmbeddings
import pinecone
from pinecone import Pinecone
from langchain.agents import initialize_agent, Tool
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

load_dotenv('/upstage-ai-advanced-ir7/.env')

os.environ["UPSTAGE_API_KEY"] = os.getenv('UPSTAGE_API_KEY')
pinecone_api_key = os.getenv('PINE_API_KEY')

embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
pc = Pinecone(api_key=pinecone_api_key)
pc_index = pc.Index("session-chat-index")
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7, api_key=os.getenv("OPENAI_API_KEY"))


#RAG

import os
import json
import faiss
import numpy as np
from openai import OpenAI
import traceback
import openai  # 추가
from dotenv import load_dotenv
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import jsonlines
import MeCab
from rank_bm25 import BM25Okapi
import ast


# .env 파일 로드
load_dotenv('/upstage-ai-advanced-ir7/.env')

# API_KEY 값을 가져옴
openai_api_key = os.getenv("OPENAI_API_KEY")
upstage_api_key = os.getenv('UPSTAGE_API_KEY')



os.environ["OPENAI_API_KEY"] = openai_api_key

# Upstage API 클라이언트 설정
client = OpenAI(
    api_key= upstage_api_key,
    base_url="https://api.upstage.ai/v1/solar"
)

client_gpt = OpenAI()

mecab = MeCab.Tagger()

# JSONL 파일 경로
jsonl_file_path = '/upstage-ai-advanced-ir7/data/documents.jsonl'

# 문서와 docid 저장할 리스트
documents = []
docids = []

stoptags = {"E", "J", "SC", "SE", "SF", "VCN", "VCP", "VX"}

# JSONL 파일 읽기
with jsonlines.open(jsonl_file_path) as reader:
    for obj in reader:
        docids.append(obj['docid'])      # docid 저장
        documents.append(obj['content']) # content 저장

# Mecab을 사용하여 문서 토큰화
def tokenize_with_mecab(text):
    tokens = mecab.parse(text).splitlines()  # Mecab 결과를 라인 단위로 분리
    processed_tokens = []
    
    for token in tokens:
        if "\t" in token:  # 형태소와 품사 태그가 \t로 구분됨
            word, tag_info = token.split("\t")
            pos_tag = tag_info.split(",")[0]  # 품사 태그는 ,로 구분된 첫 번째 요소
            if pos_tag not in stoptags:  # 불필요한 품사 태그가 아닌 경우에만 추가
                processed_tokens.append(word)
    
    return processed_tokens

tokenized_corpus = [tokenize_with_mecab(doc) for doc in documents]

# BM25 인덱서 생성
bm25 = BM25Okapi(tokenized_corpus)

with open("/upstage-ai-advanced-ir7/data/eval.jsonl", "r") as f:
    eval_doc_mapping = [json.loads(line) for line in f]
    

doc_mapping = {}
with open("/upstage-ai-advanced-ir7/data/documents.jsonl", "r") as f:
    for line in f:
        doc = json.loads(line)
        doc_mapping[doc['docid']] = doc 

index = faiss.read_index("../knn_index_cosine.faiss")

