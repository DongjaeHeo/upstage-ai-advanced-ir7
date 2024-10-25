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

# gpu_index = faiss.index_cpu_to_gpu(res, 0, index)

with open("../chunk_mappings.json", "r") as f:
    chunk_doc_mapping = json.load(f)
    
model_path = 'Dongjin-kr/ko-reranker'

def exp_normalize(x):
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()
    
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.to('cuda')
model.eval()

def normalize(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return (embeddings / norms).astype(np.float32)  # float32로 변환

def min_max_normalize(ranked_docs):
    """
    Min-Max 정규화를 수행하는 함수
    Args:
        ranked_docs (list of tuple): [(id, score), (id, score), ...] 형태의 리스트
    Returns:
        normalized_ranked_docs (list of tuple): [(id, normalized_score), ...] 형태의 리스트
    """
    # 점수만 추출
    scores = np.array([score for _, score in ranked_docs])

    # Min-Max 정규화
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    # 0으로 나누는 오류를 방지 (최소값과 최대값이 같은 경우)
    if max_score == min_score:
        normalized_scores = np.ones_like(scores)
    else:
        normalized_scores = (scores - min_score) / (max_score - min_score)

    # 정규화된 점수와 id를 다시 결합
    normalized_ranked_docs = [(docid, score) for (docid, _), score in zip(ranked_docs, normalized_scores)]
    
    return normalized_ranked_docs


def z_score_normalize(ranked_docs):
    """
    Z-Score 정규화를 수행하는 함수
    Args:
        ranked_docs (list): (docid, score)의 리스트
    Returns:
        normalized_ranked_docs (list): Z-Score로 정규화된 (docid, normalized_score)의 리스트
    """
    # 점수만 추출
    scores = np.array([score for _, score in ranked_docs])

    # Z-Score 정규화
    mean_score = np.mean(scores)
    std_dev = np.std(scores)
    
    # 표준편차가 0인 경우(모든 점수가 동일한 경우) 처리
    if std_dev == 0:
        normalized_scores = np.zeros_like(scores)
    else:
        normalized_scores = (scores - mean_score) / std_dev

    # 정규화된 점수와 docid를 다시 결합
    normalized_ranked_docs = [(docid, score) for (docid, _), score in zip(ranked_docs, normalized_scores)]
    
    return normalized_ranked_docs

def merge_and_sum_scores(knn_retrieved_docs, bm25_retrieved_docs, p):
    """
    knn_retrieved_docs와 bm25_retrieved_docs에서 동일한 id를 가진 항목은 스코어를 p와 (1 - p)의 비율로 가중 합산하고,
    그렇지 않은 항목은 그대로 유지하며, 마지막에 점수대로 정렬하는 함수.
    
    Args:
        knn_retrieved_docs (list of tuple): [(id, score), ...] 형태의 리스트
        bm25_retrieved_docs (list of tuple): [(id, score), ...] 형태의 리스트
        p (float): knn 점수에 적용할 가중치 (0 <= p <= 1)
    
    Returns:
        merged_docs (list of tuple): [(id, combined_score), ...] 형태의 리스트, 점수 내림차순 정렬
    """
    # 두 리스트를 딕셔너리로 변환하여 빠르게 검색할 수 있게 함
    knn_dict = {doc_id: score for doc_id, score in knn_retrieved_docs}
    bm25_dict = {doc_id: score for doc_id, score in bm25_retrieved_docs}
    
    # 모든 unique id를 set으로 결합
    all_ids = set(knn_dict.keys()).union(set(bm25_dict.keys()))

    # 동일한 id가 있으면 스코어를 p와 (1 - p) 비율로 가중 합산
    merged_docs = []
    for doc_id in all_ids:
        knn_score = knn_dict.get(doc_id, 0)  # knn에 없으면 0으로 간주
        bm25_score = bm25_dict.get(doc_id, 0)  # bm25에 없으면 0으로 간주
        combined_score = p * knn_score + (1 - p) * bm25_score
        merged_docs.append((doc_id, combined_score))
    
    # 점수 내림차순으로 정렬
    merged_docs = sorted(merged_docs, key=lambda x: x[1], reverse=True)
    
    return merged_docs

def retrieval_with_score(query):
    query_result = client.embeddings.create(
        model="solar-embedding-1-large-query",
        input=query
    ).data[0].embedding

    query_embedding = np.array(query_result).reshape(1, -1)

    # 쿼리 임베딩 정규화
    normalized_query = normalize(query_embedding)

    # FAISS로 500개의 유사한 chunk 검색
    k = 200
    distances, indices = index.search(normalized_query, k)


    retrieved_doc_ids = []

    # 코사인 거리를 코사인 유사도로 변환 (1 - 거리)
    for i, idx in enumerate(indices[0]):
        chunk_info = chunk_doc_mapping[idx]
        # content = chunk_info['content']
        doc_id = chunk_info['doc_id']
        
        # 코사인 거리를 코사인 유사도로 변환
        cosine_distance = distances[0][i]
        cosine_similarity = 1 - cosine_distance  # 유사도는 1에서 거리값을 뺀 것

        # retrieved_chunks.append((query, content))
        retrieved_doc_ids.append((doc_id, cosine_similarity))
        # retrieved_scores.append(cosine_similarity)  # 유사도 점수 추가
    
    knn_retrieved_docs = z_score_normalize(retrieved_doc_ids)
    
    # BM25
    
    tokenized_query = tokenize_with_mecab(query)

    # BM25로 점수 계산
    doc_scores = bm25.get_scores(tokenized_query)

    # 점수가 높은 순서대로 문서 정렬 (상위 200개만)
    ranked_docs = sorted(zip(docids, doc_scores), key=lambda x: x[1], reverse=True)[:k]
    
    bm25_retrieved_docs = z_score_normalize(ranked_docs)

    merged_docs = merge_and_sum_scores(knn_retrieved_docs, bm25_retrieved_docs, 0.7)
    
    retrieved_chunks = []
    for i, idx in enumerate(merged_docs):
        doc_info = doc_mapping[idx[0]]
        content = doc_info['content']
        retrieved_chunks.append((query, content))



    with torch.no_grad():
        inputs = tokenizer(retrieved_chunks, padding=True, truncation=True, return_tensors='pt', max_length=512).to('cuda')
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()

    # 상위 5개의 문서 선택 (내림차순 정렬)
    top_5_indices = torch.argsort(scores, descending=True)

    # 상위 5개의 문서 출력
    final_docs = []
    print("상위 5개의 문서:")
    for i in top_5_indices:
        doc_index = retrieved_chunks[i][0]
        doc_text = retrieved_chunks[i][1]
        doc_id = merged_docs[i][0]
        final_docs.append((doc_id, scores[i].item()))
        # print(f"문서 인덱스: {doc_index}")
        # print(f"연관된 문서 ID: {doc_id}")
        # print(f"문서 내용: {doc_text}")
        
        # print(f"유사도 점수: {scores[i].item()}")
        # print("-" * 50)
        
    final_docs = z_score_normalize(final_docs)
    
    final_merged_docs = merge_and_sum_scores(final_docs, merged_docs, 0.475)
    
    
    return final_merged_docs[:5]

def llm_reranking(query, documents):
    """
    LLM을 사용하여 사용자의 여러 메시지를 기반으로 검색에 적합한 단일 쿼리 생성.
    """
    # 시스템 메시지로 LLM에게 과제 부여 (한국어로)
    
    
    
    check_doc = ''
    for i, doc in enumerate(documents):
        check_doc += f"문서 {i+1}: {doc_mapping[doc[0]]['content']} \n"

    # 시스템 메시지 생성
    system_message = {
        "role": "system",
        "content": f"""
        당신은 문서와 질문을 비교하여, 질문에 가장 적합한 문서들을 찾고 그 순서를 반환하는 전문가입니다.
        
        다음의 규칙을 반드시 따르세요:
        1. 주어진 질문과 문서들의 내용을 비교하여, 질문의 답을 찾을 수 있는 문서를 가장 적합한 순서대로 나열하세요.
        2. 반드시 리스트 형태로만 출력하세요. 다른 형식은 허용되지 않습니다. 예를 들어 [1, 3, 2, 4]와 같이 반환하세요.
        3. 리스트에 들어갈 문서 번호는 1부터 시작하며, 문서들의 순서를 중요도에 따라 배열하세요.
        4. 리스트 외의 다른 정보를 출력하지 마세요. 리스트 이외의 내용은 모델이 자동으로 무시해야 합니다.

        질문: {query}

        문서들:
        {check_doc}
        """
    }

# 사용자 메시지 준비


    # 사용자 메시지 준비
    # system_message['content'] = system_message['content'] + check_doc

    # LLM 호출을 위한 메시지 배열 생성
    system_message = [system_message]

    # OpenAI API 호출하여 적절한 검색 쿼리 생성
    result = client_gpt.chat.completions.create(
        model="chatgpt-4o-latest",  # LLM 모델 지정
        messages=system_message,
        temperature=0
    )

    # LLM이 생성한 쿼리 반환 (ChatCompletionMessage 형식에서 content에 직접 접근)
    transformed_query = result.choices[0].message.content
    print(f"변환된 쿼리: {transformed_query}")
    return transformed_query

def retrieve_documents(query):
    retrieved_doc = retrieval_with_score(query)
    check_list = []
    for doc in retrieved_doc:
        print(retrieved_doc[0][1] * 0.1)
        print(doc[1])
        print('*' * 100)
        if doc[1] > retrieved_doc[0][1] * 0.1:
            check_list.append(doc)
    
    if len(check_list) > 1:
        llm_result = llm_reranking(query, check_list)
        llm_result = ast.literal_eval(llm_result)
            
        
        new_result = []
        for f_r in llm_result:
            # print(result[f_r  - 1])
            new_result.append(retrieved_doc[f_r - 1])

        retrieved_doc = new_result + retrieved_doc[len(new_result):]
        
    return doc_mapping[retrieved_doc[0][0]]['content']

#lang chain

from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

# ThreadPoolExecutor 생성 (최대 3개의 스레드)
executor = ThreadPoolExecutor(max_workers=3)




# 대화 저장 함수 (시간 추가)
async def async_store_conversation(session_id, text, role):
    loop = asyncio.get_event_loop()
    
    # user와 ai의 대화를 한 번에 저장
    await loop.run_in_executor(executor, store_conversations, session_id, text, role)

# 사용자 입력과 AI 응답을 한 번에 저장하는 함수
def store_conversations(session_id, text, role):
    try:
        # 사용자 입력 저장
        # text_len = len(text.split())
        # if text_len > 100:
        #     text = summarize(text)
        
        store_conversation(session_id, text, role)
        
        # print(text,role)
        print(f"Conversations stored successfully for session: {session_id}")
    except Exception as e:
        print(f"Error storing conversations: {e}")

# 기존의 저장 함수 (임베딩 생성 및 Pinecone 업서트 처리)
def store_conversation(session_id, text, role):
    text_embedding = embeddings.embed_query(text)
    unique_id = f"{session_id}-{role}-{len(text)}"
    metadata = {
        "session_id": session_id,
        "text": text,
        "role": role,
        "timestamp": int(time.time())  # 초 단위로 저장
    }
    pc_index.upsert(vectors=[(unique_id, text_embedding, metadata)])

# 대화 검색 함수 (최근 3개 + 연관성 높은 3개)
def retrieve_conversations(session_id, query_text, top_k=3):
    query_embedding = embeddings.embed_query(query_text)  # 쿼리 임베딩 생성

    # Pinecone에서 연관성이 높은 대화 검색
    related_results = pc_index.query(
        vector=query_embedding,  # 벡터 전달
        top_k=top_k,
        filter={"session_id": {"$eq": session_id}},
        include_metadata=True
    )

    related_texts = [
        {
            "text": match['metadata'].get('text', 'No text available'),
            "timestamp": match['metadata'].get('timestamp', None)
        }
        for match in related_results["matches"]
    ]

    # Pinecone에서 최근 대화 가져오기
    recent_results = pc_index.query(
        vector=query_embedding,
        top_k=10000,
        filter={"session_id": {"$eq": session_id}},
        include_metadata=True
    )

    recent_texts = [
        {
            "text": res['metadata'].get('text', 'No text available'),
            "timestamp": res['metadata'].get('timestamp', None)
        }
        for res in recent_results["matches"]
    ]

    # timestamp를 기준으로 내림차순 정렬
    recent_texts = sorted(recent_texts, key=lambda x: float(x['timestamp']), reverse=True)[:3]
    related_texts = sorted(related_texts, key=lambda x: float(x['timestamp']), reverse=True)

    return recent_texts, related_texts

# 검색된 대화 기록을 프롬프트에 맞게 정렬 후 반환
def format_conversation_history(recent_texts, related_texts):
    recent_history = "\n".join([f"Timestamp: {entry['timestamp']}, Text: {entry['text']}" for entry in recent_texts])
    related_history = "\n".join([f"Timestamp: {entry['timestamp']}, Text: {entry['text']}" for entry in related_texts])

    return recent_history, related_history


# 세션별 메모리 생성 (최근 3개의 대화만 유지)
def create_memory_for_session(session_id):
    memory_key = f"chat_history_{session_id}"
    return ConversationBufferMemory(memory_key=memory_key, k=3)


# memory = create_memory_for_session(session_id)

# 검색 도구 정의 (검색된 상위 3개 대화 내용 사용)
def search_history_tool(session_id, query_text):
    print(f"Searching history for session: {session_id}, query: {query_text}")
    result = retrieve_conversations(session_id, query_text)
    if result:
        return result  # 상위 3개의 대화 반환
    else:
        return "No relevant information found in the chat history."

def rag_tool(query):
    # 1. 질문에 맞는 문서 검색 (이 단계에서 검색만 수행)
    retrieved_docs = retrieve_documents(query)
    
    # 검색된 문서를 반환 (생성은 LLM이 담당)
    return retrieved_docs


# 프롬프트 템플릿 정의 (한국어로 변경)
prompt_template = PromptTemplate.from_template(
    """
    당신은 유용한 조수입니다. 가능한 한 신속하고 정확하게 답변을 제공하세요.
    - 질문에 대한 답변을 바로 알고 있으면, 대화 기록이나 검색 없이 바로 답변해 주세요.
    - 질문의 답변이 최근 대화에 관련되어 있다면, 최근 대화를 참조하여 답변을 생성하세요.
    - 질문과 관련된 과거 대화가 유용하다면, 관련 대화를 참고하여 답변을 작성하세요.
    - 과학적이거나 복잡한 정보에 대한 질문일 경우에만, 검색된 문서를 참조하여 답변을 생성하세요.

    [최근 대화 기록]
    {recent_history}

    [관련된 과거 질문들]
    {related_history}

    [검색된 문서들]
    {retrieved_docs} (검색된 문서는 과학적 질문에만 사용됩니다)

    사용자의 질문: {user_input}

    조수:
    """
)


# Tool을 통한 검색 정의 (Pinecone과 메모리 사용)
tools = [
    Tool(
        name="SearchHistory",
        func=lambda query: search_history_tool(session_id, query),
        description="Searches the chat history based on the user's question."
    ),
    Tool(
        name="RAGTool",
        func=lambda query: rag_tool(query),  # RAG 툴을 통한 문서 검색 기능
        description="Searches documents and retrieves relevant passages using RAG."
    )
]

# 에이전트 실행 (툴을 필요할 때만 자동으로 호출)
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",  # 필요할 때만 툴을 사용하도록 설정
    verbose=True,
    handle_parsing_errors=True  # 파싱 오류 발생 시 다시 시도하도록 설정
)
# 대화 정보를 프롬프트에 제공하는 함수
# 대화 히스토리를 포맷하는 함수
def format_conversation_history(recent_texts, related_texts):
    # 최근 대화 형식 지정
    recent_history = "\n".join([f"Timestamp: {entry.get('timestamp', 'Unknown')}, Role: {entry.get('role', 'Unknown')}, Text: {entry.get('text', 'No text available')}" for entry in recent_texts])
    
    # 관련 대화 형식 지정
    related_history = "\n".join([f"Timestamp: {entry.get('timestamp', 'Unknown')}, Role: {entry.get('role', 'Unknown')}, Text: {entry.get('text', 'No text available')}" for entry in related_texts])

    return recent_history, related_history


def generate_prompt(recent_texts, related_texts, user_input, retrieved_docs=None):
    recent_history, related_history = format_conversation_history(recent_texts, related_texts)
    
    # retrieved_docs가 없는 경우 빈 문자열로 설정
    if retrieved_docs is None:
        retrieved_docs = ""

    # 프롬프트 템플릿을 이용해 최종 프롬프트 생성
    prompt = prompt_template.format(
        recent_history=recent_history,
        related_history=related_history,
        retrieved_docs=retrieved_docs,  # retrieved_docs 값 추가
        user_input=user_input
    )
    
    return prompt

session_id = "session1042"


# 챗봇 함수 (최근 대화와 검색된 상위 3개의 대화 내용 사용)
# # 챗봇 함수 (사용자 입력과 AI 응답 저장)
# async def chatbot(session_id, user_input):
#     # 최근 대화 및 관련 대화 가져오기
#     recent_texts, related_texts = retrieve_conversations(session_id, user_input)
    
#     # 프롬프트 생성
#     prompt = generate_prompt(recent_texts, related_texts, user_input)
    
#     # 에이전트에 사용자 입력 전달 (툴을 필요로 할 때만 사용)
#     response = agent.run(prompt)
    
#     # 대화 내용 저장
#     await async_store_conversation(session_id, user_input, "user")
#     await async_store_conversation(session_id, response, "ai")
    
#     return response

# session_id = "session1003"


# def chatbot_sync(session_id, user_input):
#     # 비동기 챗봇 호출을 동기적으로 처리
#     return asyncio.run(chatbot(session_id, user_input))

# # 대화 상태를 초기화하는 함수
# def initialize_chat_state():
#     if "messages" not in st.session_state:
#         st.session_state["messages"] = []

# # 사용자가 보낸 메시지를 처리하는 함수
# def handle_user_message(user_input):
#     # 사용자가 입력한 메시지를 저장
#     st.session_state.messages.append({"role": "user", "content": user_input})

#     # 비동기 AI 답변을 동기 함수로 처리
#     response = chatbot_sync(session_id, user_input)
    
#     # AI의 응답을 저장
#     st.session_state.messages.append({"role": "bot", "content": response})

# # 메인 애플리케이션 실행
# def main():
#     st.title("Streamlit Chat - 베이스라인")

#     # 대화 상태 초기화
#     initialize_chat_state()

#     # 사용자 입력
#     user_input = st.text_input("여기에 메시지를 입력하세요:")

#     if user_input:
#         handle_user_message(user_input)

#     # 이전 대화 기록을 화면에 표시
#     for chat in st.session_state.messages:
#         if chat["role"] == "user":
#             message(chat["content"], is_user=True)
#         else:
#             message(chat["content"])

# # 실행
# if __name__ == "__main__":
#     main()

import chainlit as cl
import asyncio

# 비동기 챗봇 함수
async def chatbot(session_id, user_input):
    # 최근 대화 및 관련 대화 가져오기
    recent_texts, related_texts = retrieve_conversations(session_id, user_input)
    
    # 프롬프트 생성
    prompt = generate_prompt(recent_texts, related_texts, user_input)
    
    # 에이전트에 사용자 입력 전달 (툴을 필요로 할 때만 사용)
    response = agent.run(prompt)
    
    # 대화 내용 저장
    await async_store_conversation(session_id, user_input, "user")
    await async_store_conversation(session_id, response, "ai")
    
    return response

# 동기 챗봇 호출 함수 (비동기 호출을 동기 처리)
def chatbot_sync(session_id, user_input):
    return asyncio.run(chatbot(session_id, user_input))

# 사용자가 보낸 메시지를 처리하는 함수
@cl.on_message
async def main(user_message):


    # 사용자가 입력한 메시지 처리 (Message 객체 속성에 접근)
    user_input = user_message.content

    # 비동기 챗봇 호출
    response = await chatbot(session_id, user_input)

    # Chainlit을 통해 AI의 응답을 사용자에게 전송
    await cl.Message(content=response).send()
