import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
#text의 token개수를 새기 위한 라이브러리
import tiktoken

from loguru import logger

from langchain_community.llms import Bedrock
from langchain.globals import set_verbose

#Memory를 가지고 있는 Chain
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import ConversationChain
#from langchain.chat_models import ChatOpenAI

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader


#Text Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
#Huggingface Embedding 모델
from langchain_community.embeddings import HuggingFaceEmbeddings

#몇개까지의 대화를 메모리에 넣어줄 것인지 결정하는 Buffer
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS 

from langchain_community.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory
#chat_model = ChatOpenAI()

import streamlit as st

def bedrock_chatbot() : 

    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

    #print(aws_access_key_id)

    bedrock_llm = Bedrock(
        #credentials_profile_name = 'default',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name = 'us-east-1',
        model_id= 'anthropic.claude-v2:1',
        model_kwargs= {
            "prompt": "\n\nHuman:<prompt>\n\nAssistant:",
            "temperature": 0.5,
            "top_p": 1,
            "top_k": 250,
            "max_tokens_to_sample": 512
        }
    )

    return bedrock_llm

#token 개수를 기준으로 Text를 split
def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

#각각의 파일들을 로드해서 넘겨줌
def get_text(docs):
    doc_list = []
    for doc in docs:
        file_name = doc.name # doc 객체의 이름을 파일 이름으로 사용 
        with open(file_name, "wb") as file:
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")

        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)        
    return doc_list

#text를 chunk단위로 변환하여 return
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function= tiktoken_len
    )

    chunks = text_splitter.split_documents(text)
    return chunks

#chunk를 vectordb에 embedding하여 저장
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name = "jhgan/ko-sroberta-multitask",
        model_kwargs = {'device': 'cpu'},
        encode_kwargs = {'normalize_embeddings': True}
    )
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_conversation_chain(vectorstore):
    llm = bedrock_chatbot()
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type = 'mmr', vervose = True),
        #chat_history라는 key 값을 가진 메모리를 저장(이전 대화를 기억), output_key = 'answer' 답변에 해당하는 내용만 담겠다는 의미
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose = True
    )
    return conversation_chain

def buff_memory() : 
    llm = bedrock_chatbot()
    memory = ConversationBufferMemory(llm = llm, max_token_limit=200)
    return memory

def cnvs_chain(input_text, memory) : 
    chain_data = bedrock_chatbot()
    cnvs_chain = ConversationChain(llm = chain_data, memory = memory)
    chat_reply = cnvs_chain.predict(input = input_text)
    return chat_reply

def get_basic_chatbot(memory) : 
    chain_data = bedrock_chatbot()
    conversation_chain = ConversationChain(llm = chain_data, memory = memory)
    #chat_reply = conversation_chain.predict(input = input_text)
    return conversation_chain


# 초기 챗봇 설정 및 기본 챗봇 설정 로직
def setup_chatbot(uploaded_files, memory):
    try:
        if uploaded_files:
            # 파일 처리 로직
            files_text = get_text(uploaded_files)
            text_chunks = get_text_chunks(files_text)
            vectorstore = get_vectorstore(text_chunks)
            return get_conversation_chain(vectorstore)
        else:
            # 업로드된 파일이 없을 경우 기본 챗봇 반환
            return get_basic_chatbot(memory)
    except Exception as e:
        logger.error(f"Error setting up the chatbot: {e}")
        return None

st.title("_:orange[HTC-ChatGPT]_ :books:")

#현재 대화 저장소
if "conversation" not in st.session_state:
    st.session_state.conversation = None

#히스토리 저장소
if "chat_history" not in st.session_state:
    st.session_state.chat_history = None

#메모리 저장소
st.session_state.memory = buff_memory()

#set_verbose(True)

#화면 사이드바 구성
with st.sidebar:
    uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx', 'pptx'], accept_multiple_files=True)
    process = st.button("Process")

#버튼을 누를 시
if process: 
    if uploaded_files:
        #upload 파일을 text변환
        files_text = get_text(uploaded_files)
        #text를 chunk로 나눔
        text_chunks = get_text_chunks(files_text)
        #vectorstore로 chunk를 vector화
        vectorstore = get_vectorstore(text_chunks)

        #vectorstore에서 대화를 chain으로 구성
        st.session_state.conversation = get_conversation_chain(vectorstore)
    else:
        # 업로드된 파일이 없으면 기본 챗봇 사용
        st.session_state.conversation = get_basic_chatbot(st.session_state.memory)

#챗봇에 처음 들어갔을 때 인사말 삽입
if 'messages' not in st.session_state:
    st.session_state['messages'] = [{"role": "assistant", "content": "안녕하세요! HTC CHATBOT 입니다. 1)문서업로드 2) Process 버튼 클릭 후 3) 업로드한 문서에 대해 궁금하신 것이 있으면 질문해주세요."}]

#페이지에 메시지가 입력이 될 때마다 화면상에 해당 컨텐츠를 표기
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#대화내용 저장을 통해 메모리화
history = StreamlitChatMessageHistory(key="chat_messages")

# 챗봇 대화 처리
if query := st.chat_input("질문을 입력해주세요."):
    st.session_state.messages.append({"role": "user", "content": query})

    #질문을 표기
    with st.chat_message("user"):
        st.markdown(query)

    #답변을 표기
    with st.chat_message("assistant"):

        # 질문 시점에 업로드된 파일 확인 및 챗봇 설정
        #uploaded_files = st.session_state.get('uploaded_files', [])
        if uploaded_files :
            st.session_state.conversation = setup_chatbot(uploaded_files, st.session_state.memory)
            chain = st.session_state.conversation
            #대답 생성 중간 로딩 표기
            with st.spinner("Thinking..."):
                result = chain({"question": query})
                response = result['answer']

                #with get_openai_callback() as cb:
                st.session_state.chat_history = result['chat_history']

                source_documents = result['source_documents']

                st.markdown(response)
                #참고한 소스 확인
                with st.expander("참고 문서 확인"): 
                    for doc in source_documents:
                        st.markdown(f"{doc.metadata['source']}: {doc.page_content}")

            st.session_state.messages.append({"role": "assistant", "content": response})

        else:
            chat_response = cnvs_chain(input_text= query, memory= st.session_state.memory)
            st.markdown(chat_response)
            
            st.session_state.messages.append({"role": "assistant", "content": chat_response})





