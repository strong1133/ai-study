
from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic import hub
from langchain_classic.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import streamlit as st;

load_dotenv()

def get_ai_message(query):

    embedding = OpenAIEmbeddings(model='text-embedding-3-large')

    index_name = "tax-makrdown-index"

    database = PineconeVectorStore.from_existing_index( index_name=index_name, embedding=embedding)
    llm = ChatOpenAI(model='gpt-4o')

    prompt = hub.pull("rlm/rag-prompt")

    retriever = database.as_retriever()

    dictionary=["사람을 나타내는 표현 -> 거주자"]

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs={"prompt":prompt}
    )
    
    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변결할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        사전: {dictionary}

        질문: {{query}}
    """)

    dictionary_chain = prompt | llm | StrOutputParser()

    tax_chain = {"query": dictionary_chain} | qa_chain
    ai_message = tax_chain.invoke({"query":query})
    return ai_message['result']

st.set_page_config(page_title="소득세 챗봇", )

st.title("소득세 챗봇")
st.caption("소득세에 관련된 모든것을 답해드립니다.")

if 'message_list' not in st.session_state:
    st.session_state.message_list=[]
    
    
for message in  st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])


if user_question := st.chat_input(placeholder="소득세에 관련된 궁금한 내용들을 말씀해주세요!"):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append(
        {
            "role":"user",
            "content":user_question
        }
    )
    
    with st.spinner("답변을 생성하는 중입니다"):
        ai_msg = get_ai_message(user_question)
        with st.chat_message("ai"):
            st.write(ai_msg)
        st.session_state.message_list.append(
            {
                "role":"ai",
                "content":ai_msg
            }
        )
        
    
    
