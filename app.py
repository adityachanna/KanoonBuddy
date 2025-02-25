import os
import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_community.tools import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from pydantic import BaseModel, Field
from typing import List, Dict
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph
import json  
load_dotenv()
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')
os.environ['GOOGLE_API_KEY'] = os.getenv('GEMINI_API_KEY')
os.environ['TAVILY_API_KEY'] = os.getenv('SEARCH')
embeddings = OllamaEmbeddings(
    model="snowflake-arctic-embed2:latest",
)
new_vector_store = FAISS.load_local(
    "vectorstore", embeddings, allow_dangerous_deserialization=True
)
similarity_threshold_retriever = new_vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 3,
        "score_threshold": 0.3
    }
)
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

SYS_PROMPT = """You are an expert grader assessing relevance of a retrieved document to a user question.
                Follow these instructions for grading:
                  - If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
                  - Your grade should be either 'yes' or 'no' to indicate whether the document is relevant to the question or not."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYS_PROMPT),
        ("human", """Retrieved document:
                     {document}
                     User question:
                     {question}
                  """),
    ]
)

doc_grader = (grade_prompt | structured_llm_grader)

prompt = """You are an assistant for answering legal queries.  
Use the following pieces of retrieved context to answer the question.  
If no context is present or if you don't know the answer, just say that you don't know the answer.  
Do not make up the answer unless it is there in the provided context.  
Give a detailed and to-the-point answer with regard to the question.  

If the query is related to Indian law, provide legal information strictly under Indian law.  
If the query is ambiguous or pertains to laws outside India, prioritize answering according to Indian law first.  
If necessary, general legal principles can be provided later in the conversation.  

Question:  
{question}  
Context:  
{context}  
Answer:  
         """
prompt_template = ChatPromptTemplate.from_template(prompt)

gemini = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0.2)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

qa_rag_chain = (
    {
        "context": (itemgetter('context') | RunnableLambda(format_docs)),
        "question": itemgetter('question')
    }
    | prompt_template
    | gemini
    | StrOutputParser()
)

legal_question_rewriter = (
    ChatPromptTemplate.from_messages([
        ("system", """Act as a legal query rewriter and perform the following tasks:
                 - Convert the given legal question into a clearer, more precise version that improves retrieval quality.
                 - Ensure the rewritten question is optimized for legal research, particularly for Indian law.
                 - Retain the core legal intent while improving clarity and specificity.
                 - If the question is ambiguous, reframe it to make it more legally precise.
             """),
        ("human", """Here is the initial legal query:
                     {question}
                     Reformulate it into a more precise and optimized legal question.
                  """),
    ])
    | llm
    | StrOutputParser()
)

search_tool = TavilySearchResults(
    api_key=os.getenv('SEARCH'), 
    max_results=5,
    search_depth="advanced",  
    include_domains=[
        'indiankanoon.org',
        'lawrato.com',
        'kaanoon.com',
        'legalserviceindia.com',
        'advocatekhoj.com',
        'legalbites.in',
        'legalservicesindia.com',
        'indiacode.nic.in',
        'scconline.com',
        'manupatra.com'
    ]
)

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: The current question.
        generation: LLM response generation.
        web_search_needed: Flag indicating whether web search is needed ('yes' or 'no').
        documents: List of retrieved documents.
        intermediate_steps: List of intermediate steps (for debugging and display).
        chat_history: List of chat messages (question, answer pairs).

    """
    question: str
    generation: str
    web_search_needed: str
    documents: List[Document]
    intermediate_steps: List[str] 
    chat_history: List[Dict[str, str]] 
def retrieve(state):
    """
    Retrieve documents
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): New key added to state, documents - that contains retrieved context documents
    """
    print("---RETRIEVAL FROM VECTOR DB---")
    question = state["question"]

    documents = similarity_threshold_retriever.invoke(question)
    return {"documents": documents, "question": question, "intermediate_steps": state.get("intermediate_steps", []) + ["Retrieved documents from vector DB"]}

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    by using an LLM Grader.
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    web_search_needed = "No"
    
    if documents and len(documents) > 0:
        for d in documents:
            if not d.page_content or len(d.page_content.strip()) < 10:
                print("---EMPTY DOCUMENT FOUND---")
                web_search_needed = "Yes"
                continue
                
            try:
                score = doc_grader.invoke(
                    {"question": question, "document": d.page_content}
                )
                grade = score.binary_score
                if grade.lower() == "yes":
                    print("---GRADE: DOCUMENT RELEVANT---")
                    filtered_docs.append(d)
                else:
                    print("---GRADE: DOCUMENT NOT RELEVANT---")
                    web_search_needed = "Yes"
            except Exception as e:
                print(f"---ERROR GRADING DOCUMENT: {str(e)}---")
                web_search_needed = "Yes"
    else:
        print("---NO DOCUMENTS RETRIEVED---")
        web_search_needed = "Yes"
        
    return {
        "documents": filtered_docs, 
        "question": question, 
        "web_search_needed": web_search_needed,
        "intermediate_steps": state.get("intermediate_steps", []) + ["Graded documents for relevance"]
    }
def rewrite_query(state):
    """
    Rewrite the query to produce a better question.
    """
    print("---REWRITE QUERY---")
    question = state["question"]
    documents = state["documents"]
    
    try:
        search_query = question
        if len(question) > 300:
            search_query = " ".join(question.split()[:30])
        better_question = legal_question_rewriter.invoke({"question": search_query})
        print(f"---REWRITTEN QUERY: {better_question[:100]}...---")
        if len(better_question) > 300 and "?" in better_question:
            lines = better_question.split("\n")
            for line in lines:
                if "?" in line and len(line) < 300:
                    better_question = line
                    break
    except Exception as e:
        print(f"---ERROR REWRITING QUERY: {str(e)}---")
        better_question = question  
    return {"documents": documents, "question": better_question, "intermediate_steps": state.get("intermediate_steps", []) + ["Rewrote query"]}
def web_search(state):
    """
    Web search based on the re-written question.
    """
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]
    
    try:
        search_query = question
        if len(question) > 200:
            if "?" in question:
                for line in question.split("\n"):
                    if "?" in line:
                        search_query = line.strip()
                        break
            else:
                search_query = question.split("\n")[0]
        
        print(f"---SEARCHING WEB FOR: {search_query}---")
        
        search_results = search_tool.invoke(search_query)
        
        web_docs = []
        if isinstance(search_results, list) and len(search_results) > 0:
            print(f"---FOUND {len(search_results)} SEARCH RESULTS---")
            
            for result in search_results:
                if isinstance(result, dict):
                    title = result.get("title", "No Title")
                    content = result.get("content", "")
                    url = result.get("url", "")
                    
                    if content and len(content.strip()) > 50:
                        formatted_content = f"SOURCE: {title}\nURL: {url}\n\nCONTENT:\n{content}"
                        web_docs.append(Document(page_content=formatted_content))
        
        if web_docs:
            documents.extend(web_docs)
            print(f"---ADDED {len(web_docs)} WEB DOCUMENTS TO CONTEXT---")
        else:
            print("---NO USABLE WEB RESULTS FOUND---")
            documents.append(Document(page_content="Web search did not return relevant information for this query."))
    
    except Exception as e:
        error_msg = str(e)
        print(f"---WEB SEARCH ERROR: {error_msg}---")
        documents.append(Document(page_content=f"Error during web search: {error_msg}. Using only local knowledge base."))
    
    return {"documents": documents, "question": question, "intermediate_steps": state.get("intermediate_steps", []) + [f"Performed web search for: {search_query}"]}
def generate_answer(state):
    """
    Generate answer from context document using LLM
    """
    print("---GENERATE ANSWER---")
    question = state["question"]
    documents = state["documents"]
    intermediate_steps = state.get("intermediate_steps", [])
    
    try:
        generation = qa_rag_chain.invoke({"context": documents, "question": question})
        intermediate_steps.append("Generated answer using RAG")  
    except Exception as e:
        print(f"---ERROR GENERATING ANSWER: {str(e)}---")
        generation = "I apologize, but I encountered an error while generating an answer. Please try rephrasing your question."
        intermediate_steps.append(f"Error generating answer: {str(e)}")
    
    chat_history = state.get("chat_history", [])
    chat_history.append({"question": question, "answer": generation})
    
    return {
        "documents": documents, 
        "question": question, 
        "generation": generation,
        "intermediate_steps": intermediate_steps,  
        "chat_history": chat_history 
    }
def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.
    """
    print("---ASSESS GRADED DOCUMENTS---")
    web_search_needed = state["web_search_needed"]
    
    if web_search_needed == "Yes":
        print("---DECISION: SOME or ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, REWRITE QUERY---")
        return "rewrite_query"
    else:
        print("---DECISION: GENERATE RESPONSE---")
        return "generate_answer"

agentic_rag = StateGraph(GraphState)
agentic_rag.add_node("retrieve", retrieve)
agentic_rag.add_node("grade_documents", grade_documents)
agentic_rag.add_node("rewrite_query", rewrite_query)
agentic_rag.add_node("web_search", web_search)
agentic_rag.add_node("generate_answer", generate_answer)
agentic_rag.set_entry_point("retrieve")
agentic_rag.add_edge("retrieve", "grade_documents")
agentic_rag.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {"rewrite_query": "rewrite_query", "generate_answer": "generate_answer"},
)
agentic_rag.add_edge("rewrite_query", "web_search")
agentic_rag.add_edge("web_search", "generate_answer")
agentic_rag.add_edge("generate_answer", END)
agentic_rag = agentic_rag.compile()
def answer_legal_query_with_steps(query):
    print(f"\n=== ANSWERING QUERY: {query} ===\n")
    initial_state = {
        "question": query,
        "intermediate_steps": [],
        "chat_history": []
    }
    response = agentic_rag.invoke(initial_state)
    print("\n=== RESPONSE ===\n")
    print(response.get("generation", "No answer generated"))
    return response
st.title("KanoonBuddy")
st.markdown("Welcome to KanoonBuddy, your legal assistant. Ask any legal query and I will try to help you with the answer.")
st.markdown("Please note that the information provided here is for informational purposes only and should not be considered legal advice.")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
for message in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(message["question"])
    with st.chat_message("assistant"):
        st.write(message["answer"])

user_query = st.chat_input("Enter your legal query here...")

if user_query:
    with st.chat_message("user"):
        st.write(user_query)
    with st.spinner("Processing your query..."):
        response = answer_legal_query_with_steps(user_query)
        if response and "generation" in response and "chat_history" in response:
            st.session_state.chat_history = response["chat_history"]
            with st.chat_message("assistant"):
                st.write(response["generation"])
        else:
            with st.chat_message("assistant"):
                st.write("I encountered an error processing your request.  Please try again.")
    with st.expander("Show Intermediate Steps"):
        for step in response.get("intermediate_steps", []):
            st.text(step)