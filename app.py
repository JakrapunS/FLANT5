import streamlit as st

from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain import HuggingFaceHub
from langchain.chains import RetrievalQA
import time



st.title('Document Question Answering')

st.subheader("🛠 How It Works 🛠")

st.write("""
Our Document Question Answering App is a combination of language models with vector databases. Here's an overview workflow, as visualized in the provided diagram:
""")

st.image("diagram.png", caption="Flow Diagram of the Document Question Answering App.", use_column_width=True)

st.subheader("🔧 Preprocessing 🔧")

st.write("""
Before diving into the real-time workings of our app, there's some behind-the-scenes magic that happens:

1. **Document Preprocess**: We start with extracting content from PDF files and then split this content into manageable chunks.
2. **Create Embedding**: Each chunk undergoes a transformation using the HuggingFace Inference API, turning them into sentence embeddings.
3. **Vector Database Population**: These meaningful embeddings are then stored securely in the Qdrant cloud vector database, ready for swift retrieval later on.
""")

st.subheader("🚀 Application Workflow 🚀")

st.write("""
When you interact with our app, here's the journey your query undertakes:

1. **User Query Submission**: It all starts with you! Submit a question, and the gears start turning.
2. **Embedding Extraction**: Your question is promptly transformed into embeddings using the HuggingFace Inference API.
3. **Vector Database Search**: The Qdrant vector database, housing knowledge from the Australian Energy Policy Review 2023 (and more), is queried with these embeddings to find matches.
4. **Language Model Integration**: To elevate the quality of answers, the 'google/flan-t5-large' model from HuggingFaceHub is invoked.
5. **RetrievalQA Mechanism**: LangChain's RetrievalQA plays the pivotal role of fetching the most contextually aligned data from the vector database.
6. **Answer Delivery**: And there you have it! A comprehensive answer is presented, often with references to the source pages for further exploration.

This intricate dance of operations ensures that every query receives a well-informed and contextually rich response.
""")

st.write("""
🐢 **A Fun Caution**:
Given our commitment to using top-notch free resources, sometimes our app prefers a leisurely stroll over a sprint. Think of it as a relaxed turtle 🐢, not a hurried hare 🐇. But as they say, good things come to those who wait!
""")


# get embeddings
def get_enbeddings():
    
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=st.secrets["hf_api"],
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    return embeddings


# vectordb

def get_qdrant(embeddings):
    client = QdrantClient(
    url=st.secrets["qdrant_url"],
    api_key=st.secrets["qdrant_api"],
    prefer_grpc=True,
    )
    collection_name = "test_collection"
    #from langchain
    qdrant = Qdrant(client, collection_name, embeddings)

    return qdrant
#create LLM
def create_llm(vector_db):
    llm=HuggingFaceHub(repo_id="google/flan-t5-large", 
    model_kwargs={"temperature":1, "max_length":10000},huggingfacehub_api_token = st.secrets['hf_api'])
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="map_reduce",
    retriever=vector_db.as_retriever(search_kwargs={"k": 5},search_type="mmr"),return_source_documents=True)

    return qa

# run qa
def run_qa(qa,query):
    result = qa({"query": query})
    source_page=[]
    for i in result['source_documents']:
        source_page.append(i.metadata['page']+2)
    #text = f"{result['result']} Source pages: {source_page}"
    return result['result'], source_page

# run 
embeddings = get_enbeddings()
vectordb = get_qdrant(embeddings)
qa = create_llm(vectordb)


# Input a question
predefined_questions = [
    "Explain in details about natural gas usage situation?",
    "How does economic growth influence the demand for electricity?",
    "Can you provide a summary of solar energy developments and trends from 2000 to 2021?",
    "Can you provide a summary of electricity demand and trends from 2000 to 2021?"
]

st.subheader("💡 Query to Answer💡")
st.write("Sample questions")

# Check if 'selected_question' exists in the session state
if 'selected_question' not in st.session_state:
    st.session_state.selected_question = ""

for q in predefined_questions:
    if st.button(q):
        st.session_state.selected_question = q

# Text input for the question
question = st.text_input("Your question:", value=st.session_state.selected_question)



if st.button("Ask"):
    print(question)
    loading_message = st.empty()
    start_time = time.time()

    with st.spinner('Wait for it...'):
        answer,source_page = run_qa(qa,question)
    
    end_time = time.time()
    loading_message.text(f"Loaded in {round(end_time - start_time,2)} seconds")
    st.write(f"Answer: {answer}")
    st.write(f"Source Pages: {source_page}")

st.markdown("For more details, check out this [Australia 2023: Energy Policy Review](https://iea.blob.core.windows.net/assets/02a7a120-564b-4057-ac6d-cf21587a30d9/Australia2023EnergyPolicyReview.pdf).")
