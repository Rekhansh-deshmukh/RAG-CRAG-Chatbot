import streamlit as st
import pandas as pd
from src.crag import build_graph
from src.vectorstore import build_vectorstore
from src.eval import run_comprehensive_eval, evaluate_batch

st.set_page_config(page_title="CRAG Metrics Dashboard", layout="wide")

@st.cache_resource
def get_crag_app():
    return build_graph()

app = get_crag_app()

st.title("🛠️ Corrective RAG + Metrics Dashboard")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar
with st.sidebar:
    st.header("1. Data Ingestion")
    url_input = st.text_input("URL to Index:")
    if st.button("Build Vector Index"):
        with st.spinner("Indexing..."):
            build_vectorstore([url_input])
            st.success("Vector DB Updated!")
            
    st.divider()
    st.header("2. Performance Analytics")
    if st.button("Generate Session Report"):
        if st.session_state.history:
            with st.spinner("Analyzing history..."):
                st.session_state.eval_report = evaluate_batch(st.session_state.history)
        else:
            st.warning("Start a conversation first!")

    if "eval_report" in st.session_state:
        st.write("### Aggregate Metrics")
        avg_rag = st.session_state.eval_report["Total RAG Score"].mean()
        st.metric("Mean RAG Score", f"{avg_rag}/10")
        st.dataframe(st.session_state.eval_report)

# Main Chat Interface
chat_col, debug_col = st.columns([2, 1])

with chat_col:
    st.subheader("Interactive CRAG Chat")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Query the system..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            hist_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-3:]])
            inputs = {"question": prompt, "chat_history": hist_str, "documents": [], "is_safe": True}
            
            with st.spinner("CRAG Pipeline..."):
                output = app.invoke(inputs)
                answer = output["generation"]
                docs = output["documents"]
                st.markdown(answer)
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.session_state.history.append({"question": prompt, "answer": answer, "context": docs})

with debug_col:
    st.subheader("Real-time Evaluation Metrics")
    if st.session_state.history:
        last = st.session_state.history[-1]
        with st.container(border=True):
            with st.spinner("Calculating metrics..."):
                res = run_comprehensive_eval(last['question'], last['answer'], last['context'])
                
                st.write("**Total RAG Score**")
                st.progress(res.get('Total RAG Score', 0) / 10)
                st.caption(f"Score: {res.get('Total RAG Score')}/10")
                
                st.write("**Faithfulness (Groundedness)**")
                st.progress(res.get('Faithfulness', 0) / 10)
                
                st.write("**Answer Relevancy**")
                st.progress(res.get('Answer Relevancy', 0) / 10)
                
                st.write("**Context Precision**")
                st.progress(res.get('Context Precision', 0) / 10)
                
                st.info(f"Sources utilized: {len(last['context'])} chunks")
    else:
        st.info("Metrics will appear here after your first interaction.")
