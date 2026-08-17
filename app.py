import uuid
import tempfile
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from pdf_ingest import get_store
from graph import graph 

st.set_page_config(page_title="AI Research Assistant", layout = "wide")
st.title("AI Research Assistant")

# ── session state: stable user_id/session_id per browser session ──────
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "doc_id" not in st.session_state:
    st.session_state.doc_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # [(role, text), ...] for display only

store = get_store()

# ── sidebar: upload ─────────────────────────────────────────────────
with st.sidebar:
    st.header("Upload a document")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded_file is not None:
        if st.button("Ingest document"):
            with st.spinner("Processing document..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                doc_id = store.ingest(tmp_path, user_id=st.session_state.user_id, filename=uploaded_file.name)
                st.session_state.doc_id = doc_id
            st.success(f"Ingested: {uploaded_file.name}")

    st.divider()
    st.subheader("Your documents")
    docs = store.list_user_docs(st.session_state.user_id)
    if docs:
        for d in docs:
            st.write(f"📄 {d['filename']}")
    else:
        st.caption("No documents uploaded yet.")

# ── main: chat ──────────────────────────────────────────────────────
for role, text in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(text)

query = st.chat_input("Ask a question...")

if query:
    st.session_state.messages.append(("user", query))
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Researching..."):
            result = graph.invoke({
                "query": query,
                "user_id": st.session_state.user_id,
                "session_id": st.session_state.session_id,
                "doc_id": st.session_state.doc_id,
            },
            config={"configurable":{"thread_id": st.session_state.session_id}}
            )
            answer = result["final_report"]
        st.markdown(answer)

    st.session_state.messages.append(("assistant", answer))