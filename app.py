import uuid
import requests
import streamlit as st

# Replace with your actual deployed FastAPI URL once you have it
API_URL = "https://agentic-research-assistance-u7fm.onrender.com"

st.set_page_config(page_title="AI Research Assistant", layout="wide")
st.title("AI Research Assistant")

# ── session state: stable user_id/session_id per browser session ──────
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "doc_id" not in st.session_state:
    st.session_state.doc_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── sidebar: upload ─────────────────────────────────────────────────
with st.sidebar:
    st.header("Upload a document")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded_file is not None:
        if st.button("Ingest document"):
            with st.spinner("Processing document..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                data = {"user_id": st.session_state.user_id}
                resp = requests.post(f"{API_URL}/upload", files=files, data=data)

                if resp.status_code == 200:
                    st.session_state.doc_id = resp.json()["doc_id"]
                    st.success(f"Ingested: {uploaded_file.name}")
                else:
                    st.error(f"Upload failed: {resp.text}")

    st.divider()
    st.subheader("Your documents")
    try:
        docs_resp = requests.get(f"{API_URL}/documents/{st.session_state.user_id}")
        docs = docs_resp.json().get("documents", []) if docs_resp.status_code == 200 else []
    except requests.exceptions.RequestException:
        docs = []

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
            resp = requests.post(f"{API_URL}/query", json={
                "query": query,
                "user_id": st.session_state.user_id,
                "session_id": st.session_state.session_id,
                "doc_id": st.session_state.doc_id,
            })

            if resp.status_code == 200:
                answer = resp.json()["final_report"]
            else:
                answer = f"Error: {resp.text}"

        st.markdown(answer)

    st.session_state.messages.append(("assistant", answer))