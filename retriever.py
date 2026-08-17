from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import SQLChatMessageHistory

from pdf_ingest import get_store

CHAT_DB = "sqlite:///chat_history.db"  # persists chat history to disk — survives restarts


class Retriever:
    """Retrieves relevant chunks for a user's query, using persistent chat
    history so follow-up questions resolve correctly."""

    def __init__(self, k: int = 4, chat_db: str = CHAT_DB):
        self.store = get_store()
        self.k = k
        self.chat_db = chat_db
        self.llm = ChatOpenAI(model="gpt-4o-mini")

    def get_chat_history(self, user_id: str, session_id: str) -> SQLChatMessageHistory:
        return SQLChatMessageHistory(session_id=f"{user_id}:{session_id}", connection=self.chat_db)

    def _vector_retriever(self, user_id: str, doc_id: Optional[str] = None):
        filter_dict = {"user_id": user_id}
        if doc_id:
            filter_dict = {"$and": [{"user_id": user_id}, {"doc_id": doc_id}]}
        return self.store.vectorstore.as_retriever(search_kwargs={"k": self.k, "filter": filter_dict})

    def query(self, query: str, user_id: str, session_id: str, doc_id: Optional[str] = None) -> list:
        history = self.get_chat_history(user_id, session_id)

        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given the chat history and the latest user question, "
                       "rewrite it as a standalone question. Do not answer it."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        history_aware_retriever = create_history_aware_retriever(
            self.llm, self._vector_retriever(user_id, doc_id), contextualize_prompt
        )

        results = history_aware_retriever.invoke({
            "input": query,
            "chat_history": history.messages,
        })

        history.add_user_message(query)
        # history.add_ai_message(answer)  # call once your writer node produces the final answer

        return results
