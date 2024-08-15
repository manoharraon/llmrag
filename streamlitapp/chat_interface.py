import streamlit as st

class ChatInterface:
    def __init__(self):
        if 'messages' not in st.session_state:
            st.session_state.messages = []

    def add_user_message(self, message):
        st.session_state.messages.append({"role": "user", "content": message})

    def add_ai_message(self, message, source=None, documents=None):
        st.session_state.messages.append({
            "role": "assistant", 
            "content": message,
            "source": source,
            "documents": documents
        })

    def display_chat(self):
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])
                    if message.get("source"):
                        st.write(f"Source: {message['source']}")
                    if message.get("documents"):
                        st.write("Relevant Documents:")
                        for doc in message["documents"]:
                            st.write(f"- {doc['file_name']}, Page: {doc['page_number']}")

    def clear_chat(self):
        st.session_state.messages = []