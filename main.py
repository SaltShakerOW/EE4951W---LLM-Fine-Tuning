import streamlit as st
from ollama import chat
from ollama import ChatResponse

st.title("LLM Fine Tuning GUI")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input(" "):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    stream = chat(
        model='llama3.1:8b',
        messages=[{'role': 'user', 'content': prompt}],
        stream=True,
    )


    # Display assistant response in chat message container
    placeholder = st.empty()
    response = {"role": "assistant", "content": ""}
    with st.chat_message("assistant"):
        message = ""
        for chunk in stream:
            message += chunk.message.content
            placeholder.write(message)
            response["content"] += chunk.message.content
    # Add assistant response to chat history
    st.session_state.messages.append(response)