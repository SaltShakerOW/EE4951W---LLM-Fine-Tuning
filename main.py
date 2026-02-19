import streamlit as st
from llama_cpp import Llama

#Setup Section
st.set_page_config(
    page_title="LLM Fine Tuning Demo",
    layout="centered",
)

model_options = {
    "Vanilla LLama3.1-8B": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    "Custom Tuned Model": "custom_tuned_model.gguf",
}

system_prompt = "You are a helpful assistant. Respond to the user's messages as concisely as possible."

@st.cache_resource()
def load_model(model_path: str):
    return Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=2048,
        verbose=False,
    )

def init_messages():
    return[{"role": "system", "content": system_prompt}]

#UI Section
st.title("LLM Fine Tuning Demo")
selected_model = st.selectbox("Select a model", list(model_options.keys()), label_visibility="collapsed")
model_path = model_options[selected_model]

#reset chat upon model change
if st.session_state.get("active_model") != selected_model:
    st.session_state.active_model = selected_model
    st.session_state.messages = init_messages()

with st.spinner(f"Loading {selected_model}..."):
    llm=load_model(model_path)

st.caption(f"{selected_model} loaded successfully!")
st.divider()

#initalize fresh chat history
if "messages" not in st.session_state:
    st.session_state.messages = init_messages()

for msg in st.session_state.messages:
    if msg["role"] == "system": #skip llm facing system messages
        continue
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

#Chat Section
if prompt := st.chat_input("Enter your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.empty()
        full_response = ""

        stream = llm.create_chat_completion(
            messages = st.session_state.messages,
            stream = True,
            temperature = 0.7,
        )

        for chunk in stream:
            delta = chunk["choices"][0]["delta"].get("content", "")
            full_response += delta
            response.markdown(full_response + "▌")
        
        response.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
