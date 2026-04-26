import streamlit as st
from llama_cpp import Llama

# Setup
st.set_page_config(
    page_title="LLM Fine Tuning Demo",
    layout="centered",
)

model_options = {
    "run7 (with LoRA)": {
        "base": "run7q4km.gguf",
        "lora": "run7lora.gguf"
    },
    "run7 (base only)": {
        "base": "run7q4km.gguf",
        "lora": None
    },
    "Vanilla LLama3.1-8B": {
        "base": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "lora": None
    },
    "Custom Tuned Model": {
        "base": "custom_tuned_model.gguf",
        "lora": None
    }
}

system_prompt = "You are a helpful assistant. Respond concisely."


# IMPORTANT: include model identity in cache key
@st.cache_resource
def load_model(model_key: str, base_path: str, lora_path: str | None):
    return Llama(
        model_path=base_path,
        lora_path=lora_path,
        lora_scale=1.0 if lora_path else None,
        n_gpu_layers=0,
        n_ctx=2048,  
        verbose=False,
    )


def init_messages():
    return [{"role": "system", "content": system_prompt}]


# UI
st.title("LLM Fine Tuning Demo")

selected_model = st.selectbox(
    "Select a model",
    list(model_options.keys()),
    label_visibility="collapsed"
)

model_config = model_options[selected_model]


# =========================
# FORCE CLEAN MODEL SWITCH
# =========================
if st.session_state.get("active_model") != selected_model:
    st.session_state.active_model = selected_model
    st.session_state.messages = init_messages()

    # CRITICAL FIX:
    # clear cached model so old base/LoRA cannot persist silently
    st.cache_resource.clear()


# Load model (now guaranteed fresh per selection)
with st.spinner(f"Loading {selected_model}..."):
    llm = load_model(
        selected_model,
        model_config["base"],
        model_config["lora"]
    )


st.caption(f"{selected_model} loaded successfully")
st.divider()


# init chat history safety net
if "messages" not in st.session_state:
    st.session_state.messages = init_messages()


# display chat
for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# chat input
if prompt := st.chat_input("Enter your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.empty()
        full_response = ""

        stream = llm.create_chat_completion(
            messages=st.session_state.messages,
            stream=True,
            temperature=0.7,
        )

        for chunk in stream:
            delta = chunk["choices"][0]["delta"].get("content", "")
            full_response += delta
            response.markdown(full_response + "▌")

        response.markdown(full_response)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )
