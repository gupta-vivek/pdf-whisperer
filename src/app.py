import os
import sys

import torch

# Add project root to sys.path manually
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

torch.classes.__path__ = []  # Streamlit pytorch introspection fix

import streamlit as st
from openai import OpenAI
from config.config import Config
from hybrid_lancedb import hybrid_search_lance, init_lance_db, get_reranker


@st.cache_resource
def get_lance():
    config_dir = Config.config_dir
    URI = os.path.join(config_dir, Config.lance_db_uri)
    reranker = get_reranker()
    return init_lance_db(URI, Config.lance_db_table), reranker


table, reranker = get_lance()


def build_prompt(query):
    """
    Build a prompt from the given query.
    :param query:
    :return:
    """
    search_results = hybrid_search_lance(table, query, reranker, limit=1)
    prompt_template = Config.prompt_template
    context = ""
    for hit in search_results:
        context = context + f"{hit}\n\n"

    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt


if __name__ == "__main__":
    client = OpenAI(api_key=Config.api_key, base_url=Config.base_url)

    st.title("PDF Whisperer")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        messages = []
        for m in st.session_state.messages:
            if m["role"] == "user" and m == st.session_state.messages[-1]:
                content = build_prompt(m["content"])
            else:
                content = m["content"]
            messages.append({"role": m["role"], "content": content})

        with st.chat_message("assistant"):
            stream = client.chat.completions.create(model=Config.model, messages=messages, stream=True)
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
