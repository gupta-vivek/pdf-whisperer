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
from hybrid_elastic import hybrid_search, init_elastic_search


def build_prompt(query, retriever):
    """
    Build a prompt from the given query and retriever.
    :param query:
    :param retriever:
    :return:
    """
    search_results = hybrid_search(retriever, query, size=1)
    prompt_template = Config.prompt_template

    context = ""

    for hit in search_results:
        doc = hit.metadata['_source']
        context = context + f"section: {doc['section']}\ntext: {hit.page_content}\n\n"

    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt


if __name__ == "__main__":
    client = OpenAI(api_key=Config.api_key, base_url=Config.base_url)
    retriever = init_elastic_search()

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

        with st.chat_message("assistant"):
            stream = client.chat.completions.create(model=Config.model, messages=[
                {"role": m["role"], "content": build_prompt(m["content"], retriever)} for m in
                st.session_state.messages], stream=True)
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
