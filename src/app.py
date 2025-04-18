import sys
from pathlib import Path

import streamlit as st
from openai import OpenAI

from hybrid_search import hybrid_search

# Add project root to Python path (before any imports)
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config

client = OpenAI(api_key=Config.api_key, base_url=Config.base_url)


def build_prompt(query):
    search_results = hybrid_search(query, size=1)
    prompt_template = Config.prompt_template

    context = ""

    for hit in search_results:
        doc = hit.metadata['_source']
        context = context + f"section: {doc['section']}\ntext: {hit.page_content}\n\n"

    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt


if __name__ == "__main__":
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
            stream = client.chat.completions.create(model=Config.model,
                                                    messages=[{"role": m["role"], "content": build_prompt(m["content"])}
                                                              for m in st.session_state.messages], stream=True)
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
