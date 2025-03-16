# PDF Whisperer
An AI powered chatbot that answers customer queries by retrieving relevant information from a company's database.

## Roadmap for the first cut (will be continually updating as the project progresses)

- [x] Extract the text
  - Text is extracted using PyMuPDF
- [x] Chunk the data
  - Document is chunked into sections based on the content page. If the section is large, it is further divided by
    semantic chunking.

- [ ] Embed and store (in progress)

- [ ] Query the LLM

- [ ] Build the chatbot