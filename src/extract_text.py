"""Extract text from PDF files."""

import json
import re

import fitz


def extract_sections(sections, document_path):
    """
    Extracts sections of text from the PDF document based on section titles and their
    corresponding page numbers.

    :param sections: A list of sections with their title and page number.
    :param document_path: Path to the PDF document.
    :return: A list of dictionaries, where each dictionary contains the section title and the
        extracted and cleaned text.
    :rtype: list of dict
    """
    doc = fitz.open(document_path)
    len_sections = range(len(sections) - 1)
    documents = []
    for i in len_sections:
        start_title, start_page_num = sections[i].split(',')
        end_title, end_page_num = sections[i + 1].split(',')
        text = ""
        if start_page_num == end_page_num:
            end_page_num = int(end_page_num) + 1
        for page in doc[int(start_page_num):int(end_page_num)]:
            page_text = page.get_text()
            text += page_text
        text = re.sub(r'-\n', '', text)  # Fix hyphenated words split across lines
        documents.append({"section": start_title, "text": text})

    return documents


if __name__ == "__main__":
    document_path = "../data/Kompendium1.pdf"
    with open('../data/sections.txt', 'r', encoding='utf-8') as f:
        sections = f.read().lower().splitlines()

    documents = extract_sections(sections, document_path)
    with open('../data/document.json', 'w', encoding='utf-8') as f:
        json.dump(documents, f)
