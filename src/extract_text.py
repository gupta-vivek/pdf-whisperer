import fitz


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from pdf file
    :param pdf_path: Path to pdf file
    :return: extracted text
    """
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def preprocess_text(input_text: list[str], out_path: str):
    """
    Preprocess the text and save it in a text file
    :param input_text: input text
    :param out_path: output path
    :return:
    """
    with open(out_path, "w", encoding="utf-8") as f:
        for line in input_text:
            line = line.strip()
            line = " ".join(line.split())
            if line or len(line) > 1:
                f.write(line + "\n")


if __name__ == "__main__":
    # document = extract_text_from_pdf("../data/instructions.pdf")
    # with open('../document.txt', 'w', encoding="utf-8") as f:
    #     f.write(document)

    with open("../data/document.txt", "r", encoding="utf-8") as f:
        text = f.readlines()
        preprocess_text(text, "../data/cleaned_doc.txt")
