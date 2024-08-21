from src.database import load_documents, custom_split_text, split_documents, add_to_chroma
from src.conversation_handling import handle_conversation


if __name__ == "__main__":
    documents = load_documents()
    splitted_documents = []
    for doc in documents:
        splitted_documents.append(custom_split_text(doc))

    chunks = split_documents(splitted_documents)

    add_to_chroma(chunks)
    handle_conversation()
    print('Program ended!')

