from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents.base import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import re
from src.embeddings import get_embedding_function
from src.config import CHROMA_PATH


def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def load_documents() -> list[Document]:
    document_leader = PyPDFDirectoryLoader('pdfs_data')
    data = document_leader.load()
    return data


def custom_split_text(doc: Document):
    separators = ['\n']  # ['zł', '\n']

    pattern = f'({"|".join(map(re.escape, separators))})'

    chunks = re.split(pattern, doc.page_content)

    chunks = [s for s in chunks if s.strip()]  # Removes ' ', '\n'
    chunks = [s.lstrip() for s in chunks]  # Removes spaces from the beginning of strings

    chunks = [s.rstrip() for s in chunks]  # Removes spaces from the end of strings

    chunks = [re.sub(r'\s+', ' ', s) for s in chunks]

    # splitting strings by digit and dot
    pattern = r'\d+\.\s.*?(?=\d+\.\s|$)'
    splitted_chunks: list[str] = []

    for s in chunks:
        if re.search(pattern, s):
            for splitted_s in re.findall(pattern, s):
                splitted_chunks.append(splitted_s)
        #else:
        #    splitted_chunks.append(s)

    doc.page_content = "\n".join(splitted_chunks)
    return doc


def split_documents(documents: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=0,
        separators=["\n"]
    )
    data = text_splitter.split_documents(documents)
    return data
