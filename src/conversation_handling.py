from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from src.embeddings import get_embedding_function
from src.config import CHROMA_PATH


template = '''
Odpowiedz na pytanie poniżej

Oto historia konwersacji: {context}

Pytanie: {question}

Odpowiedź:
'''

model = OllamaLLM(model='llama3')
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


def handle_conversation():
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    context = ''
    print("Witamy w chatbocie. Naciśnij „q”, aby wyjść.")
    while True:
        user_input = input('You: ')
        if user_input.lower() == 'q':
            print('Bot: Dzięki za rozmowę!')
            break
        db_results = db.similarity_search_with_score(user_input, k=5)
        additional_context = "\n\n---\n\n".join([doc.page_content for doc, _score in db_results])

        full_context = f"{context}\n{additional_context}"

        result = chain.invoke({'context': full_context, 'question': user_input})
        print(f'Bot: {result}')
        context += f'\nUżytkownik: {user_input}\nAI: {result}'

