from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
# Notice the _classic addition here!
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

from src.rag.vector_store import get_retriever

def run_chat():
    print("🧠 Booting up local LLM (Ollama: Llama 3.2)...")
    
    llm = ChatOllama(model="llama3.2")
    retriever = get_retriever()

    system_prompt = (
        "You are an expert, patient Cambridge IGCSE Math Tutor. "
        "Use the provided syllabus context to answer the student's question accurately. "
        "If the answer is not contained within the context, do not guess; simply state that you do not know based on the syllabus.\n\n"
        "Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("\n✅ Math Tutor is ready! (Type 'exit' to quit)")
    print("-" * 50)
    
    while True:
        user_input = input("\nStudent: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Class dismissed!")
            break

        print("Tutor is thinking... (This might take a few seconds on a laptop CPU)")
        
        response = rag_chain.invoke({"input": user_input})
        
        print(f"\nTutor: {response['answer']}")

if __name__ == "__main__":
    run_chat()