"""
* This should work as a test for Ollama as a whole.
* Install Ollama separately here: https://ollama.com/
* Follow instructions to install the correct model here: https://github.com/ollama/ollama
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama.llms import OllamaLLM

llm = OllamaLLM(model="llama3.2")

chat_history = []

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI assistant that answers questions based on a knowledge base of context given to you from an AGI system. You are not allowed to give answers that are from outside the context.\n "
            "You are not allowed to give answers that are not based on the context. You are not allowed to give answers that are not based on the knowledge base.\n "
            "If a question is not based on the knowledge base, you should say 'I don't know'.\n "
            "Even if a question has a factual answer, if it is not in the knowledge base, you should say 'I don't know'.\n "
            "If you have to say 'I don't know', you should say nothing else.\n "
            "If a user input is a statement, and not a question, you should say 'Please add another fact, or ask a quesiton'. \n "
            "---------------------------\n"
            "Knowledge base: {context}\n",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

chain = prompt_template | llm

def start_app():
    while True:
        question = input("You: ")
        if question == "done":
            return

        knowledge_base = "The sky is blue."
        response = chain.invoke({"input": question, "chat_history": chat_history, "context": knowledge_base})
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=response))

        print("AI: " + response)

if __name__ == "__main__":
    start_app()