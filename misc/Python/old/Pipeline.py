import sys
import time
from NAR import AddInput, Reset
from narsese_truth_translator import process_nars_output
from english_to_narsese_modular import EnglishToNarsese
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama.llms import OllamaLLM
"""
* Currently having issues with Ollama in this script
* May not occur on MacOS or Linux, as all of the code can run together
* Running on Windows presents challenges to ferry between wsl and Windows
* Dockerizing may be an easy solution
"""
llm = OllamaLLM(model="llama3.2")
chat_history = []
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI assistant that answers questions based on a knowledge base of context given to you from an AGI system. You are not allowed to give answers that are from outside the context.\n "
            "You are not allowed to give answers that are not based on the context. You are not allowed to give answers that are not based on the knowledge base.\n "
            "You can interpret relationships like 'Bird is fly' as 'Birds fly.'\n "
            "You can interpret relationships like 'Tweety is bird' as 'Tweety is a bird.'"
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

def convert(line):
    """Convert a line of English to Narsese."""
    converter = EnglishToNarsese(
        verbose = False,
        output_truth = True,
        eternal_output = False
    )
    return converter.process_line(line)

def add_input(narsese, cycles=None, print_raw=False):
    """Add input to NARS and translate the output with enhanced truth value descriptions."""
    # Send the input to NARS
    raw_output = AddInput(narsese, Print=print_raw)
    if cycles is not None:
        AddInput(cycles)
    # Translate the output
    # translated = process_nars_output(raw_output, with_colors=True)
    # print(translated)
    return raw_output

def ollama_query(context):
    """Query the Ollama model."""
    question = input("You: ")
    # if question == "done":
    #     return
    
    response = chain.invoke({"input": question, "chat_history": chat_history, "context": context})
    chat_history.append(HumanMessage(content=question))
    
    # Check if response exists and has the expected structure
    if hasattr(response, 'content') and response.content:
        chat_history.append(AIMessage(content=response.content))
        print("NARS: " + response.content)
    else:
        # If response doesn't have content, print the raw response for debugging
        print("DEBUG - Raw response:", response)
        # Still add something to chat history to maintain continuity
        content = str(response) if response else "No response"
        chat_history.append(AIMessage(content=content))
        print("NARS: " + content)


def main():
    Reset()
    context = ""
    while True:
        print("1. Add input to NARS")
        print("2. Query Ollama")
        print("3. Exit")
        choice = input("Choose an option: ")
        if choice == "1":
            line = input("Enter a line of English: ")
            narsese = convert(line)
            print("Converted Narsese: ", narsese)
            context = add_input(narsese)
            print()
        if choice == "2":
            ollama_query(context)
            print()
        if choice == "3":
            break


if __name__ == "__main__":
    main()