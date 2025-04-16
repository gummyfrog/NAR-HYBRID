#!/usr/bin/env python3

import sys
import time
import traceback
from NAR import AddInput, Reset
from narsese_truth_translator import process_nars_output
from english_to_narsese_modular import EnglishToNarsese
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama.llms import OllamaLLM

llm = OllamaLLM(model="llama3.2")
chat_history = []

fact_extraction_template = ChatPromptTemplate.from_messages([
	(
		"system",
		"You are a fact extraction system ONLY. Your sole purpose is to identify explicit factual statements in user input and convert them to simple atomic facts. "
		"\n\nRULES:\n"
		"1. NEVER answer questions or generate new information\n"
		"2. ONLY extract facts that are explicitly stated in the input\n"
		"3. If the input is a question, or contains no explicit facts, output an empty string\n"
		"4. Each atomic fact should follow the structure: [Subject] [Relation] [Object]\n"
		"5. Format all facts as a single line separated by the pipe character: |\n"
		"\nFormatting Requirements:\n"
		"- Use simple present tense\n"
		"- Remove articles (a, an, the)\n"
		"- Keep each fact atomic (one relationship only)\n"
		"- Do not add any explanations, headers, or additional text\n"
		"\nExample inputs and expected outputs:\n"
		"Input: \"Birds can fly and they have feathers.\"\n"
		"Output: Bird can fly | Bird has feather\n"
		"\nInput: \"Tweety is a yellow bird that sings.\"\n"
		"Output: Tweety is bird | Tweety is yellow | Tweety can sing\n"
		"\nInput: \"What is a bird?\"\n"
		"Output: \n"
		"\nInput: \"Can you tell me if birds can fly?\"\n"
		"Output: \n"
		"\nInput: \"My friend John is tall and has two dogs named Max and Bella.\"\n"
		"Output: John is tall | John has dog | Dog is Max | Dog is Bella\n"
		"\nInput: \"Do you know anything about birds?\"\n"
		"Output: \n"
		"\nRemember: If the input contains a question mark or is phrased as a question, output nothing."
	),
	("human", "{input}")
])

fact_extraction_chain = fact_extraction_template | llm

answer_template = ChatPromptTemplate.from_messages([
	(
		"system",
		"You are a specialized assistant that reasons with knowledge from the Non-Axiomatic Reasoning System (NARS).\n\n"
		
		"## FORMATTING REQUIREMENT (MOST IMPORTANT)\n"
		"For EVERY response, you MUST structure your answer in exactly two parts:\n\n"
		
		"```\n"
		"<REASONING>\n"
		"... your step-by-step analysis here ...\n"
		"</REASONING>\n"
		"\n"
		"... your final answer here ...\n"
		"```\n\n"
		
		"The <REASONING> section will be hidden from the user. Your final answer after the </REASONING> tag will be shown to the user.\n\n"
		
		"## Inside <REASONING>, briefly analyze:\n"
		"1. Key entities in the question\n"
		"2. Relevant NARS statements about these entities\n"
		"3. Truth values and confidence levels of these statements\n"
		"4. Your conclusion based only on these statements\n\n"
		
		"## Your final answer (after </REASONING>) should:\n"
		"1. Directly answer the question based only on NARS knowledge\n"
		"2. Cite specific statements used, including their truth values and confidence\n"
		"3. Say 'I don't know based on the available knowledge' if no relevant information exists\n\n"
		
		"## CRITICAL RULES:\n"
		"- Only use information from the NARS knowledge base\n"
		"- Accept NARS statements as true even if they contradict common knowledge\n"
		"- When statements conflict, prefer those with higher confidence\n"
		"- NEVER introduce outside information\n\n"
		
		"## Understanding NARS Statements:\n"
		"- Truth values range from DEFINITELY TRUE to DEFINITELY FALSE\n"
		"- Confidence levels range from EXTREMELY CONFIDENT to EXTREMELY UNCERTAIN\n"
		"- 'X is Y' means 'X is a Y'\n"
		"- 'X is it leads to Y is it' means 'If X then Y'\n\n"
		
		"Remember: All reasoning must be based EXCLUSIVELY on the provided knowledge.\n"
		"---------------------------\n"
		"Knowledge base:\n{context}\n"
	),
	MessagesPlaceholder(variable_name="chat_history"),
	("human", "{input}")
])

answer_chain = answer_template | llm

def convert_to_narsese(line):
	print(f"Converting to Narsese: '{line}'")
	try:
		converter = EnglishToNarsese(
			verbose=False,
			output_truth=True,
			eternal_output=False
		)
		narsese = converter.process_line(line)
		print(f"Converted to: '{narsese}'")
		return narsese
	except Exception as e:
		print(f"Error converting to Narsese: {e}")
		traceback.print_exc()
		return None

def add_to_nars(narsese, print_raw=False):
	if not narsese:
		print("Skipping empty Narsese")
		return ""
	
	try:
		print(f"Adding to NARS: '{narsese}'")
		raw_output = AddInput(narsese, Print=print_raw)
		AddInput("300")
		# Translate the output for better readability
		translated = process_nars_output(raw_output, with_colors=True)
		print(f"NARS response: {translated}")
		
		return raw_output
	except Exception as e:
		print(f"Error adding to NARS: {e}")
		traceback.print_exc()
		return ""

def extract_nars_knowledge():
	"""Extract all knowledge from NARS as context."""
	print("Extracting knowledge from NARS...")
	try:
		concepts_output = AddInput("*concepts", Print=False)
		process_nars_output(concepts_output)

		knowledge = process_nars_output(concepts_output, with_colors=False)
		
		print(f"Extracted {len(knowledge)} characters of knowledge")
		print("Knowledge extracted from NARS:")
		print(knowledge)
		return knowledge
	except Exception as e:
		print(f"Error extracting knowledge: {e}")
		traceback.print_exc()
		return "No knowledge available"

def extract_facts_from_input(user_input):
	"""Use Ollama to extract facts from user input."""
	print(f"Extracting facts from: '{user_input}'")
	try:
		response = fact_extraction_chain.invoke({"input": user_input})
		
		if hasattr(response, 'content'):
			content = response.content
		else:
			content = response
			
		if content:
			facts = content.strip().split('|')
			print(f"Extracted facts: {facts}")
			return facts
		else:
			print("Failed to extract facts (empty response)")
			return []
	except Exception as e:
		print(f"Error extracting facts: {e}")
		traceback.print_exc()
		return []

def process_user_input(user_input):
	print("\n=== PROCESSING USER INPUT ===")
	print(f"User input: {user_input}")
	
	try:
		facts = extract_facts_from_input(user_input)
		
		print("\n=== ADDING FACTS TO NARS ===")
		for fact in facts[0].split('\n'):
			narsese = convert_to_narsese(fact.strip())
			if narsese:
				add_to_nars(narsese)
		
		nars_knowledge = extract_nars_knowledge()
		
		print("\n=== GENERATING RESPONSE BASED ON NARS KNOWLEDGE ===")
		try:
			response = answer_chain.invoke({
				"input": user_input, 
				"chat_history": [], 
				"context": nars_knowledge
			})
			
			if hasattr(response, 'content'):
				content = response.content
			else:
				content = response
			
			# DEBUG
			chat_history.append(HumanMessage(content=user_input))
			
			if content:
				print("\n=== FINAL RESPONSE ===")
				print(content)
				return content
			else:
				content = "Failed to generate a response (empty content)"
				print("\n=== FINAL RESPONSE ===")
				print(content)
				return content
		except Exception as e:
			print(f"Error generating response: {e}")
			traceback.print_exc()
			content = f"Error: {str(e)}"
			return content
	except Exception as e:
		print(f"Error in processing pipeline: {e}")
		traceback.print_exc()
		return f"Error: {str(e)}"

def main():
	"""Main function to run the pipeline."""
	print("=== INITIALIZING NARS ===")
	Reset()
	
	add_to_nars("*volume=100")  # Full output volume
	
	print("\n=== ADDING INITIAL KNOWLEDGE TO NARS ===")
	add_to_nars("<bird --> animal>. {0.9 0.9}")
	add_to_nars("<penguin --> bird>. {0.9 0.8}")
	add_to_nars("<swan --> bird>. {1.0 0.9}")
	add_to_nars("<tweety --> penguin>. {0.0 0.9}")

	print("\n=== NARS-OLLAMA PIPELINE READY ===")
	print("You can start asking questions or providing statements.")
	print("Type 'exit' to quit.")
	
	while True:
		user_input = input("\nYou: ")
		if user_input.lower() in ['exit', 'quit', 'q']:
			break
		
		process_user_input(user_input)

if __name__ == "__main__":
	main()