"""
Core pipeline for NARS-Ollama system
"""

import traceback
import re
import os
from typing import List, Optional

from nars_client import NarsClient
from llm_client import LlmClient
from english_to_narsese_modular import EnglishToNarsese

class NarsOllamaPipeline:
    """Pipeline for NARS-Ollama system."""
    
    def __init__(
        self, 
        model_name: str = "llama3.2", 
        fact_model: str = None,
        verbose: bool = False,
        init_nars: bool = True
    ):
        """Initialize the pipeline.
        
        Args:
            model_name: Name of the Ollama model to use for response generation
            fact_model: Name of the Ollama model to use for fact extraction (optional)
            verbose: Whether to print verbose output
            init_nars: Whether to initialize NARS with default knowledge
        """
        self.verbose = verbose
        
        # Initialize components
        self.nars_client = NarsClient(verbose=verbose)
        self.llm_client = LlmClient(model_name=model_name, fact_model=fact_model, verbose=verbose)
        self.converter = EnglishToNarsese(
            verbose=False,
            output_truth=True,
            eternal_output=False
        )
        
        # Reset and initialize NARS if requested
        if init_nars:
            self.initialize_nars()
    
    def initialize_nars(self) -> None:
        """Reset NARS and add initial knowledge."""
        # Reset NARS
        self.nars_client.reset()
        
        # Configure NARS
        self.nars_client.add_input("*volume=100")
        
        # Add initial knowledge... 
        initial_knowledge = []
        
        for statement in initial_knowledge:
            self.nars_client.add_input(statement)
            
        if self.verbose:
            print("NARS initialized with default knowledge")
    
    def convert_to_narsese(self, text: str) -> Optional[str]:
        """Convert English text to Narsese.
        
        Args:
            text: English text to convert
            
        Returns:
            Narsese representation or None if conversion failed
        """
        if not text or text.strip() == "":
            return None
            
        if self.verbose:
            print(f"Converting to Narsese: '{text}'")
            

        if text.startswith("(") or text.startswith("["):
            if self.verbose:
                print("Skipping statement starting with '(' or '['")
            return {"raw": ""}
            
        try:
            narsese = self.converter.process_line(text)
            
            if self.verbose:
                print(f"Converted to: '{narsese}'")
                
            return narsese
            
        except Exception as e:
            if self.verbose:
                print(f"Error converting to Narsese: {e}")
                traceback.print_exc()
            return None
        

    def process_input(self, user_input: str) -> str:
        """Process user input through the complete pipeline.

        Args:
            user_input: User input text
            
        Returns:
            Generated response
        """
        if self.verbose:
            print("\n=== PROCESSING USER INPUT ===")
            print(f"User input: {user_input}")

        try:
            # Check if input is a question (ends with ?)
            is_question = user_input.strip().endswith('?')
            
            if not is_question:
                # Stage 1: Extract simple statements using LLM (only if not a question)
                simple_statements = self.llm_client.extract_facts(user_input)
                
                if self.verbose:
                    print("\n=== EXTRACTED SIMPLE STATEMENTS ===")
                    for stmt in simple_statements:
                        print(f"- {stmt}")
                
                # Stage 2: Convert simple statements to Narsese and add to NARS
                if self.verbose:
                    print("\n=== ADDING FACTS TO NARS ===")
                
                # Process and add each simple statement
                for statement in simple_statements:
                    if not statement.strip():  # Skip empty statements
                        continue
                        
                    # Convert the simple statement to Narsese
                    narsese = self.convert_to_narsese(statement.strip())
                    
                    if narsese:
                        if self.verbose:
                            print(f"Simple: '{statement}' → Narsese: '{narsese}'")
                        
                        # Add the Narsese statement to NARS
                        self.nars_client.add_input(narsese)
                        
                        # Run inference cycles after each fact
                        # self.nars_client.run_cycles(300)
                    else:
                        if self.verbose:
                            print(f"Failed to convert: '{statement}'")
            else:
                if self.verbose:
                    print("\n=== SKIPPING FACT EXTRACTION (QUESTION DETECTED) ===")
            
            # Process the original input if it's a question
            if is_question:
                # Try to convert the question directly
                question_narsese = self.convert_to_narsese(user_input.strip())
                if question_narsese:
                    if self.verbose:
                        print(f"Question → Narsese: '{question_narsese}'")
                    self.nars_client.add_input(question_narsese)
                    # self.nars_client.run_cycles(300)
            
            # Stage 3: Extract NARS knowledge
            nars_knowledge = self.nars_client.extract_knowledge()
            
            # Stage 4: Generate response based on NARS knowledge
            if self.verbose:
                print("\n=== GENERATING RESPONSE ===")
                
            response = self.llm_client.generate_response(user_input, nars_knowledge)
            
            if self.verbose:
                print("\n=== FINAL RESPONSE ===")
                
            return response
            
        except Exception as e:
            error_msg = f"Error in processing pipeline: {e}"
            if self.verbose:
                print(error_msg)
                traceback.print_exc()
            return error_msg
    
    def process_input_without_response(self, user_input: str) -> None:
        """Process user input through the pipeline without generating a response.
        
        This method extracts facts, converts them to Narsese, and adds them to NARS,
        but does not generate a natural language response.

        Args:
            user_input: User input text
        """
        if self.verbose:
            print("\n=== PROCESSING USER INPUT (NO RESPONSE GENERATION) ===")
            print(f"User input: {user_input}")

        try:
            # Stage 1: Extract simple statements using LLM
            simple_statements = self.llm_client.extract_facts(user_input)
            
            if self.verbose:
                print("\n=== EXTRACTED SIMPLE STATEMENTS ===")
                for stmt in simple_statements:
                    print(f"- {stmt}")
            
            # Stage 2: Convert simple statements to Narsese and add to NARS
            if self.verbose:
                print("\n=== ADDING FACTS TO NARS ===")
            
            # Process and add each simple statement
            for statement in simple_statements:
                if not statement.strip():  # Skip empty statements
                    continue
                    
                # Convert the simple statement to Narsese
                narsese = self.convert_to_narsese(statement.strip())
                
                if narsese:
                    if self.verbose:
                        print(f"Simple: '{statement}' → Narsese: '{narsese}'")
                    
                    # Add the Narsese statement to NARS
                    self.nars_client.add_input(narsese)
                    
                    # Run inference cycles after each fact
                    self.nars_client.run_cycles(3)
                else:
                    if self.verbose:
                        print(f"Failed to convert: '{statement}'")
            
            # Process the original input if it's a question
            if "?" in user_input:
                # Try to convert the question directly
                question_narsese = self.convert_to_narsese(user_input.strip())
                if question_narsese:
                    if self.verbose:
                        print(f"Question → Narsese: '{question_narsese}'")
                    self.nars_client.add_input(question_narsese)
                    # self.nars_client.run_cycles(300)
            
            if self.verbose:
                print("\n=== PROCESSING COMPLETE (NO RESPONSE GENERATED) ===")
                
        except Exception as e:
            error_msg = f"Error in processing pipeline: {e}"
            if self.verbose:
                print(error_msg)
                traceback.print_exc()
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using basic rules.
        
        Args:
            text: Text to split into sentences
            
        Returns:
            List of sentences
        """
        # This is a simple sentence splitter - you might want a more sophisticated one
        # for production use (like using NLTK or spaCy)
        
        # Remove multiple whitespaces and newlines
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split on common sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Clean up and filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def process_file(self, file_path: str) -> None:
        """Process a text file sentence-by-sentence through the pipeline without generating responses.
        
        Args:
            file_path: Path to the text file to process
        """
        try:
            if not os.path.exists(file_path):
                print(f"Error: File not found: {file_path}")
                return
            
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Split content into sentences
            sentences = self.split_into_sentences(content)
            
            total_sentences = len(sentences)
            
            print(f"Processing file: {file_path}")
            print(f"Found {total_sentences} sentences to process")
            
            # Process each sentence individually
            for i, sentence in enumerate(sentences, 1):
                if self.verbose:
                    print(f"\n=== Processing sentence {i}/{total_sentences} ===")
                    print(f"Sentence: {sentence}")
                else:
                    # Show progress without being verbose
                    print(f"Processing sentence {i}/{total_sentences}...", end='\r')
                
                # Process this sentence without generating a response
                self.process_input_without_response(sentence)
            
            print("\n" + "=" * 50)
            print(f"Successfully processed {total_sentences} sentences from: {file_path}")
            print("=" * 50)
            
        except Exception as e:
            error_msg = f"Error processing file {file_path}: {e}"
            print(error_msg)
            if self.verbose:
                traceback.print_exc()