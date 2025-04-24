"""
Core pipeline for NARS-Ollama system
"""

import traceback
from typing import List, Optional

from nars_client import NarsClient
from llm_client import LlmClient
from english_to_narsese_modular import EnglishToNarsese

class NarsOllamaPipeline:
    """Pipeline for NARS-Ollama system."""
    
    def __init__(
        self, 
        model_name: str = "llama3.2", 
        verbose: bool = False,
        init_nars: bool = True
    ):
        """Initialize the pipeline.
        
        Args:
            model_name: Name of the Ollama model to use
            verbose: Whether to print verbose output
            init_nars: Whether to initialize NARS with default knowledge
        """
        self.verbose = verbose
        
        # Initialize components
        self.nars_client = NarsClient(verbose=verbose)
        self.llm_client = LlmClient(model_name=model_name, verbose=verbose)
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
        
        # Add initial knowledge
        initial_knowledge = [
            "<bird --> animal>. {0.9 0.9}",
            "<penguin --> bird>. {0.9 0.8}",
            "<swan --> bird>. {1.0 0.9}",
            "<tweety --> penguin>. {0.0 0.9}"
        ]
        
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
                    self.nars_client.run_cycles(300)
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
                    self.nars_client.run_cycles(300)
            
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