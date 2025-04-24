"""
Interface to LLM (Ollama) for the NARS-Ollama pipeline
"""

import traceback
from typing import List, Dict, Any, Optional, Union

# Import LangChain Ollama components
try:
    from langchain_ollama.llms import OllamaLLM
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.messages import HumanMessage, AIMessage
except ImportError:
    print("Warning: LangChain libraries not found. LLM functionality will be limited.")

class LlmClient:
    """Client for interacting with Ollama LLM."""
    
    def __init__(self, model_name: str = "llama3.2", verbose: bool = False):
        """Initialize LLM client.
        
        Args:
            model_name: Name of the Ollama model to use
            verbose: Whether to print verbose output
        """
        self.verbose = verbose
        self.model_name = model_name
        
        try:
            if self.verbose:
                print(f"Initializing Ollama with model: {model_name}...")
            self.llm = OllamaLLM(model=model_name)
            self.chat_history = []
        except Exception as e:
            if self.verbose:
                print(f"Error initializing Ollama: {e}")
                traceback.print_exc()
            self.llm = None
    
    def extract_facts(self, user_input: str) -> List[str]:
        """Use Ollama to extract facts from user input.
        
        Args:
            user_input: User input text
            
        Returns:
            List of extracted facts
        """
        if self.verbose:
            print(f"Extracting facts from: '{user_input}'")
        
        if not self.llm:
            if self.verbose:
                print("LLM not initialized, cannot extract facts")
            return []
        
        try:
            from prompts import fact_extraction_template
            
            chain = fact_extraction_template | self.llm
            response = chain.invoke({"input": user_input})
            
            # Handle both object with .content and direct string responses
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = response
                
            if content and content.strip():
                # Remove the thinking section if present
                if '<think>' in content and '</think>' in content:
                    # Extract only the content after the </think> tag
                    content = content.split('</think>')[1].strip()
                
                # Extract facts as separate lines
                facts = [fact.strip() for fact in content.strip().split('\n') if fact.strip()]
                
                # Basic validation: ensure each fact has at least a subject and verb
                validated_facts = []
                for fact in facts:
                    words = fact.split()
                    if len(words) >= 2:  # At minimum "subject verb"
                        validated_facts.append(fact)
                    else:
                        if self.verbose:
                            print(f"Skipping invalid fact: '{fact}'")
                
                if self.verbose:
                    print(f"Extracted {len(validated_facts)} valid facts: {validated_facts}")
                return validated_facts
            else:
                if self.verbose:
                    print("No facts extracted (empty response)")
                return []
                
        except Exception as e:
            if self.verbose:
                print(f"Error extracting facts: {e}")
                traceback.print_exc()
            return []
    
    def generate_response(self, user_input: str, nars_knowledge: str) -> str:
        """Generate a response based on NARS knowledge.
        
        Args:
            user_input: User input text
            nars_knowledge: Knowledge extracted from NARS
            
        Returns:
            Generated response
        """
        if self.verbose:
            print("Generating response based on NARS knowledge...")
        
        if not self.llm:
            if self.verbose:
                print("LLM not initialized, cannot generate response")
            return "Error: LLM not initialized"
        
        try:
            from prompts import answer_template
            
            chain = answer_template | self.llm
            response = chain.invoke({
                "input": user_input,
                "chat_history": [],
                "context": nars_knowledge
            })
            
            # Handle both object with .content and direct string responses
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = response
            
            if self.verbose:
                print("\n=== RAW RESPONSE ===")
                print(content)


            if '<think>' in content and '</think>' in content:
                final_answer = content.split('</think>')[1].strip()
            else:
                final_answer = content
            
            # Update chat history
            self.chat_history.append(HumanMessage(content=user_input))
            self.chat_history.append(AIMessage(content=final_answer))
            
            return final_answer
            
        except Exception as e:
            error_msg = f"Error generating response: {e}"
            if self.verbose:
                print(error_msg)
                traceback.print_exc()
            return error_msg