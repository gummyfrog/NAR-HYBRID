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
    
    def __init__(self, model_name: str = "llama3.2", verbose: bool = False, fact_model: str = None):
        """Initialize LLM client.
        
        Args:
            model_name: Name of the Ollama model to use for response generation
            verbose: Whether to print verbose output
            fact_model: Optional separate model for fact extraction (defaults to model_name)
        """
        self.verbose = verbose
        self.model_name = model_name
        self.fact_model = fact_model or model_name
        
        try:
            if self.verbose:
                print(f"Initializing main model: {model_name}")
                print(f"Initializing fact extraction model: {self.fact_model}")
            
            self.llm = OllamaLLM(model=model_name, base_url="http://ollamaNARS:11434")
            self.fact_llm = OllamaLLM(model=self.fact_model)
            self.chat_history = []
        except Exception as e:
            if self.verbose:
                print(f"Error initializing Ollama: {e}")
                traceback.print_exc()
            self.llm = None
            self.fact_llm = None
    
    def extract_facts(self, user_input: str) -> List[str]:
        """Use LLM to extract simple statements from user input.

        Args:
            user_input: User input text
            
        Returns:
            List of simple statements
        """
        if self.verbose:
            print(f"Extracting facts from: '{user_input}'")
            print(f"Using model: {self.fact_model}")

        if not self.fact_llm:
            if self.verbose:
                print("Fact extraction LLM not initialized")
            return []

        try:
            from prompts import fact_extraction_template
            
            chain = fact_extraction_template | self.fact_llm
            response = chain.invoke({"input": user_input})
            
            # Handle both object with .content and direct string responses
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = response
                
            if content and content.strip():
                # Filter out thinking sections
                if "<think>" in content and "</think>" in content:
                    thinking_part = content.split("<think>")[1].split("</think>")[0]
                    content = content.replace(f"<think>{thinking_part}</think>", "")
                
                # Split the response into lines and filter out empty ones
                raw_facts = [line.strip() for line in content.strip().split('\n') if line.strip()]
                
                # Filter out any lines that are clearly not facts
                facts = []
                for line in raw_facts:
                    # Skip lines that are part of the thinking process or metadata
                    if line.startswith('-') or line.startswith('*') or line.startswith('#') or \
                        line.startswith('<') or line.startswith('>') or line.startswith('1.') or \
                        "think" in line.lower():
                        continue
                    facts.append(line)
                
                if self.verbose:
                    print(f"Extracted {len(facts)} facts:")
                    for i, fact in enumerate(facts):
                        print(f"  {i+1}. {fact}")
                        
                return facts
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
            print(f"Using model: {self.model_name}")
        
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