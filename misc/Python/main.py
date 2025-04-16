#!/usr/bin/env python3
"""
NARS-Ollama Integration System

This implements a natural language interface to the NARS (Non-Axiomatic Reasoning System)
by using Ollama to translate between natural language and Narsese.

Usage:
  python main.py [--model MODEL] [--verbose] [--no-init]

Options:
  --model MODEL    Specify the Ollama model to use (default: llama3.2)
  --verbose        Enable verbose output
  --no-init        Don't initialize NARS with default knowledge
"""

import sys
import argparse
from pipeline import NarsOllamaPipeline

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="NARS-Ollama Integration System")
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="llama3.2", 
        help="Specify the Ollama model to use"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--no-init", 
        action="store_true", 
        help="Don't initialize NARS with default knowledge"
    )
    
    return parser.parse_args()

def main():
    """Main function to run the NARS-Ollama system."""
    args = parse_args()
    
    print(f"Initializing NARS-Ollama pipeline with model: {args.model}")
    print("This may take a moment...\n")
    
    # Initialize the pipeline
    pipeline = NarsOllamaPipeline(
        model_name=args.model,
        verbose=args.verbose,
        init_nars=not args.no_init
    )
    
    print("\n=== NARS-OLLAMA PIPELINE READY ===")
    print("You can start asking questions or providing statements.")
    print("Type 'exit' to quit.")
    
    # Main interaction loop
    while True:
        try:
            user_input = input("\nYou: ")
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Exiting...")
                break
                
            # Special commands
            if user_input.startswith("*"):
                if user_input == "*reset":
                    pipeline.initialize_nars()
                    print("NARS has been reset.")
                    continue
                else:
                    # Pass NARS commands directly
                    result = pipeline.nars_client.add_input(user_input)
                    from truth_translator import process_nars_output
                    print(process_nars_output(result))
                    continue
            
            # Process regular input
            response = pipeline.process_input(user_input)
            print(f"\nNARS: {response}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
            
        except Exception as e:
            print(f"Error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    main()