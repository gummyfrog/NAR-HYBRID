#!/usr/bin/env python3
"""
NARS-Ollama Integration System

This implements a natural language interface to the NARS (Non-Axiomatic Reasoning System)
by using Ollama to translate between natural language and Narsese.

Usage:
  python main.py [--model MODEL] [--verbose] [--no-init] [--load FILE] [--save FILE]

Options:
  --model MODEL    Specify the Ollama model to use (default: llama3.2)
  --verbose        Enable verbose output
  --no-init        Don't initialize NARS with default knowledge
  --load FILE      Load NARS knowledge from a file at startup
  --save FILE      Save NARS knowledge to a file on exit
"""

import sys
import os
import argparse
import atexit
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
    
    parser.add_argument(
        "--load", 
        type=str,
        help="Load NARS knowledge from a file at startup"
    )
    
    parser.add_argument(
        "--save", 
        type=str,
        help="Save NARS knowledge to a file on exit"
    )
    
    parser.add_argument(
        "--no-auto-save",
        action="store_true",
        help="Disable automatic saving on exit"
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
    
    # Load knowledge if specified
    if args.load:
        if os.path.exists(args.load):
            print(f"Loading knowledge from {args.load}...")
            result = pipeline.nars_client.load_knowledge(args.load)
            print(result.get("raw", "Loading complete"))
        else:
            print(f"Knowledge file not found: {args.load}")
    
    # Set up auto-save on exit if specified
    if args.save and not args.no_auto_save:
        def save_on_exit():
            print(f"\nSaving knowledge to {args.save}...")
            result = pipeline.nars_client.save_knowledge(args.save)
            print(result.get("raw", "Saving complete"))
        
        atexit.register(save_on_exit)
    
    print("\n=== NARS-OLLAMA PIPELINE READY ===")
    print("You can start asking questions or providing statements.")
    print("Type 'exit' to quit.")
    print("Special commands:")
    print("  *reset - Reset NARS knowledge")
    print("  *save [FILE] - Save NARS knowledge to a file")
    print("  *load [FILE] - Load NARS knowledge from a file")
    print("  *run N - Run N inference cycles")
    print("  *concepts - Show all concepts in NARS")
    
    # Main interaction loop
    while True:
        try:
            user_input = input("\nYou: ")
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Exiting...")
                break
                
            # Process the input
            if user_input.startswith("*"):
                # Handle special commands
                if user_input.startswith("*save") or user_input.startswith("*load"):
                    result = pipeline.nars_client.add_input(user_input)
                    print(result.get("raw", "Command processed"))
                else:
                    # Other NARS commands
                    result = pipeline.nars_client.add_input(user_input)
                    
                    # For readability, use the truth translator for concept output
                    if user_input.startswith("*concepts"):
                        from truth_translator import process_nars_output
                        print(process_nars_output(result))
                    else:
                        print(result.get("raw", "Command processed"))
            else:
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