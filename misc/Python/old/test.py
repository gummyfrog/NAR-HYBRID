#!/usr/bin/env python3
"""
Enhanced OpenNARS Demo with Truth Value Translation
This demonstrates:
1. Adding statements to NARS
2. Asking questions with natural language translations of statements AND truth values
3. Viewing knowledge with enhanced readability
"""

import sys
import time
from NAR import AddInput, Reset
from narsese_truth_translator import process_nars_output

# Check if colors should be disabled
with_colors = "noColors" not in sys.argv

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*50)
    print(f"  {text}")
    print("="*50)

def add_input_with_translation(narsese, cycles=None, print_raw=False):
    """Add input to NARS and translate the output with enhanced truth value descriptions"""
    # Send the input to NARS
    raw_output = AddInput(narsese, Print=print_raw)
    if cycles != None:
        AddInput(cycles)
    

    
    # Translate the output
    translated = process_nars_output(raw_output, with_colors)
    print(translated)
    
    return raw_output

def main():
    # Reset NARS to start fresh
    print_header("INITIALIZING NARS")
    Reset()
    
    # Configure NARS
    add_input_with_translation("*volume=100")  # Full output volume
    
    # 1. ADDING STATEMENTS TO NARS
    print_header("ADDING STATEMENTS TO NARS")
    
    # Add some taxonomic knowledge with varying truth values
    print("Adding statements about animals...")
    add_input_with_translation("<bird --> animal>. {0.9 0.9}")  # Birds are animals (high confidence)
    add_input_with_translation("<penguin --> bird>. {0.9 0.8}")  # Penguins are birds (good confidence)
    add_input_with_translation("<swan --> bird>. {1.0 0.9}")     # Swans are birds (very high confidence)
    add_input_with_translation("<swan --> [white]>. {0.8 0.7}")  # Swans are white (moderate confidence)
    add_input_with_translation("<penguin --> [swim]>. {0.9 0.8}")  # Penguins can swim (good confidence)
    add_input_with_translation("<{Tweety} --> bird>. {1.0 0.95}")   # Tweety is a bird (very high confidence)
    add_input_with_translation("<{Tweety} --> [yellow]>. {0.9 0.8}")  # Tweety is yellow (good confidence)
    add_input_with_translation("<bird --> machine>. {0.1 0.9}")  # Birds are animals (high confidence)

    # Add a more uncertain statement
    add_input_with_translation("<{Tweety} --> [fly]>. {0.6 0.4}")  # Tweety can fly (uncertain)
    
    # Add a negative statement
    add_input_with_translation("<penguin --> [fly]>. {0.1 0.9}")  # Penguins don't fly (confident negative)
    
    # Add a temporal relationship
    add_input_with_translation("<(&/, <$x --> bird>, <$x --> [fly]>) ==> <$x --> [healthy]>>. {0.8 0.7}")  # If a bird flies, it's healthy
    
    # Wait a moment to let NARS process
    time.sleep(0.5)
    
    # 2. ASKING QUESTIONS WITH ENHANCED TRANSLATIONS
    print_header("ASKING QUESTIONS (WITH NATURAL LANGUAGE BELIEF LEVELS)")
    
    # Simple inheritance question
    print("\nQuestion: Is Tweety an animal?")
    add_input_with_translation("<{Tweety} --> animal>?")
    
    # Question about an unknown property
    print("\nQuestion: Are penguins white?")
    add_input_with_translation("<penguin --> [white]>?")
    
    # Question with expected negative answer
    print("\nQuestion: Can penguins fly?")
    add_input_with_translation("<penguin --> [fly]>?")
    
    # Question using variables
    print("\nQuestion: What can swim?")
    add_input_with_translation("<?x --> [swim]>?")
    
    # Question about health
    print("\nQuestion: Is Tweety healthy?")
    add_input_with_translation("<{Tweety} --> [healthy]>?")
    
    # Wait a moment to let NARS process
    time.sleep(0.5)
    
    # 3. VIEWING KNOWLEDGE WITH ENHANCED TRANSLATIONS
    print_header("EXPLORING NARS KNOWLEDGE (WITH ENHANCED TRANSLATIONS)")
    
    # List all concepts in the system
    print("All concepts in the system:")
    add_input_with_translation("*concepts")
    
    # Explore what we know about birds
    print("\nWhat is a bird? (What category does bird belong to?)")
    add_input_with_translation("<bird --> {?1}>?")
    
    print("\nWhat properties do birds have?")
    add_input_with_translation("<bird --> [?1]>?", "100")
    
    print("\nWhat are examples of birds?")
    add_input_with_translation("<?x --> bird>?")
    
    # Explore what we know about Tweety
    print("\nWhat is Tweety?")
    add_input_with_translation("<{Tweety} --> {?1}>?")
    
    print("\nWhat properties does Tweety have?")
    add_input_with_translation("<{Tweety} --> [?1]>?")
    
    # Explore rules/implications
    print("\nWhat happens if something flies?")
    add_input_with_translation("<(<?x --> [fly]>) ==> {?1}>?")
    
    # Check for any negative beliefs
    print("\nWhat doesn't fly?")
    add_input_with_translation("<?x --> [fly]>? {0.0 0.9}")
    
    # Try a deduction that requires multiple steps
    print("\nIs a penguin an animal?")
    add_input_with_translation("<penguin --> animal>?")
    
    # View NARS statistics
    print_header("NARS STATISTICS")
    add_input_with_translation("*stats")
    
    print("\nDemo complete")

if __name__ == "__main__":
    main()