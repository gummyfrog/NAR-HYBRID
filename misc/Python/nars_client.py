"""
Interface to the NARS system
"""

import traceback
from typing import Dict, Any, Optional, Union

# Import the original NAR module functions
try:
    from NAR import AddInput, Reset
except ImportError:
    # Create stub functions if module not available
    def AddInput(input_str: str, Print: bool = False) -> Dict[str, Any]:
        """Stub for AddInput function when NAR module is not available."""
        print(f"[STUB] AddInput: {input_str}")
        return {"raw": f"STUB OUTPUT for: {input_str}"}

    def Reset() -> None:
        """Stub for Reset function when NAR module is not available."""
        print("[STUB] Reset NARS")

class NarsClient:
    """Client for interacting with the NARS system."""

    def __init__(self, verbose: bool = False):
        """Initialize NARS client.
        
        Args:
            verbose: Whether to print verbose output
        """
        self.verbose = verbose
    
    def reset(self) -> None:
        """Reset the NARS system."""
        if self.verbose:
            print("Resetting NARS...")
        Reset()
    
    def add_input(self, narsese: str, print_raw: bool = False) -> Dict[str, Any]:
        """Add input to NARS and return the output.
        
        Args:
            narsese: Narsese statement to add
            print_raw: Whether to print raw output
            
        Returns:
            Raw output from NARS
        """
        if not narsese or narsese.strip() == "":
            if self.verbose:
                print("Skipping empty Narsese")
            return {"raw": ""}
        
        try:
            if self.verbose:
                print(f"Adding to NARS: '{narsese}'")
            
            # Send the input to NARS
            raw_output = AddInput(narsese, Print=print_raw)
            
            if self.verbose and isinstance(raw_output, dict) and "raw" in raw_output:
                print(f"NARS responded with {len(raw_output['raw'])} characters")
            return raw_output
            
        except Exception as e:
            error_msg = f"Error adding to NARS: {e}"
            if self.verbose:
                print(error_msg)
                traceback.print_exc()
            return {"raw": error_msg}
    
    def run_cycles(self, cycles: int = 300) -> Dict[str, Any]:
        """Run inference cycles in NARS.
        
        Args:
            cycles: Number of inference cycles to run
            
        Returns:
            Output from running cycles
        """
        return self.add_input(str(cycles))
    
    def extract_knowledge(self) -> str:
        """Extract all knowledge from NARS as context.
        
        Returns:
            Knowledge extracted from NARS
        """
        if self.verbose:
            print("Extracting knowledge from NARS...")
        
        try:
            # This will get all concepts in NARS
            concepts_output = self.add_input("*concepts", print_raw=False)
            
            from truth_translator import process_nars_output
            knowledge = process_nars_output(concepts_output, with_colors=False)
            
            if self.verbose:
                print(knowledge)
                print(f"Extracted {len(knowledge)} characters of knowledge")
                
            return knowledge
            
        except Exception as e:
            error_msg = f"Error extracting knowledge: {e}"
            if self.verbose:
                print(error_msg)
                traceback.print_exc()
            return "No knowledge available"