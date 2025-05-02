"""
Interface to the NARS system
"""

import os
import re
import traceback
from typing import Dict, Any, Optional, Union, List

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
            # Handle save knowledge command
            if narsese.startswith("*save"):
                parts = narsese.split(maxsplit=1)
                filename = parts[1].strip() if len(parts) > 1 else "nars_knowledge.nal"
                return self.save_knowledge(filename)
                    
            # Handle load knowledge command
            elif narsese.startswith("*load"):
                parts = narsese.split(maxsplit=1)
                filename = parts[1].strip() if len(parts) > 1 else "nars_knowledge.nal"
                return self.load_knowledge(filename)
            
            elif narsese.startswith("*dump"):
                concepts_output = self.add_input("*concepts", print_raw=False)
                return concepts_output

            # Normal NARS command processing
            if self.verbose:
                print(f"Adding to NARS: '{narsese}'")
            
            narsese = narsese.strip()

            if narsese.startswith("(") or narsese.startswith("["):
                if self.verbose:
                    print("Skipping statement starting with '(' or '['")
                return {"raw": ""}

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
    
    def save_knowledge(self, filename: str) -> Dict[str, Any]:
        """Save NARS knowledge to a file.
        
        Args:
            filename: Path to save the knowledge
            
        Returns:
            Result of the operation
        """
        if self.verbose:
            print(f"Saving NARS knowledge to {filename}...")
        
        try:
            # Create the directory if it doesn't exist
            directory = os.path.dirname(os.path.abspath(filename))
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            # Get all concepts from NARS
            concepts_output = self.add_input("*concepts", print_raw=False)
            
            # Extract all valid Narsese statements
            knowledge_lines = []
            
            if isinstance(concepts_output, dict) and "raw" in concepts_output:
                raw_output = concepts_output["raw"]
                
                # Process each line of the output
                for line in raw_output.split("\n"):
                    # Skip comments and empty lines
                    if line.startswith("//") or not line.strip():
                        continue

                    # try no need to match {}?
                    knowledge_lines.append(line)
            
            # Write to file
            if knowledge_lines:
                with open(filename, 'w', encoding='utf-8') as f:
                    for line in knowledge_lines:
                        f.write(line + "\n")
                
                if self.verbose:
                    print(f"Saved {len(knowledge_lines)} statements to {filename}")
                return {"raw": f"Saved {len(knowledge_lines)} statements to {filename}"}
            else:
                error_msg = "No knowledge statements found to save"
                if self.verbose:
                    print(error_msg)
                return {"raw": error_msg}
                
        except Exception as e:
            error_msg = f"Error saving knowledge: {e}"
            if self.verbose:
                print(error_msg)
                traceback.print_exc()
            return {"raw": error_msg}
    
    def load_knowledge(self, filename: str) -> Dict[str, Any]:
        """Load NARS knowledge from a file.
        
        Args:
            filename: Path to load the knowledge from
            
        Returns:
            Result of the operation
        """
        if not os.path.exists(filename):
            error_msg = f"Knowledge file not found: {filename}"
            if self.verbose:
                print(error_msg)
            return {"raw": error_msg}
        
        if self.verbose:
            print(f"Loading knowledge from {filename}...")
        
        try:
            # Read statements from file
            with open(filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Process and add each statement
            successful_loads = 0
            
            for line in lines:
                line = line.strip()
                
                # Skip comments and empty lines
                if not line or line.startswith("//"):
                    continue
                
                # Convert from loadable format to NARS format if needed
                if "%" in line:
                    # Extract statement and truth values
                    match = re.search(r"(.*)\s+%([0-9.]+);([0-9.]+)%", line)
                    if match:
                        statement = match.group(1)
                        frequency = match.group(2)
                        confidence = match.group(3)
                        
                        # Format for NARS
                        narsese = f"{statement} {{{frequency} {confidence}}}"
                    else:
                        narsese = line
                else:
                    narsese = line
                
                # Add to NARS
                result = self.add_input(narsese, print_raw=False)
                if result and "error" not in result.get("raw", "").lower():
                    successful_loads += 1
            
            result_msg = f"Loaded {successful_loads} statements from {filename}"
            if self.verbose:
                print(result_msg)
            return {"raw": result_msg}
                
        except Exception as e:
            error_msg = f"Error loading knowledge: {e}"
            if self.verbose:
                print(error_msg)
                traceback.print_exc()
            return {"raw": error_msg}
    
    def extract_knowledge(self) -> str:
        """Extract all knowledge from NARS as context.
        
        Returns:
            Knowledge extracted from NARS
        """
        if self.verbose:
            print("Extracting knowledge from NARS...")
        
        try:
            # Get concepts from NARS
            concepts_output = self.add_input("*concepts", print_raw=False)
            
            from truth_translator import process_nars_output
            knowledge = process_nars_output(concepts_output, with_colors=False)
            
            if self.verbose:
                print(f"Extracted {len(knowledge)} characters of knowledge")
                
            return knowledge
            
        except Exception as e:
            error_msg = f"Error extracting knowledge: {e}"
            if self.verbose:
                print(error_msg)
                traceback.print_exc()
            return "No knowledge available"