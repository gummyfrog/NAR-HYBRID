"""
Enhanced Narsese Translator for OpenNARS
Translates Narsese statements and their truth values (frequency and confidence)
into natural language descriptions optimized for LLM consumption.
"""

import re
import sys
from typing import Tuple, Union, Dict, Any, Optional

# Import the original translator if available
try:
    from narsese_to_english import narseseToEnglish
except ImportError:
    def narseseToEnglish(narsese: str) -> Optional[str]:
        """Stub function if the original translator is not available."""
        return f"Translation of: {narsese}"

# ANSI color codes (can be disabled with noColors arg)
GREEN = "\x1B[32m"
YELLOW = "\x1B[33m"
CYAN = "\x1B[36m"
MAGENTA = "\x1B[35m"
RED = "\x1B[31m"
BLUE = "\x1B[34m"
RESET = "\x1B[0m"
BOLD = "\x1B[1m"

def get_frequency_descriptor(frequency: float) -> str:
    """Get plain text descriptor for frequency without colors.
    
    Args:
        frequency: Value between 0.0 and 1.0 representing truth frequency
        
    Returns:
        Natural language description of the frequency
    """
    if frequency < 0.01:
        return "DEFINITELY FALSE"
    elif frequency < 0.2:
        return "VERY UNLIKELY"
    elif frequency < 0.4:
        return "SOMEWHAT UNLIKELY"
    elif frequency < 0.45:
        return "SLIGHTLY UNLIKELY"
    elif frequency > 0.55 and frequency < 0.6:
        return "SLIGHTLY LIKELY"
    elif frequency >= 0.45 and frequency <= 0.55:
        return "UNCERTAIN"
    elif frequency < 0.8:
        return "SOMEWHAT LIKELY"
    elif frequency < 0.99:
        return "VERY LIKELY"
    else:
        return "DEFINITELY TRUE"

def get_confidence_descriptor(confidence: float) -> str:
    """Get plain text descriptor for confidence without colors.
    
    Args:
        confidence: Value between 0.0 and 1.0 representing confidence
        
    Returns:
        Natural language description of the confidence
    """
    if confidence < 0.05:
        return "EXTREMELY UNCERTAIN"
    elif confidence < 0.2:
        return "VERY UNCERTAIN"
    elif confidence < 0.4:
        return "SOMEWHAT UNCERTAIN"
    elif confidence < 0.6:
        return "MODERATELY CONFIDENT"
    elif confidence < 0.8:
        return "CONFIDENT"
    elif confidence < 0.95:
        return "VERY CONFIDENT"
    else:
        return "EXTREMELY CONFIDENT"

def translate_truth_value(frequency: float, confidence: float, with_colors: bool = False) -> str:
    """
    Translate frequency and confidence values into natural language certainty descriptions.
    
    Args:
        frequency: Value between 0.0 and 1.0 representing truth frequency
        confidence: Value between 0.0 and 1.0 representing confidence
        with_colors: Whether to include ANSI color codes
    
    Returns:
        Natural language description of the belief strength
    """
    # Convert string values to floats if necessary
    if isinstance(frequency, str):
        frequency = float(frequency)
    if isinstance(confidence, str):
        confidence = float(confidence)
    
    # Get descriptors in plain text
    frequency_desc = get_frequency_descriptor(frequency)
    confidence_desc = get_confidence_descriptor(confidence)
    
    # Set up color scheme if colors are enabled
    if with_colors:
        # Add colors based on the descriptors
        if "UNLIKELY" in frequency_desc or "FALSE" in frequency_desc:
            frequency_desc = f"{RED}{frequency_desc}{RESET}"
        elif "LIKELY" in frequency_desc or "TRUE" in frequency_desc:
            frequency_desc = f"{GREEN}{frequency_desc}{RESET}"
        else:
            frequency_desc = f"{YELLOW}{frequency_desc}{RESET}"
            
        if "UNCERTAIN" in confidence_desc:
            confidence_desc = f"{BLUE}{confidence_desc}{RESET}"
        elif "CONFIDENT" in confidence_desc:
            confidence_desc = f"{GREEN}{confidence_desc}{RESET}"
    
    return f"({frequency_desc} | {confidence_desc})"

def parse_narsese_truth(line: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract truth values (frequency and confidence) from a Narsese statement.
    
    Args:
        line: A line of NARS output potentially containing truth values
        
    Returns:
        Tuple of (frequency, confidence) if found, otherwise (None, None)
    """
    # Try to match frequency and confidence pattern
    truth_match = re.search(r"Truth: frequency=([0-9.]+), confidence=([0-9.]+)", line)
    if truth_match:
        frequency = float(truth_match.group(1))
        confidence = float(truth_match.group(2))
        return frequency, confidence
    
    # Try to match truth value in braces pattern {frequency confidence}
    brace_match = re.search(r"\{([0-9.]+) ([0-9.]+)\}", line)
    if brace_match:
        frequency = float(brace_match.group(1))
        confidence = float(brace_match.group(2))
        return frequency, confidence
    
    # Try to match raw numbers pattern (e.g., "swan is penguin. 1.000000 0.393204")
    raw_match = re.search(r"([0-9.]+) ([0-9.]+)$", line)
    if raw_match:
        frequency = float(raw_match.group(1))
        confidence = float(raw_match.group(2))
        return frequency, confidence
    
    return None, None

def enhanced_narsese_translation(line: str, with_colors: bool = True) -> Optional[str]:
    """
    Translate a single line of Narsese to enhanced natural language.
    
    Args:
        line: A line of NARS output
        with_colors: Whether to include ANSI color codes
        
    Returns:
        Enhanced natural language translation or None if no translation
    """
    # Skip comment lines (starting with //)
    if line.strip().startswith("//"):
        return None
    
    # Get the basic English translation
    translation = narseseToEnglish(line)
    
    # If no translation was produced, return None
    if not translation:
        return None
    
    # Extract truth values
    frequency, confidence = parse_narsese_truth(line)
    
    # Format the result with truth descriptors at the beginning
    if frequency is not None and confidence is not None:
        # Remove any numeric patterns that look like truth values from the translation
        # This matches patterns like: "0.900000 0.418605" at the end of a string
        translation = re.sub(r"\s+[0-9]+\.[0-9]+\s+[0-9]+\.[0-9]+\s*$", "", translation)
        # Also match any other stray decimal numbers at the end
        translation = re.sub(r"\s+[0-9]+\.[0-9]+\s*$", "", translation)
        
        # Capitalize the first letter of the statement
        if translation and len(translation) > 0:
            translation = translation[0].upper() + translation[1:]
        
        # Add truth descriptor at the beginning
        return f"{translate_truth_value(frequency, confidence, with_colors)} {translation}"
    
    # If no truth values found, just return the capitalized translation
    if translation and len(translation) > 0:
        return translation[0].upper() + translation[1:]
    
    return translation

def process_nars_output(output: Union[Dict[str, Any], str], with_colors: bool = True) -> str:
    """
    Process NARS output dictionary or string and translate to enhanced natural language.
    
    Args:
        output: NARS output (dict with 'raw' key or string)
        with_colors: Whether to include ANSI color codes
    
    Returns:
        Translated output as a string
    """
    # Handle dictionary output (from AddInput function)
    if isinstance(output, dict) and "raw" in output:
        lines = output["raw"].split("\n")
    # Handle string output
    elif isinstance(output, str):
        lines = output.split("\n")
    else:
        return str(output)
        
    translated_lines = []
    
    for line in lines:
        if line.strip():
            translation = enhanced_narsese_translation(line, with_colors)
            if translation:  # Only add non-None translations
                translated_lines.append(translation)
    
    return "\n".join(translated_lines)

# For standalone usage
if __name__ == "__main__":
    # Check if colors should be disabled
    with_colors = "noColors" not in sys.argv
    
    # Process input lines from stdin
    for line in sys.stdin:
        translated = enhanced_narsese_translation(line, with_colors)
        if translated:
            print(translated)
            sys.stdout.flush()