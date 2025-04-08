#!/usr/bin/env python3
"""
Enhanced Narsese Translator for OpenNARS
Translates Narsese statements and their truth values (frequency and confidence)
into natural language descriptions.
"""

import re
from narsese_to_english import narseseToEnglish

# ANSI color codes (can be disabled with noColors arg)
GREEN = "\x1B[32m"
YELLOW = "\x1B[33m"
CYAN = "\x1B[36m"
MAGENTA = "\x1B[35m"
RED = "\x1B[31m"
BLUE = "\x1B[34m"
RESET = "\x1B[0m"
BOLD = "\x1B[1m"

def translate_truth_value(frequency, confidence, with_colors=True):
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
    
    # Set up color scheme
    if with_colors:
        color_blue = BLUE
        color_cyan = CYAN
        color_green = GREEN
        color_yellow = YELLOW
        color_red = RED
        color_reset = RESET
        color_bold = BOLD
    else:
        color_blue = color_cyan = color_green = color_yellow = color_red = color_reset = color_bold = ""
    
    # Describe confidence level
    if confidence < 0.05:
        confidence_desc = f"{color_blue}extremely uncertain{color_reset}"
    elif confidence < 0.2:
        confidence_desc = f"{color_blue}very uncertain{color_reset}"
    elif confidence < 0.4:
        confidence_desc = f"{color_cyan}somewhat uncertain{color_reset}"
    elif confidence < 0.6:
        confidence_desc = f"{color_cyan}moderately confident{color_reset}"
    elif confidence < 0.8:
        confidence_desc = f"{color_green}confident{color_reset}"
    elif confidence < 0.95:
        confidence_desc = f"{color_green}very confident{color_reset}"
    else:
        confidence_desc = f"{color_bold}{color_green}extremely confident{color_reset}"
    
    # Describe frequency (truth value)
    if frequency < 0.01:
        frequency_desc = f"{color_red}definitely false{color_reset}"
    elif frequency < 0.2:
        frequency_desc = f"{color_red}very unlikely{color_reset}"
    elif frequency < 0.4:
        frequency_desc = f"{color_yellow}somewhat unlikely{color_reset}"
    elif frequency < 0.45:
        frequency_desc = f"{color_yellow}slightly unlikely{color_reset}"
    elif frequency > 0.55 and frequency < 0.6:
        frequency_desc = f"{color_yellow}slightly likely{color_reset}"
    elif frequency >= 0.45 and frequency <= 0.55:
        frequency_desc = f"{color_yellow}uncertain{color_reset}"
    elif frequency < 0.8:
        frequency_desc = f"{color_green}somewhat likely{color_reset}"
    elif frequency < 0.99:
        frequency_desc = f"{color_green}very likely{color_reset}"
    else:
        frequency_desc = f"{color_bold}{color_green}definitely true{color_reset}"
    
    return f"{frequency_desc} ({confidence_desc})"

def translate_priority(priority, with_colors=True):
    """
    Translate priority/usefulness value into natural language.
    
    Args:
        priority: Priority value from NARS (typically between 0.0 and 1.0)
        with_colors: Whether to include ANSI color codes
    
    Returns:
        Natural language description of the priority
    """
    # Convert string value to float if necessary
    if isinstance(priority, str):
        priority = float(priority)
    
    # Set up color scheme
    if with_colors:
        color_blue = BLUE
        color_cyan = CYAN
        color_green = GREEN
        color_yellow = YELLOW
        color_red = RED
        color_reset = RESET
    else:
        color_blue = color_cyan = color_green = color_yellow = color_red = color_reset = ""
    
    # Describe priority level
    if priority < 0.01:
        return f"{color_blue}completely irrelevant{color_reset}"
    elif priority < 0.2:
        return f"{color_blue}very low importance{color_reset}"
    elif priority < 0.4:
        return f"{color_cyan}low importance{color_reset}"
    elif priority < 0.6:
        return f"{color_yellow}moderate importance{color_reset}"
    elif priority < 0.8:
        return f"{color_green}high importance{color_reset}"
    elif priority < 0.95:
        return f"{color_green}very high importance{color_reset}"
    else:
        return f"{color_red}critically important{color_reset}"

def parse_narsese_truth(line):
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
    
    return None, None

def parse_narsese_priority(line):
    """
    Extract priority value from a Narsese statement.
    
    Args:
        line: A line of NARS output potentially containing priority
        
    Returns:
        Priority value if found, otherwise None
    """
    priority_match = re.search(r"Priority=([0-9.]+)", line)
    if priority_match:
        return float(priority_match.group(1))
    return None

def enhanced_narsese_translation(line, with_colors=True):
    """
    Translate a line of Narsese output to natural language, including
    statement content and truth/priority values.
    
    Args:
        line: A line of NARS output
        with_colors: Whether to include ANSI color codes
    
    Returns:
        Natural language translation with belief strength description
    """
    # Get the basic English translation
    translation = narseseToEnglish(line) if with_colors else narseseToEnglish(line)
    
    # If no translation was produced, return the original line
    if not translation:
        return line
    
    # Extract truth values and priority
    frequency, confidence = parse_narsese_truth(line)
    priority = parse_narsese_priority(line)
    
    # Add belief strength description if truth values were found
    if frequency is not None and confidence is not None:
        translation += f" - {translate_truth_value(frequency, confidence, with_colors)}"
    
    # Add priority description if found
    if priority is not None:
        translation += f" [Relevance: {translate_priority(priority, with_colors)}]"
    
    return translation

def process_nars_output(output, with_colors=True):
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
        translated_lines = []
        
        for line in lines:
            if line.strip():
                translation = enhanced_narsese_translation(line, with_colors)
                translated_lines.append(translation)
        
        return "\n".join(translated_lines)
    
    # Handle string output
    elif isinstance(output, str):
        lines = output.split("\n")
        translated_lines = []
        
        for line in lines:
            if line.strip():
                translation = enhanced_narsese_translation(line, with_colors)
                translated_lines.append(translation)
        
        return "\n".join(translated_lines)
    
    # Return the original output if it's not a processable type
    return str(output)

# Example usage in a standalone script
if __name__ == "__main__":
    import sys
    
    # Check if colors should be disabled
    with_colors = "noColors" not in sys.argv
    
    # Process input lines from stdin
    for line in sys.stdin:
        translated = enhanced_narsese_translation(line, with_colors)
        if translated:
            print(translated)
            sys.stdout.flush()