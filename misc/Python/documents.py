#!/usr/bin/env python3
"""
Document ingestion script for NARS
Processes various document types and converts content to Narsese
"""

import os
import sys
import argparse
import time
from typing import List, Optional, Dict, Any, Tuple

from english_to_narsese_modular import EnglishToNarsese
from llm_client import LlmClient
from nars_client import NarsClient

# Try to import document processing libraries
try:
    import PyPDF2
    HAS_PDF = True
except ImportError:
    HAS_PDF = False
    print("Warning: PyPDF2 not found. PDF processing will be unavailable.")

try:
    from bs4 import BeautifulSoup
    import requests
    HAS_WEB = True
except ImportError:
    HAS_WEB = False
    print("Warning: BeautifulSoup/requests not found. Web scraping will be unavailable.")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Document Ingestion Tool for NARS")
    
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
        "--input",
        type=str,
        required=True,
        help="Input file or URL to process"
    )
    
    parser.add_argument(
        "--type",
        type=str,
        choices=["auto", "text", "pdf", "web"],
        default="auto",
        help="Input type (auto-detect by default)"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Maximum characters per chunk for processing"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output file to save the Narsese statements (optional)"
    )
    
    parser.add_argument(
        "--add-to-nars",
        action="store_true",
        help="Add statements directly to NARS"
    )
    
    parser.add_argument(
        "--load-knowledge",
        type=str,
        help="Load NARS knowledge from file before processing"
    )
    
    parser.add_argument(
        "--save-knowledge",
        type=str,
        help="Save NARS knowledge to file after processing"
    )
    
    parser.add_argument(
        "--inference-cycles",
        type=int,
        default=1000,
        help="Number of inference cycles to run after processing"
    )
    
    return parser.parse_args()

def detect_input_type(input_path: str) -> str:
    """Auto-detect the input type based on extension or URL."""
    if input_path.startswith(("http://", "https://")):
        return "web"
    
    _, ext = os.path.splitext(input_path)
    ext = ext.lower()
    
    if ext in [".txt", ".md", ".csv"]:
        return "text"
    elif ext == ".pdf":
        return "pdf"
    else:
        return "text"  # Default to text

def read_text_file(file_path: str) -> str:
    """Read content from a text file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def read_pdf_file(file_path: str) -> str:
    """Read content from a PDF file."""
    if not HAS_PDF:
        raise ImportError("PyPDF2 is required for PDF processing but not installed.")
    
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n\n"
    return text

def read_web_content(url: str) -> str:
    """Read content from a web page."""
    if not HAS_WEB:
        raise ImportError("BeautifulSoup and requests are required for web scraping but not installed.")
    
    response = requests.get(url)
    response.raise_for_status()  # Raise exception for bad responses
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.extract()
    
    # Get text
    text = soup.get_text()
    
    # Clean up text: break into lines and remove leading/trailing whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    
    return text

def split_into_chunks(text: str, max_length: int = 2000) -> List[str]:
    """Split text into manageable chunks for processing."""
    # First, split by paragraphs
    paragraphs = [p for p in text.split('\n') if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If paragraph itself exceeds max_length, split it into sentences
        if len(paragraph) > max_length:
            sentences = []
            # Simple sentence splitting (this could be improved)
            for sentence in paragraph.replace("! ", "!|").replace("? ", "?|").replace(". ", ".|").split("|"):
                if sentence.strip():
                    sentences.append(sentence.strip())
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 <= max_length:
                    current_chunk += sentence + " "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
        # If adding the paragraph doesn't exceed limit, add it
        elif len(current_chunk) + len(paragraph) + 1 <= max_length:
            current_chunk += paragraph + "\n"
        # Otherwise, start a new chunk
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n"
    
    # Add the last chunk if it has content
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def process_text(llm_client: LlmClient, converter: EnglishToNarsese, 
                 text: str, chunk_size: int = 2000, 
                 verbose: bool = False) -> List[str]:
    """Process text and convert to Narsese statements."""
    # Split text into manageable chunks
    chunks = split_into_chunks(text, chunk_size)
    
    if verbose:
        print(f"Split text into {len(chunks)} chunks")
    
    all_narsese_statements = []
    
    for i, chunk in enumerate(chunks):
        if verbose:
            print(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} characters)")
        
        # Extract facts from chunk
        facts = llm_client.extract_facts(chunk)
        
        if verbose:
            print(f"Extracted {len(facts)} facts from chunk {i+1}")
        
        # Convert facts to Narsese
        for fact in facts:
            narsese = converter.process_line(fact)
            if narsese and narsese.strip():
                all_narsese_statements.append(narsese)
                if verbose:
                    print(f"Fact: {fact}")
                    print(f"Narsese: {narsese}")
                    print("---")
    
    return all_narsese_statements

def main():
    """Run the document ingestion process."""
    args = parse_args()
    
    # Initialize NARS client
    print("Initializing NARS client...")
    nars_client = NarsClient(verbose=args.verbose)
    
    # Load existing knowledge if specified
    if args.load_knowledge:
        if os.path.exists(args.load_knowledge):
            print(f"Loading knowledge from {args.load_knowledge}...")
            result = nars_client.load_knowledge(args.load_knowledge)
            print(result.get("raw", "Loading complete"))
        else:
            print(f"Knowledge file not found: {args.load_knowledge}")
    
    # Determine input type
    input_type = args.type
    if input_type == "auto":
        input_type = detect_input_type(args.input)
    
    print(f"Processing {input_type} input: {args.input}")
    
    # Read content based on input type
    try:
        if input_type == "text":
            content = read_text_file(args.input)
        elif input_type == "pdf":
            if not HAS_PDF:
                print("Error: PyPDF2 library not found. Cannot process PDF files.")
                return
            content = read_pdf_file(args.input)
        elif input_type == "web":
            if not HAS_WEB:
                print("Error: BeautifulSoup/requests libraries not found. Cannot process web pages.")
                return
            content = read_web_content(args.input)
        else:
            print(f"Error: Unsupported input type: {input_type}")
            return
    except Exception as e:
        print(f"Error reading input: {e}")
        return
    
    print(f"Read {len(content)} characters of content")
    
    # Initialize components
    print("Initializing LLM client...")
    llm_client = LlmClient(model_name=args.model, verbose=args.verbose)
    
    print("Initializing English to Narsese converter...")
    converter = EnglishToNarsese(
        verbose=args.verbose,
        output_truth=True,
        eternal_output=False,
        disable_grammar_learning=True
    )
    
    # Process the text
    print("Processing content...")
    start_time = time.time()
    narsese_statements = process_text(
        llm_client, 
        converter, 
        content, 
        chunk_size=args.chunk_size,
        verbose=args.verbose
    )
    processing_time = time.time() - start_time
    
    print(f"Processing completed in {processing_time:.2f} seconds")
    print(f"Generated {len(narsese_statements)} Narsese statements")
    
    # Save to output file if specified
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as file:
                for statement in narsese_statements:
                    file.write(statement + "\n")
            print(f"Saved Narsese statements to {args.output}")
        except Exception as e:
            print(f"Error saving output: {e}")
    
    # Add to NARS if requested
    if args.add_to_nars:
        print("Adding statements to NARS...")
        
        for i, statement in enumerate(narsese_statements):
            nars_client.add_input(statement)
            # Brief pause to allow NARS to process
            time.sleep(0.1)
            
            # Show progress for large sets
            if args.verbose or (i+1) % 10 == 0:
                print(f"Added {i+1}/{len(narsese_statements)} statements")
        
        print("All statements added to NARS")
        # Run inference cycles
        if args.inference_cycles > 0:
            print(f"Running {args.inference_cycles} inference cycles...")
            nars_client.run_cycles(args.inference_cycles)
            print("Inference complete")
    
    # Save NARS knowledge if specified
    if args.save_knowledge:
        print(f"Saving knowledge to {args.save_knowledge}...")
        result = nars_client.save_knowledge(args.save_knowledge)
        print(result.get("raw", "Saving complete"))
    
    print("Processing complete")

if __name__ == "__main__":
    main()