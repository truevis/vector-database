import tkinter as tk
from tkinter import filedialog, simpledialog
import sqlite3
import struct
import numpy as np
from openai import OpenAI
import json
import argparse
import sys
import os
import streamlit as st

try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception as e:
    print("Error: OpenAI API key not found in Streamlit secrets.")
    print("Please set up your .streamlit/secrets.toml file with OPENAI_API_KEY=your_key")
    sys.exit(1)

def deserialize_vector(blob):
    return np.array(struct.unpack('f' * (len(blob)//4), blob))

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_embedding(text, model="text-embedding-3-large"):
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def select_db_file():
    root = tk.Tk()
    root.withdraw()

    # Load last used values from config
    config_path = 'config.json'
    try:
        with open(config_path, 'r') as f:
            last_values = json.load(f)
            initial_dir = last_values.get('last_db_folder', '.')
    except (FileNotFoundError, json.JSONDecodeError):
        initial_dir = '.'

    db_path = filedialog.askopenfilename(
        title="Select vectors.db file",
        initialdir=initial_dir,
        filetypes=[("SQLite DB files", "*.db"), ("All files", "*.*")]
    )
    
    # Save the selected path to config
    if db_path:
        try:
            last_values = {}
            try:
                with open(config_path, 'r') as f:
                    last_values = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            
            last_values['last_db_folder'] = os.path.dirname(db_path)
            
            with open(config_path, 'w') as f:
                json.dump(last_values, f, indent=4)
        except Exception as e:
            print(f"Warning: Could not save config: {e}")
    
    root.destroy()
    return db_path

def get_query_from_user():
    """Get search query from user via dialog box"""
    root = tk.Tk()
    root.withdraw()
    query = simpledialog.askstring("Vector Search", "Enter your search query:", initialvalue="")
    root.destroy()
    return query

def search_vectors(query, db_path, top_k=5):
    """
    Search for vectors similar to the query embedding.
    
    Args:
        query (str): The search query
        db_path (str): Path to the SQLite database
        top_k (int): Number of results to return
        
    Returns:
        list: Top k results as (similarity, id, text, page) tuples
    """
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get query embedding
    prompt_embedding = get_embedding(query)

    # Fetch all vectors from database
    cursor.execute("SELECT id, vector, text, page FROM vectors")
    results = cursor.fetchall()

    # Calculate similarities and store results
    similarities = []
    for row in results:
        vector_id, vector_blob, text, page = row
        vector = deserialize_vector(vector_blob)
        similarity = cosine_similarity(prompt_embedding, vector)
        similarities.append((similarity, vector_id, text, page))

    # Sort by similarity (highest first)
    similarities.sort(reverse=True)
    
    # Close connection
    conn.close()
    
    # Return top k results
    return similarities[:top_k]

def format_results_for_display(results, query):
    """Format search results for display"""
    output = [f"\nTop {len(results)} results for query: '{query}'\n"]
    
    for i, (similarity, vector_id, text, page) in enumerate(results, 1):
        # Truncate text if it's too long for display
        text_preview = text[:300] + "..." if len(text) > 300 else text
        
        output.append(f"\n{i}. Similarity: {similarity:.4f}")
        output.append(f"Document ID: {vector_id}")
        output.append(f"Page: {page}")
        output.append(f"Text preview: {text_preview}")
        output.append("-" * 80)
    
    return "\n".join(output)

def format_results_for_llm(results, max_tokens=4000):
    """
    Format search results for use as context in an LLM prompt.
    Optimizes for token usage by prioritizing higher similarity results.
    
    Args:
        results: List of (similarity, id, text, page) tuples
        max_tokens: Approximate maximum tokens to include
        
    Returns:
        str: Formatted context for LLM
    """
    context_parts = []
    total_chars = 0
    char_per_token = 4  # Approximation: ~4 chars per token
    max_chars = max_tokens * char_per_token
    
    for i, (similarity, vector_id, text, page) in enumerate(results, 1):
        # Format this result
        result_text = f"[Document {i}: {page}]\n{text}\n"
        
        # Check if adding this would exceed our token budget
        if total_chars + len(result_text) > max_chars:
            # If this is the first result, include at least a portion
            if i == 1:
                chars_remaining = max_chars - total_chars
                result_text = result_text[:chars_remaining] + "..."
                context_parts.append(result_text)
            break
        
        context_parts.append(result_text)
        total_chars += len(result_text)
    
    return "\n".join(context_parts)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Vector Search Tool")
    parser.add_argument("--db", help="Path to the vectors database")
    parser.add_argument("--query", help="Search query")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--format", choices=["display", "llm"], default="display", 
                        help="Output format: 'display' for human reading or 'llm' for LLM context")
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Get database path
    db_path = args.db
    if not db_path:
        db_path = select_db_file()
        if not db_path:
            print("No database selected. Exiting...")
            return

    # Get query
    query = args.query
    if not query:
        query = get_query_from_user()
        if not query:
            print("No query provided. Exiting...")
            return

    # Perform search
    results = search_vectors(query, db_path, args.top_k)
    
    # Format and display results
    if args.format == "display":
        print(format_results_for_display(results, query))
    else:  # llm format
        print(format_results_for_llm(results))
    
    return results

if __name__ == "__main__":
    main()
