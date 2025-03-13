import tkinter as tk
from tkinter import filedialog, messagebox
import os
from openai import OpenAI
import sqlite3
import struct
import numpy as np
import re
import shutil
import time
import json
import sys
import streamlit as st
from datetime import datetime
from collections import deque
from datetime import datetime, timedelta

try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception as e:
    print("Error: OpenAI API key not found in Streamlit secrets.")
    print("Please set up your .streamlit/secrets.toml file with OPENAI_API_KEY=your_key")
    sys.exit(1)

MAX_TOKENS = 8191  # Maximum tokens per request for text-embedding-3-large
TPM_LIMIT = 350000  # Tokens per minute limit
BATCH_SIZE = 2048  # Maximum batch size per request
WORDS_PER_TOKEN = 0.75  # Estimation ratio for words to tokens

# Token rate limiting
class TokenRateLimiter:
    def __init__(self, tpm_limit):
        self.tpm_limit = tpm_limit
        self.token_queue = deque()
        
    def wait_if_needed(self, num_tokens):
        current_time = datetime.now()
        
        # Remove tokens older than 1 minute from the queue
        while self.token_queue and (current_time - self.token_queue[0][1]) > timedelta(minutes=1):
            self.token_queue.popleft()
        
        # Calculate current TPM
        current_tpm = sum(tokens for tokens, _ in self.token_queue)
        
        # If adding these tokens would exceed TPM, wait
        if current_tpm + num_tokens > self.tpm_limit:
            oldest_time = self.token_queue[0][1]
            wait_seconds = 60 - (current_time - oldest_time).total_seconds()
            if wait_seconds > 0:
                time.sleep(wait_seconds)
                # Clear expired tokens after waiting
                self.token_queue.clear()
        
        # Add new tokens to the queue
        self.token_queue.append((num_tokens, current_time))

# Initialize rate limiter
rate_limiter = TokenRateLimiter(TPM_LIMIT)

def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text string based on word count."""
    # Split text into words and count them
    words = text.split()
    # Estimate tokens using the words-to-tokens ratio
    estimated_tokens = int(len(words) / WORDS_PER_TOKEN)
    # Add 10% buffer for safety
    return int(estimated_tokens * 1.1)

def chunk_text(text: str, max_tokens: int = MAX_TOKENS) -> list:
    """Split text into chunks that don't exceed max_tokens."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_token_estimate = 0
    tokens_per_word = 1 / WORDS_PER_TOKEN
    
    for word in words:
        # Estimate tokens for this word (including space)
        word_tokens = int((len(word) + 1) * tokens_per_word)
        
        if current_token_estimate + word_tokens > max_tokens:
            # Join current chunk and add to chunks
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_token_estimate = word_tokens
        else:
            current_chunk.append(word)
            current_token_estimate += word_tokens
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def get_embedding(text: str, model="text-embedding-3-large") -> list:
    """Get embeddings for text, handling token limits and rate limiting."""
    # Estimate tokens
    num_tokens = estimate_tokens(text)
    
    # Wait if needed for rate limiting
    rate_limiter.wait_if_needed(num_tokens)
    
    # Get embedding for full text
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def setup_folders():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Load last used values from config
    config_path = 'config.json'
    last_values = {}
    try:
        with open(config_path, 'r') as f:
            last_values = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        last_values = {}

    # Get source folder path using file dialog
    initial_source_dir = last_values.get('last_source_folder', os.path.expanduser("~"))
    folder_path = filedialog.askdirectory(
        title="Select Source Folder (containing text files)",
        initialdir=initial_source_dir
    )
    
    if not folder_path:
        print("No source folder selected. Exiting...")
        root.destroy()
        exit()

    # Get destination folder path using file dialog
    initial_dest_dir = last_values.get('last_destination_folder', os.path.dirname(folder_path))
    move_to_folder = filedialog.askdirectory(
        title="Select Destination Folder (for processed files)",
        initialdir=initial_dest_dir
    )
    
    if not move_to_folder:
        print("No destination folder selected. Exiting...")
        root.destroy()
        exit()

    # Get database folder path using file dialog
    initial_db_dir = last_values.get('last_db_folder', move_to_folder)
    db_folder = filedialog.askdirectory(
        title="Select Database Folder",
        initialdir=initial_db_dir
    )
    
    if not db_folder:
        print("No database folder selected. Exiting...")
        root.destroy()
        exit()

    # Calculate remaining paths
    log_file_path = os.path.join(move_to_folder, "log", "sqlite_vectors.log")
    db_path = os.path.join(db_folder, "vectors.db")

    # Show confirmation dialog with full absolute paths
    confirm_msg = (
        f"Selected paths:\n\n"
        f"Source: {os.path.abspath(folder_path)}\n"
        f"  - This folder contains text files (.txt and .md) to be processed\n"
        f"  - Files will be read, cleaned, and have embeddings generated\n\n"
        f"Destination: {os.path.abspath(move_to_folder)}\n"
        f"  - Processed files will be moved here after successful processing\n"
        f"  - Contains the log subfolder for processing records\n\n"
        f"Database: {os.path.abspath(db_path)}\n"
        f"  - SQLite database file storing text embeddings and related information\n\n"
        f"Log: {os.path.abspath(log_file_path)}\n"
        f"  - Records processing status, warnings, and errors"
    )
    if not messagebox.askyesno("Confirm Paths", confirm_msg):
        print("Operation cancelled by user.")
        root.destroy()
        exit()

    # Create necessary directories
    os.makedirs(move_to_folder, exist_ok=True)
    os.makedirs(db_folder, exist_ok=True)
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # Save the values to config
    last_values.update({
        'last_source_folder': folder_path,
        'last_destination_folder': move_to_folder,
        'last_db_folder': db_folder,
        'last_log_folder': log_file_path,
        'last_db_path': db_path
    })
    
    try:
        with open(config_path, 'w') as f:
            json.dump(last_values, f, indent=4)
    except Exception as e:
        print(f"Warning: Could not save config: {e}")

    root.destroy()
    return folder_path, move_to_folder, log_file_path, db_path

def init_db(db_path):
    db = sqlite3.connect(db_path)
    db.execute("""
    CREATE TABLE IF NOT EXISTS vectors (
        id TEXT PRIMARY KEY,
        vector BLOB NOT NULL,
        text TEXT,
        page TEXT
    );
    """)
    return db

def serialize_vector(vector):
    return struct.pack("%sf" % len(vector), *vector)

def insert_vector(db, vector_id, vector, text, page):
    serialized_vector = serialize_vector(vector)
    db.execute(
        "INSERT OR REPLACE INTO vectors (id, vector, text, page) VALUES (?, ?, ?, ?)",
        (vector_id, serialized_vector, text, page)
    )
    db.commit()

def reduce_text_preserving_semantics(content: str, max_tokens: int) -> str:
    """Reduce text while preserving the most semantically important content for search."""
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', content)
    
    # Remove very short or low-information sentences
    sentences = [s for s in sentences if len(s.split()) > 3]
    
    # Technical and specification terms to look for
    technical_pattern = r'\d+(?:\.\d+)?|[A-Z][a-z]+[A-Z]|[A-Z]{2,}|\d+[kKmMgGtT][bBwW]|\d+[vV]|[0-9]+%|\d+(?:mm|cm|m|ft|in)'
    spec_terms = (
        r'must|shall|required|important|specify|specification|standard|procedure|method|system|'
        r'install|equipment|material|device|circuit|power|voltage|current|phase|wire|cable|'
        r'safety|emergency|test|measure|control|monitor|maintain|operate|function|performance|'
        r'quality|compliance|code|regulation|requirement|guideline|protocol|assembly|component|'
        # NEC-specific terms
        r'article|section|nec|nfpa 70|class i|class ii|class iii|division 1|division 2|'
        r'listed|labeled|certified|conductor|raceway|conduit|junction box|grounding|bonding|'
        r'ground fault|overcurrent|short circuit|overload|disconnect|service entrance|panelboard|'
        r'gfci|afci|rcd|clearance|spacing|separation|accessible|readily accessible|working space|'
        r'support|securing|fastening|wet location|damp location|dry location|ampacity|rating|'
        r'capacity|awg|kcmil|interrupting rating|withstand rating|ampere|volt-ampere|kva|kw|'
        r'hazard|warning|danger|prohibited|not permitted|shall not|minimum|maximum|'
        r'not less than|not more than|approved|identified|suitable'
    )
    
    # Prioritize sentences that are likely to be more informative
    scored_sentences = []
    for i, sentence in enumerate(sentences):
        score = 0
        words = sentence.split()
        
        # Position score (earlier sentences slightly preferred)
        score += max(0, 1 - (i / len(sentences)))
        
        # Technical terms and measurements (+3)
        technical_matches = len(re.findall(technical_pattern, sentence))
        score += min(3, technical_matches)
        
        # Specification key phrases (+2 each, max +4)
        spec_matches = len(re.findall(spec_terms, sentence, re.I))
        score += min(4, spec_matches * 2)
        
        # Optimal length (not too short, not too verbose)
        if 5 <= len(words) <= 25:
            score += 1
        
        # Bonus for sentences with both technical terms and spec phrases
        if technical_matches > 0 and spec_matches > 0:
            score += 1
        
        scored_sentences.append((score, i, sentence))
    
    # Sort sentences by score, highest first
    scored_sentences.sort(reverse=True)
    
    # First pass: select highest scoring sentences
    selected_sentences = []
    current_tokens = 0
    
    for score, orig_index, sentence in scored_sentences:
        sentence_tokens = estimate_tokens(sentence)
        if current_tokens + sentence_tokens <= max_tokens:
            selected_sentences.append((orig_index, sentence))
            current_tokens += sentence_tokens
        else:
            break
    
    # Sort selected sentences by original position to maintain flow
    selected_sentences.sort()
    
    # Join sentences and return
    return ' '.join(sentence for _, sentence in selected_sentences)

def clean_up_text(content: str) -> str:
    # Remove (cid:1) and similar placeholders
    content = re.sub(r'\(cid:\d+\)', '', content)
    # Remove non-printable and control characters
    content = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', content)
    # Replace multiple spaces, tabs, or newlines with single space
    content = re.sub(r'[\s\n\r\t]+', ' ', content)
    # Remove leading/trailing whitespace
    content = content.strip()
    # Remove any non-ASCII characters
    content = re.sub(r'[^\x00-\x7F]+', '', content)
    
    return content

def clean_up_text_severe(content: str) -> str:
    # First apply basic cleaning
    content = clean_up_text(content)
    # Additional aggressive cleaning steps
    # Remove all punctuation except periods and hyphens
    content = re.sub(r'[^\w\s.-]', '', content)
    # Replace multiple periods with single period
    content = re.sub(r'\.{2,}', '.', content)
    # Remove spaces around periods
    content = re.sub(r'\s*\.\s*', '.', content)
    # Remove any remaining multiple spaces
    content = re.sub(r'\s+', ' ', content)
    
    return content

def move_file(file_path, move_to_folder):
    try:
        # Ensure the destination is a directory by checking and possibly creating it
        if not os.path.exists(move_to_folder):
            os.makedirs(move_to_folder)
        # Construct the destination path by combining the folder path and the basename of the file
        destination_path = os.path.join(move_to_folder, os.path.basename(file_path))
        # Move the file
        shutil.move(file_path, destination_path)
        print(f"Moved '{file_path}' to '{destination_path}'.\n___________________________________________")
    except Exception as e:
        print(f"Error moving file: {e}")

def log_processed_file(filename, log_file_path):
    with open(log_file_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f"{filename} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

def log_processed_err(err, log_file_path):
    with open(log_file_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f"{err} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

def log_warning(warning, log_file_path):
    message = f"WARNING: {warning} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    print(message)
    with open(log_file_path, 'a', encoding='utf-8') as log_file:
        log_file.write(message)

def log_error(error, log_file_path):
    message = f"ERROR: {error} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    print(message)
    with open(log_file_path, 'a', encoding='utf-8') as log_file:
        log_file.write(message)

def process_text_batch(texts: list, model="text-embedding-3-large") -> list:
    """Process a batch of texts, respecting token and batch size limits."""
    embeddings = []
    current_batch = []
    current_batch_tokens = 0
    
    for text in texts:
        num_tokens = estimate_tokens(text)
        
        # If text exceeds token limit, chunk it
        if num_tokens > MAX_TOKENS:
            chunks = chunk_text(text)
            text = chunks[0]  # Use first chunk that fits
            num_tokens = estimate_tokens(text)
        
        # Check if adding this text would exceed batch token limit
        if current_batch_tokens + num_tokens > MAX_TOKENS or len(current_batch) >= BATCH_SIZE:
            # Process current batch
            if current_batch:
                rate_limiter.wait_if_needed(current_batch_tokens)
                response = client.embeddings.create(input=current_batch, model=model)
                embeddings.extend([data.embedding for data in response.data])
                current_batch = []
                current_batch_tokens = 0
        
        # Add text to current batch
        current_batch.append(text)
        current_batch_tokens += num_tokens
    
    # Process remaining batch
    if current_batch:
        rate_limiter.wait_if_needed(current_batch_tokens)
        response = client.embeddings.create(input=current_batch, model=model)
        embeddings.extend([data.embedding for data in response.data])
    
    return embeddings

def read_file_content(file_path):
    """Read and return the content of a file, trying different encodings."""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
                return content
        except UnicodeDecodeError:
            continue
        except Exception as e:
            raise e
    
    # If all encodings fail, try binary read and decode with replacement
    try:
        with open(file_path, 'rb') as file:
            content = file.read()
            return content.decode('utf-8', errors='replace')
    except Exception as e:
        raise Exception(f"Failed to read file with any encoding: {str(e)}")

if __name__ == "__main__":
    # Get paths from UI
    folder_path, move_to_folder, log_file_path, db_path = setup_folders()
    
    # Initialize SQLite database
    db = init_db(db_path)
    
    # Initialize OpenAI client for embeddings
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Get all .txt and .md files from the folder
    text_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                 if os.path.isfile(os.path.join(folder_path, f)) and 
                 (f.lower().endswith('.txt') or f.lower().endswith('.md'))]
    text_files.sort()
    total_files = len(text_files)
    
    print(f"Found {total_files} text files (.txt and .md) to process.")
    
    for file_num, file_path in enumerate(text_files, 1):
        file_name_without_extension = os.path.splitext(os.path.basename(file_path))[0]
        print(f"\nProcessing file {file_num}/{total_files}: '{file_name_without_extension}'...")
        
        try:
            text_content = read_file_content(file_path)
            text_content = clean_up_text(text_content)
            
            try:
                # First attempt with normal cleaning
                embedding = get_embedding(text_content, model="text-embedding-3-large")
                
                # Display first few values of the embedding vector to show it's working
                vector_preview = ", ".join([f"{val:.6f}" for val in embedding[:5]])
                print(f"Generated embedding vector. First 5 values: [{vector_preview}, ...]")
                
                insert_vector(
                    db,
                    file_name_without_extension,
                    embedding,
                    text_content,
                    file_name_without_extension
                )
                log_processed_file(file_path, log_file_path)
                move_file(file_path, move_to_folder)
                print(f"Successfully processed '{file_name_without_extension}'")
                continue
                
            except Exception as e:
                log_warning(f"First attempt failed for {file_path}: {str(e)}", log_file_path)
                
                # Second attempt with severe cleaning
                try:
                    text_content = clean_up_text_severe(text_content)
                    embedding = get_embedding(text_content)
                    
                    # Display first few values of the embedding vector to show it's working
                    vector_preview = ", ".join([f"{val:.6f}" for val in embedding[:5]])
                    print(f"Generated embedding vector with severe cleaning. First 5 values: [{vector_preview}, ...]")
                    
                    insert_vector(
                        db,
                        file_name_without_extension,
                        embedding,
                        text_content,
                        file_name_without_extension
                    )
                    log_processed_file(file_path, log_file_path)
                    move_file(file_path, move_to_folder)
                    print(f"Successfully processed '{file_name_without_extension}' with severe cleaning")
                    
                except Exception as e:
                    log_error(f"Both attempts failed for {file_path}: {str(e)}", log_file_path)
            
        except Exception as e:
            log_error(f"Failed to read file {file_path}: {str(e)}", log_file_path)
            continue

    # Close database connection
    db.close()
    print("Processing complete.")

