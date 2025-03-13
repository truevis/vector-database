# Building a Vector Database with Python, SQLite, and OpenAI

## Introduction (30 seconds)

- This tool lets you create and search through vector embeddings of text documents
- Perfect for semantic search, document retrieval, and AI applications
- Built entirely with Python and SQLite - no specialized vector DB required!

## The Python Ecosystem at Work (1 minute)

- **Core Python Libraries**:

  - `sqlite3` for database operations - Python's built-in DB that's perfect for vector storage
  - `numpy` for efficient vector operations and cosine similarity calculations
  - `tkinter` for a simple GUI interface
  - `openai` for generating embeddings via API
  - `streamlit` for API key and if it will get turned into a Web app
- **Python's Strengths Leveraged**:

  - Easy file I/O operations for document processing
  - Binary data handling for vector serialization
  - Context managers for clean resource handling
  - Type hints for better code documentation

## Key Python Implementation Details (2 minutes)

### Document Processing

```python
def process_text_files(folder_path):
    for file in Path(folder_path).glob('**/*.txt'):
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()
        chunks = chunk_text(text, max_tokens=1000)
        for chunk in chunks:
            vector = get_embedding(chunk, model="text-embedding-3-small")
            store_in_sqlite(chunk, vector)
```

### Vector Storage with SQLite

```python
def store_in_sqlite(text, vector):
    # Convert numpy array to binary for efficient storage
    vector_binary = np.array(vector, dtype=np.float32).tobytes()
  
    with sqlite3.connect('vectors.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO vectors (text, embedding) 
            VALUES (?, ?)
        ''', (text, vector_binary))
        conn.commit()
```

### Semantic Search with Cosine Similarity

```python
def search(query, top_k=5):
    query_vector = get_embedding(query, model="text-embedding-3-small")
  
    with sqlite3.connect('vectors.db') as conn:
        cursor = conn.cursor()
        results = []
      
        for row in cursor.execute('SELECT id, text, embedding FROM vectors'):
            db_vector = np.frombuffer(row[2], dtype=np.float32)
            similarity = cosine_similarity(query_vector, db_vector)
            results.append((row[0], row[1], similarity))
      
        # Sort by similarity and return top_k
        return sorted(results, key=lambda x: x[2], reverse=True)[:top_k]
```

## Python-Specific Optimizations (1 minute)

- **Rate Limiting Implementation**:

  - Using Python's `time.sleep()` for API rate limiting
  - Implementing token counting with regex and custom functions
  - Batch processing with generator expressions
- **Error Handling**:

  - Try/except blocks for robust API interactions
  - Context managers for database connections
  - Graceful degradation with fallback options
- **Pythonic Code Patterns**:

  - List comprehensions for data transformation
  - Generator functions for memory efficiency
  - Decorators for rate limiting and logging

## Conclusion (30 seconds)

- Python makes building a vector database accessible to everyone
- No specialized knowledge of vector databases required
- Combines the simplicity of SQLite with the power of modern embeddings
- All code is open source - feel free to fork, modify, and contribute!
- Questions? Find me after the talk or check out the GitHub repo **https://github.com/truevis/vector-database**

Thank you! ขอบคุณครับ/คะ!
