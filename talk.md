# Storing Vector Embeddings in SQLite with Python

## Introduction (30 seconds)

- The key insight: You don't need specialized vector databases - SQLite works perfectly!
- This talk shows how Python + SQLite can power semantic search through vector embeddings
- Perfect for projects where simplicity and portability matter more than extreme scale

## Why SQLite for Vector Storage? (1 minute)

- **The Vector Storage Challenge**:

  - No need to pay Pinecone
  - Vector embeddings are high-dimensional arrays (768-1536 dimensions)
  - Need efficient storage and retrieval for similarity search
  - Most solutions require specialized databases or complex setups
- **SQLite Advantages**:

  - Built into Python's standard library - zero dependencies
  - Stores binary data efficiently with BLOB type
  - Single file database - perfect for portability
  - Surprisingly good performance for most use cases
  - Familiar SQL interface for queries

## Getting Embeddings from OpenAI (1 minute)

```python
import openai
import os
import time
from typing import List

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """Get a vector embedding from OpenAI API."""
    # Basic rate limiting
    time.sleep(0.1)
    
    try:
        response = openai.Embedding.create(
            input=text,
            model=model
        )
        # Extract the embedding vector from the response
        embedding = response['data'][0]['embedding']
        return embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        # Return empty vector as fallback
        return [0.0] * 1536  # Default size for text-embedding-3-small

def batch_get_embeddings(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """Process multiple texts in a single API call for efficiency."""
    try:
        response = openai.Embedding.create(
            input=texts,
            model=model
        )
        # Extract embeddings for all texts
        embeddings = [data['embedding'] for data in response['data']]
        return embeddings
    except Exception as e:
        print(f"Error in batch embedding: {e}")
        # Return empty vectors as fallback
        return [[0.0] * 1536 for _ in range(len(texts))]
```

## Implementation: Storing Vectors in SQLite (2 minutes)

### Database Schema

```python
def create_vector_table():
    with sqlite3.connect('vectors.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vectors (
                id INTEGER PRIMARY KEY,
                text TEXT,
                embedding BLOB,  # Binary storage for vector data
                metadata TEXT    # Optional JSON for additional info
            )
        ''')
        conn.commit()
```

### Converting Vectors to Binary for Storage

```python
def store_vector(text, vector):
    # Convert numpy array to binary for efficient storage
    vector_binary = np.array(vector, dtype=np.float32).tobytes()
  
    with sqlite3.connect('vectors.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO vectors (text, embedding) 
            VALUES (?, ?)
        ''', (text, vector_binary))
        conn.commit()
```

### Retrieving and Using Vectors

```python
def search_similar(query_vector, top_k=5):
    # Convert query to binary for comparison
    query_vector = np.array(query_vector, dtype=np.float32)
  
    with sqlite3.connect('vectors.db') as conn:
        cursor = conn.cursor()
        results = []
      
        # Retrieve all vectors and calculate similarity
        for row in cursor.execute('SELECT id, text, embedding FROM vectors'):
            # Convert binary back to numpy array
            db_vector = np.frombuffer(row[2], dtype=np.float32)
          
            # Calculate cosine similarity
            similarity = np.dot(query_vector, db_vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(db_vector)
            )
          
            results.append((row[0], row[1], similarity))
      
        # Return top matches
        return sorted(results, key=lambda x: x[2], reverse=True)[:top_k]
```

## Complete Example: From Text to Search (1 minute)

```python
# Process a document and search for similar content
def process_and_search():
    # 1. Get some text
    document = "Python is a high-level programming language known for its readability and versatility."
    query = "What programming languages are easy to read?"
    
    # 2. Get embeddings from OpenAI
    doc_embedding = get_embedding(document)
    
    # 3. Store in SQLite
    store_vector(document, doc_embedding)
    
    # 4. Later, search with a query
    query_embedding = get_embedding(query)
    results = search_similar(query_embedding)
    
    # 5. Display results
    for id, text, similarity in results:
        print(f"Similarity: {similarity:.4f} - {text}")
```

## Conclusion (30 seconds)

- SQLite is a surprisingly capable vector database for many use cases
- Perfect for prototyping, small-to-medium applications, and edge deployments
- Python + SQLite + NumPy = powerful semantic search with minimal complexity
- All code is open source - check out the GitHub repo: **https://github.com/truevis/vector-database**
- Questions? Find me after the talk!

Thank you! ขอบคุณครับ/คะ!
