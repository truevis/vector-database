# Storing Vector Embeddings in SQLite with Python

## Introduction (30 seconds)
- Hello Bangkok Python Users Group! I'm excited to share a practical approach to vector embeddings
- The key insight: You don't need specialized vector databases - SQLite works perfectly!
- This talk shows how Python + SQLite can power semantic search through vector embeddings
- Perfect for projects where simplicity and portability matter more than extreme scale

## Why SQLite for Vector Storage? (1 minute)
- **The Vector Storage Challenge**:
  - Vector embeddings are high-dimensional arrays (768-1536 dimensions)
  - Need efficient storage and retrieval for similarity search
  - Most solutions require specialized databases or complex setups

- **SQLite Advantages**:
  - Built into Python's standard library - zero dependencies
  - Stores binary data efficiently with BLOB type
  - Single file database - perfect for portability
  - Surprisingly good performance for most use cases
  - Familiar SQL interface for queries

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

## Performance Optimizations (1 minute)

- **Binary Storage Efficiency**:
  - 1536-dimension vector as float32 = 6KB per vector
  - SQLite compression reduces storage footprint
  - Batch inserts for faster processing

- **Search Optimizations**:
  - In-memory database option for speed (`sqlite3.connect(':memory:')`)
  - Indexing metadata for filtered searches
  - Implementing pagination for large result sets

- **Scaling Considerations**:
  - Works well up to ~100K vectors on standard hardware
  - Can partition by creating multiple database files
  - Async processing with `aiosqlite` for better throughput

## Conclusion (30 seconds)
- SQLite is a surprisingly capable vector database for many use cases
- Perfect for prototyping, small-to-medium applications, and edge deployments
- Python + SQLite + NumPy = powerful semantic search with minimal complexity
- All code is open source - check out the GitHub repo: **https://github.com/truevis/vector-database**
- Questions? Find me after the talk!

Thank you! ขอบคุณครับ/คะ!
