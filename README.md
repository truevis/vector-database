# Vector Database with SQLite and OpenAI Embeddings

A powerful tool for creating and searching through vector embeddings of text documents using SQLite and OpenAI's text embeddings. This project provides a complete solution for document vectorization, storage, and semantic search capabilities.

## Features

- **Document Processing**
  - Automatic text chunking and processing
  - Support for various text file formats (.txt, .md)
  - Smart text cleaning and semantic preservation
  - Batch processing with rate limiting
  - Progress tracking and logging

- **Vector Storage**
  - SQLite-based vector database
  - Efficient binary vector serialization
  - Upsert capability for document updates
  - Configurable storage locations

- **Semantic Search**
  - Cosine similarity-based vector search
  - Top-K results retrieval
  - Multiple output formats (human-readable and LLM-optimized)
  - GUI interface for easy searching

## How Vectorization Works

Text vectorization is a fundamental concept in modern natural language processing that converts words, phrases, or documents into numerical vectors in a high-dimensional space. This mathematical representation allows computers to understand and process language in ways that capture semantic meaning.

### Understanding Vector Embeddings

When text is converted into vectors, similar concepts end up closer together in the vector space. For example, as shown in the visualization below:

![Vector Space Example](vector%20sample.jpg)

This 3D visualization demonstrates how different concepts are organized in vector space:
- Linux distributions (blue dots) like Ubuntu, Fedora, and Debian cluster together
- Vegetables (green dots) like broccoli, pepper, and potato form their own cluster
- Emotions (red dots) like surprise, fun, and fear group separately

Each point in this space is actually a high-dimensional vector (typically 768 or 1536 dimensions), but we can visualize it in 3D to understand the relationships. The distance between points represents semantic similarity - closer points are more related concepts.

### How Our System Uses Vectors

1. **Embedding Generation**: When you input text, our system uses OpenAI's embedding models to convert it into high-dimensional vectors
2. **Vector Storage**: These vectors are efficiently stored in SQLite, preserving their mathematical relationships
3. **Similarity Search**: When you search, your query is converted to a vector and compared with stored vectors using cosine similarity
4. **Result Ranking**: The most similar vectors (closest in the high-dimensional space) are returned as search results

This mathematical approach allows for:
- Finding semantically similar documents even when they use different words
- Understanding context and meaning beyond simple keyword matching
- Grouping related concepts automatically
- Performing "fuzzy" searches that understand conceptual relationships

## Components

1. **Text Vector Upserter (`text_vector_upserter_sqlite.py`)**
   - Processes text documents and generates embeddings
   - Handles rate limiting for API calls
   - Manages file organization and logging
   - Features a GUI for folder selection and configuration

2. **Vector Retrieval Tool (`sqlite_vector_retreval_test.py`)**
   - Provides semantic search functionality
   - Supports both GUI and command-line interfaces
   - Configurable result formatting
   - Maintains search history and preferences

## Prerequisites

- Python 3.x
- OpenAI API key
- Required Python packages:
  ```
  openai
  numpy
  tkinter
  streamlit
  ```

## Setup

1. Clone the repository
2. Install required packages:
   ```bash
   pip install openai numpy tkinter streamlit
   ```
3. Set up your OpenAI API key in the Streamlit secrets file:
   - Create or edit `.streamlit/secrets.toml`
   - Add your OpenAI API key:
     ```toml
     OPENAI_API_KEY = "your-api-key-here"
     ```
   - Keep this file secure and never commit it to version control
4. Create necessary folders for:
   - Source documents
   - Processed documents
   - Database storage
   - Logs

## Usage

### Document Processing

1. Run the upserter script:
   ```bash
   python text_vector_upserter_sqlite.py
   ```
2. Follow the GUI prompts to select:
   - Source folder (containing text files)
   - Destination folder (for processed files)
   - Database folder

### Semantic Search

1. Run the retrieval script:
   ```bash
   python sqlite_vector_retreval_test.py
   ```

Command-line options:
```bash
python sqlite_vector_retreval_test.py --db path/to/vectors.db --query "your search query" --top_k 5 --format display
```

Options:
- `--db`: Path to the vectors database
- `--query`: Search query
- `--top_k`: Number of results (default: 5)
- `--format`: Output format (display/llm)

## Configuration

The system maintains a `config.json` file that stores:
- Last used folder locations
- Database paths
- Log file locations

## Rate Limiting

The system includes built-in rate limiting for OpenAI API calls:
- Token rate limiting (350,000 tokens per minute)
- Batch processing (2048 tokens per batch)
- Automatic wait times when approaching limits

## Best Practices

1. **Document Processing**
   - Organize documents in a clear folder structure
   - Use consistent file naming conventions
   - Monitor the logs for processing issues

2. **Search Optimization**
   - Use specific, focused search queries
   - Adjust top_k based on your needs
   - Choose appropriate output format for your use case

## Security Notes

- Store API keys securely using environment variables
- Regularly backup your vector database
- Monitor API usage and costs

## Limitations

- Maximum token limit per text chunk: 8191 tokens
- Rate limiting may affect processing speed
- SQLite database size depends on available storage

## Contributing

Feel free to submit issues and enhancement requests!

## License

WTFPL - Do What The Fuck You Want To Public License

This program is free software. It comes without any warranty, to the extent permitted by applicable law. 
You can redistribute it and/or modify it under the terms of the Do What The Fuck You Want To Public License, 
Version 2, as published by Sam Hocevar. See http://www.wtfpl.net/ for more details.