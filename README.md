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

1. **Text Vector Upserter (`