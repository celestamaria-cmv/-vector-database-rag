# Embeddings and Text Vectors

## 1. Definition
An embedding is a numerical representation of an input. For text, an embedding model maps a sentence or passage to a fixed-length vector. The goal is to place texts with related meanings near one another in the vector space.

## 2. How text becomes an embedding
A simplified workflow is:
1. Prepare text, such as a sentence or chunk.
2. Tokenize the text into model inputs.
3. Pass the inputs through a trained neural model.
4. Obtain a fixed-size vector representation.
5. Store the vector with the original text and metadata.

The vector values are not a human-readable summary. They are learned numerical features used for comparison and retrieval.

## 3. The project's model: all-MiniLM-L6-v2
The Sentence Transformers model `sentence-transformers/all-MiniLM-L6-v2` is intended to encode sentences and short paragraphs. Its model card reports 384-dimensional embeddings and says inputs longer than 256 word pieces are truncated by default. This makes sensible chunking important: chunks should fit the model's input limit.

## 4. Same model for documents and queries
For meaningful vector comparison, document chunks and user queries should be embedded in a compatible vector space. In this project, use the same `all-MiniLM-L6-v2` model for both stored chunks and incoming questions. Changing embedding models generally requires re-embedding the stored content.

## 5. Embedding quality considerations
- Use clear, meaningful chunks rather than arbitrary fragments.
- Avoid sending empty or extremely short chunks.
- Keep the embedding model consistent.
- Record which model produced the embeddings.
- Test retrieval with realistic questions, not only examples copied from the documents.

## 6. Embeddings are not the database
The embedding model creates vectors. ChromaDB stores and indexes those vectors and associates them with documents and metadata. These are separate responsibilities.

## References
- Hugging Face model card: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Sentence Transformers usage: https://www.sbert.net/docs/sentence_transformer/usage/usage.html
