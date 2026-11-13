# Vector Databases in Retrieval-Augmented Generation (RAG)

## 1. What is RAG?
Retrieval-Augmented Generation combines information retrieval with language-model generation. Instead of relying only on the model's pretrained knowledge, an application retrieves relevant material from an external knowledge base and supplies it as context for the answer.

## 2. Ingestion workflow
1. Load source files such as PDFs.
2. Extract text and useful metadata.
3. Split text into chunks.
4. Generate an embedding for each chunk.
5. Store each chunk, embedding, and metadata in a vector database.

## 3. Question-answering workflow
1. Receive the user's question.
2. Embed the question with the same compatible embedding model.
3. Search the vector database for the nearest chunks.
4. Select and format the retrieved context.
5. Send the question and context to the language model.
6. Generate an answer grounded in the retrieved sources.
7. Display source citations where available.

## 4. Why the vector database matters
The vector database is the retrieval layer of this workflow. It allows the system to locate semantically related passages even when the query uses different wording from the source. It also stores the text and metadata needed to provide traceable context.

## 5. Grounding and fallback behaviour
The language model should be instructed to answer from the retrieved context, cite sources, and say when the supplied evidence does not contain the answer. Retrieval can return weak matches, so the application should not assume that receiving K chunks means sufficient evidence was found.

## 6. Common failure points
- Poor PDF text extraction;
- chunks that are too large, too small, or split mid-topic;
- embedding model mismatch between ingestion and query time;
- incorrect distance metric or collection configuration;
- irrelevant retrieved chunks;
- missing source metadata;
- prompts that allow unsupported claims.

## 7. How to demonstrate this project
Use questions that are answerable from the knowledge base and questions that are not. Show the retrieved chunks, their source/page metadata, distances, and final answer. Compare K values such as 1, 3, and 5. This demonstrates the database and retrieval behaviour, not only the chatbot response.

## References
- Chroma concepts: https://cookbook.chromadb.dev/core/concepts/
- Sentence Transformers usage: https://www.sbert.net/docs/sentence_transformer/usage/usage.html
