class VectorDB:
    def __init__(self, embedding_model):
        if embedding_model is None:
            raise ValueError("embedding_model cannot be None")

        self.embedding_model = embedding_model
        self.vectorstore = None
        self.text_chunks = []

    def embed_texts(self, texts):
        print('texts type', type(texts))
        return self.embedding_model.encode(texts)

    def update_vectorstore(self, text_chunks):
        import faiss  # have to import it here to avoid segfault for some reason
        self.text_chunks = text_chunks
        embeddings = self.embed_texts(text_chunks)
        print('embeddings type', type(embeddings))
        print("Embeddings shape:", embeddings.shape)
        self.vectorstore = faiss.IndexFlatL2(embeddings.shape[1])
        self.vectorstore.add(embeddings)

    def search(self, query_text, k=1):
        query_embedding = self.embed_texts([query_text])
        distances, indices = self.vectorstore.search(query_embedding, k)
        return distances, indices

    def get_text_chunk(self, index):
        return self.text_chunks[index]

    # In the VectorDB class
    def is_initialized(self):
        return self.vectorstore is not None


