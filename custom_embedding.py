from transformers.models.auto.modeling_auto import AutoModel


class JinaEmbeddings:
    def __init__(self):
        self.model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en',
                                               trust_remote_code=True)

    def encode(self, texts):
        return self.model.encode(texts)

    def embed_documents(self, texts):
        return self.encode(texts)

    def embed_query(self, query_text):
        return self.encode([query_text])[0]

    def __call__(self, texts):
        return self.embed_documents(texts)

    def _call(self, texts):
        return self.embed_documents(texts)


