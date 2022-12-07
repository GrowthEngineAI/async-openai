from async_openai.schemas import *

class OpenAI:
    completions: CompletionSchema = CompletionSchema()
    edits: EditSchema = EditSchema()
    embeddings: EmbeddingSchema = EmbeddingSchema()

