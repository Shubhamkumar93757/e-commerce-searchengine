from sentence_transformers import SentenceTransformer
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, embedding_dim =512, model_name= 'sentence-transformers/paraphrase-MiniLM-L6-v2'):
        super().__init__()
        self.sbert= SentenceTransformer(model_name)
        
        self.sbert_dim= self.sbert.get_embedding_dimension()
        if self.sbert_dim != embedding_dim:
            self.projection= nn.Linear(self.sbert_dim, embedding_dim)
        else:
            self.projection= None

    def forward(self, texts):
        embeddings = self.sbert.encode(texts, convert_to_tensor= True)
        if self.projection:
            embeddings = self.projection(embeddings)
        return embeddings 


# test
model = TextEncoder(embedding_dim=512)
texts = ["blue silk dress with lace trim", "vintage leather jacket brown"]
import torch

# Wrap the call in no_grad()
with torch.no_grad():
    embeddings = model(texts)