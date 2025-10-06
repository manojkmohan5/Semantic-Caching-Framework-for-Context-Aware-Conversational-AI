import numpy as np

class SemanticCache:
    def __init__(self, threshold=0.85):
        self.entries = []
        self.threshold = threshold

    def cosine_similarity(self, emb1, emb2):
        emb1, emb2 = np.array(emb1), np.array(emb2)
        if np.linalg.norm(emb1) == 0 or np.linalg.norm(emb2) == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

    def search(self, embedding, session_id):
        for emb, resp, sid, ctx in self.entries:
            if sid == session_id and self.cosine_similarity(embedding, emb) > self.threshold:
                return resp
        return None

    def add(self, embedding, response, session_id, context):
        self.entries.append((embedding, response, session_id, context))
