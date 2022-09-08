from sentence_transformers import SentenceTransformer, util
import numpy as np
from math import pi

class MiniLMModel():
    def __init__(self, wanted_model='all-MiniLM-L6-v2'):
        self.sts_model = SentenceTransformer(wanted_model)

    def __call__(self, sent_pairs):
        all_sents = []
        for pair in sent_pairs:
            all_sents += [pair[0],pair[1]]

        embds = self.sts_model.encode(all_sents)
        scores = []
        for i in range(int(len(all_sents)/2)):
            scores.append(util.pytorch_cos_sim(embds[i*2], embds[i*2+1])[0][0])

        return np.array(scores)



