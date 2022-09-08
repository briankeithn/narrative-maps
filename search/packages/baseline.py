import pandas as pd
import numpy as np
import os.path
import pickle

#np.seterr(all='raise')
from ast import literal_eval
from math import log, exp, pi, sqrt
from sklearn.metrics.pairwise import cosine_similarity
import sklearn.cluster as cluster
import umap
import hdbscan
from scipy.spatial import distance
from urllib.parse import urlparse

from packages.globals import *
from ftfy import fix_encoding
from datetime import date, datetime, timezone
from time import time

from transformers import BertTokenizer, TFBertModel
from sentence_transformers import SentenceTransformer

def search_dataset(search_query, query, dataset,
    min_samples=2, min_cluster_size=2,
    n_neighbors=2, min_dist=0.0,
    sigma_t = 30, verbose=True, random_state=42):

    global start_time
    start_time = time()
    bert_model = SentenceTransformer('all-MiniLM-L6-v2')
    if verbose:
        print("Loaded embedding model.")
        print("--- %s seconds ---" % (time() - start_time))
    # Min-cluster-size = 10, n_neighbors=2
    doc_list = []
    for index, article in query.iterrows():
        doc_list.append(article['embed'])
    X = np.array(doc_list)
    if verbose:
        print("Generated data set embedding list.")
        print("--- %s seconds ---" % (time() - start_time))
    q = bert_model.encode(search_query)
    q = q.reshape(1, -1)
    sim_list = 1 - np.arccos(1 - distance.cdist(q, X, metric='cosine')) / pi
    sim_list = sim_list[0] # Remove one dimension.
    #sorted_idx_list = np.argsort(sim_list)[0] # We extract the first "row"
    if verbose:
        print("Computed sorted list (search results).")
        print("--- %s seconds ---" % (time() - start_time))

    return sim_list