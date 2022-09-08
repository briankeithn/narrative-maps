import pandas as pd
import numpy as np
#import random
import os.path
import pickle

#np.seterr(all='raise')
from ast import literal_eval
from pulp import *
from pulp import GLPK_CMD
from math import log, exp, pi, sqrt, ceil
from sklearn.metrics.pairwise import cosine_similarity
import sklearn.cluster as cluster
import umap
import hdbscan
from scipy.spatial import distance
from urllib.parse import urlparse

#import tensorflow as tf
#import tensorflow_hub as hub
#from ast import literal_eval
#from operationsSI import add_node, add_edge, remove_node, remove_edge
#from packages.MDS import WMDS, simulate_interaction, optimize_weights, graph_position_scaling
from packages.globals import *
from ftfy import fix_encoding
from datetime import date, datetime, timezone
from time import time

from transformers import BertTokenizer, TFBertModel
from sentence_transformers import SentenceTransformer
import torch

np.random.seed(42)
#random.seed = 42
start_time = None
window_i_j = {}
window_j_i = {}

def create_LP(query, sim_table, membership_vectors, clust_sim_table, exp_temp_table, ent_table, numclust, relevance_table,
    K, mincover, sigma_t, credibility=[], bias=[], operations=[],
    has_start=True, has_end=False, window_time=None, cluster_list=[], start_nodes=[], verbose=True, force_cluster=True, previous_varsdict=None):
    n = len(query.index) #We can cut out everything after the end.
    # Variable names and indices
    var_i = []
    var_ij = []
    var_k = [str(k) for k in range(0,numclust)]

    for i in range(0, n): # This goes up from 0 to n-1.
        var_i.append(str(i))
        for j in window_i_j[i]:
            if i == j:
                print("ERROR IN WINDOW - BASE")
            var_ij.append(str(i) + "_" + str(j))

    # Linear program variable declaration.
    minedge = LpVariable("minedge", lowBound = 0, upBound = 1)
    node_act_vars = LpVariable.dicts("node_act", var_i, lowBound = 0, upBound = 1)
    node_next_vars = LpVariable.dicts("node_next", var_ij, lowBound = 0,  upBound = 1)
    clust_active_vars = LpVariable.dicts("clust_active", var_k, lowBound = 0, upBound = 1)

    # Create the 'prob' variable to contain the problem data
    prob = LpProblem("StoryChainProblem", LpMaximize)
    # The objective function is added to 'prob' first
    prob += minedge, "WeakestLink"

    # Check duplicates in interactions
    replace = [False] * len(operations)
    for idx, op in enumerate(operations):
        op_ref = op.split(":",1)[1]
        for k in range(idx):
            if op_ref == operations[k].split(":",1)[1]:
                replace[k] = True
    # Fixtures.
    min_threshold_edge = 0.1 / K
    min_threshold_node = 0.4 / K
    for idx, op in enumerate(operations):
        op_ref = op.split(":",1)[1]
        # Omit ACL in this part, handle separetely.
        if "ACL" in op:
            continue

        temp = ""
        if "-" in op_ref:
            i = int(op_ref.split("-",1)[0])
            j = int(op_ref.split("-",1)[1])
            temp = str(i) + "_" + str(j)
        else:
            i = int(op_ref)
            temp = str(i)
        op_ref = temp

        if "AN" in op:
            if verbose:
                print('AddedNode ' + str(op_ref))
            if not replace[idx]:
                if prob.constraints.get('RemovedNode' + str(op_ref)) is not None:
                    del prob.constraints['RemovedNode' + str(op_ref)]
                prob += node_act_vars[str(op_ref)] >= min_threshold_node, 'AddedNode' + str(op_ref) # Makes problem infeasible if we add too many, divide threhsold by K? By number of added constraints?
        elif "AE" in op:
            if verbose:
                print('AddedEdge ' + str(op_ref))
            if not replace[idx]:
                if prob.constraints.get('RemovedEdge' + str(op_ref)) is not None:
                    del prob.constraints['RemovedEdge' + str(op_ref)]
                if str(op_ref) not in node_next_vars.keys():
                    # Special case for time-based windows where the time difference is too big.
                    # The connection does not exist by default and you need to add the variable manually for the specific edge.
                    node_next_vars[str(op_ref)] = LpVariable("node_next_" + str(op_ref), lowBound=0, upBound=1)
                    if i == j:
                        print("ERROR IN WINDOW - AE")
                    var_ij.append(str(op_ref))#str(i) + "_" + str(j))
                    window_i_j[i].append(j)
                    window_j_i[j].append(i)
                prob += node_next_vars[str(op_ref)] >= min_threshold_edge, 'AddedEdge' + str(op_ref)
        elif "RN" in op:
            if verbose:
                print('RemovedNode ' + str(op_ref))
            if not replace[idx]:
                if prob.constraints.get('AddedNode' + str(op_ref)) is not None:
                    del prob.constraints['AddedNode' + str(op_ref)]
                if prob.constraints.get('AddedNodeCluster' + str(op_ref)) is not None:
                    del prob.constraints['AddedNodeCluster' + str(op_ref)]
                prob += node_act_vars[str(op_ref)] == 0, 'RemovedNode' + str(op_ref)
        elif "RE" in op:
            if verbose:
                print('RemovedEdge ' + str(op_ref))
            if not replace[idx]:
                if prob.constraints.get('AddedEdge' + str(op_ref)) is not None:
                    del prob.constraints['AddedEdge' + str(op_ref)]
                prob += node_next_vars[str(op_ref)] == 0, 'RemovedEdge' + str(op_ref)

    if force_cluster:
        min_threshold_edge = 0.05
        min_threshold_node = 0.01
        for cluster in cluster_list:
            cluster_sorted = sorted(cluster)
            print("Cluster:")
            print(cluster_sorted)
            for idx, event in enumerate(cluster_sorted):
                if verbose:
                    print('AddedNodeCluster ' + str(event))
                if prob.constraints.get('RemovedNode' + str(event)) is not None:
                    del prob.constraints['RemovedNode' + str(event)]
                if prob.constraints.get('AddedNodeCluster' + str(event)) is not None:
                    del prob.constraints['AddedNodeCluster' + str(event)]

                prob += node_act_vars[str(event)] >= min_threshold_node, 'AddedNodeCluster' + str(event)
                print("Filtered cluster " + str(cluster_sorted[(idx + 1):]))
                if len(cluster_sorted[(idx + 1):]) >= 1:
                    for j in cluster_sorted[(idx + 1):]:
                        op_ref = str(event) + "_" + str(j)
                        print("Forced connections: " + str(op_ref))
                        if str(op_ref) not in node_next_vars.keys():
                            # Special case for time-based windows where the time difference is too big.
                            # The connection does not exist by default and you need to add the variable manually for the specific edge.
                            if event == j:
                                print("ERROR IN WINDOW - ACL")
                            node_next_vars[str(op_ref)] = LpVariable("node_next_" + str(op_ref), lowBound=0, upBound=1)
                            var_ij.append(str(op_ref))
                            window_i_j[event].append(j)
                            window_j_i[j].append(event)
                            #print(window_i_j[i])
                            #print(window_j_i[i])
                    prob += lpSum([node_next_vars[str(event) + "_" + str(j)] for j in cluster_sorted[(idx + 1):]]) >= min_threshold_edge, 'InternalEdgeCluster' + str(event)
                    if verbose:
                        print('InternalEdgeCluster' + str(event) + ":" + str(cluster_sorted[(idx + 1):]))
    #for combination in itertools.combinations(cluster_list,2):
    #    for node_i in combination[0]:
    #        for node_j in combination[1]:
    #            if int(node_i) < int(node_j):
    #                prob += node_act_vars[str(op_ref)] >= min_threshold_node, 'AddedNode' + str(op_ref)

    # Chain restrictions
    if has_start:
        num_starts = len(start_nodes)
        if verbose:
            print("Start node(s):")
            print(start_nodes)
            print(has_start)
        if num_starts == 0: # This is the default when no list is given and it has a start.
            prob += node_act_vars[str(0)] == 1, 'InitialNode'
        else:
            if verbose:
                print("Added start node(s)")
                print("--- %s seconds ---" % (time() - start_time))
            initial_energy = 1.0 / num_starts
            for node in start_nodes:
                prob += node_act_vars[str(node)] == initial_energy, 'InitialNode' + str(node)

    if has_end:
        prob += node_act_vars[str(n - 1)] == 1, 'FinalNode'

    if verbose:
        print("Chain constraints created.")
        print("--- %s seconds ---" % (time() - start_time))
    prob += lpSum([node_act_vars[i] for i in var_i]) == K, 'KNodes'
    prob += lpSum([node_next_vars[ij] for ij in var_ij]) == K - 1, 'K-1Edges'
    if verbose:
        print("Expected length constraints created.")
        print("--- %s seconds ---" % (time() - start_time))


    if has_start:
        if verbose:
            print("Equality constraints.")
            print("--- %s seconds ---" % (time() - start_time))
        for j in range(1, n):
            if j not in start_nodes:
                prob += lpSum([node_next_vars[str(i) + "_" + str(j)] for i in window_j_i[j]]) >= node_act_vars[str(j)], 'InEdgeReq' + str(j)
            else:
                if verbose:
                    print("Generating specific starting node constraints.")
                    print("--- %s seconds ---" % (time() - start_time))
                prob += lpSum([node_next_vars[str(i) + "_" + str(j)] for i in window_j_i[j]]) == 0, 'InEdgeReq' + str(j)
    else:
        if verbose:
            print("Inequality constraints.")
            print("--- %s seconds ---" % (time() - start_time))
        for j in range(1, n):
            prob += lpSum([node_next_vars[str(i) + "_" + str(j)] for i in window_j_i[j]]) <= node_act_vars[str(j)], 'InEdgeReq' + str(j)
    if verbose:
        print("In-degree constraints created.")

    if has_end:
        if verbose:
            print("Equality constraints.")
            print("--- %s seconds ---" % (time() - start_time))
        for i in range(0, n - 1):
            prob += lpSum([node_next_vars[str(i) + "_" + str(j)] for j in window_i_j[i]]) == node_act_vars[str(i)], 'OutEdgeReq'  + str(i)
    else:
        if verbose:
            print("Inequality constraints.")
            print("--- %s seconds ---" % (time() - start_time))
        for i in range(0, n - 1):
            prob += lpSum([node_next_vars[str(i) + "_" + str(j)] for j in window_i_j[i]]) <= node_act_vars[str(i)], 'OutEdgeReq'  + str(i)
    if verbose:
        print("Out-degree constraints created.")
        print("--- %s seconds ---" % (time() - start_time))

    #for i in range(0, n - 2):
    #    for j in window_i_j[i][1:]:
    #        for k in range(i + 1, j):
    #            prob += node_next_vars[str(i) + "_" + str(j)] <= 1 - node_act_vars[str(k)], 'NoSkippingTrans'  + str(i) + "_" + str(j) + "_" + str(k)
    #if verbose:
    #    print("No-skip constraints created.")
    #    print("--- %s seconds ---" % (time() - start_time))

    # Coverage
    if numclust > 1:
        prob += lpSum([clust_active_vars[str(k)] for k in var_k]) >= numclust * mincover, "MinCover"
        for k in range(0, numclust):
            prob += clust_active_vars[str(k)] == lpSum([node_next_vars[str(i) + "_" + str(j)] * sqrt(membership_vectors[i, k] * membership_vectors[j, k]) for i in range(0, n - 1) for j in window_i_j[i]]), "CoverDef" + str(k)
        if verbose:
            print("Clustering constraints created.")
            print("--- %s seconds ---" % (time() - start_time))
    else:
        if verbose:
            print("No clustering constraints created - only 1 cluster.")
            print("--- %s seconds ---" % (time() - start_time))

    # Objective
    for i in range(0, n):
        for j in window_i_j[i]:
            coherence_weights = [0.5, 0.5]
            entity_multiplier = min(1 + ent_table[i, j], 2) # Five or more entities in common means double the connection strength.
            relevance_multiplier = (relevance_table[i] * relevance_table[j]) ** 0.5 # Geometric mean the relevances, multiply based on how far it is from 0.5.
            coherence = (sim_table[i, j] ** coherence_weights[0]) * (clust_sim_table[i, j] ** coherence_weights[1])
            weighted_coherence = min(coherence * entity_multiplier * relevance_multiplier, 1.0)
            prob += minedge <= 1 - node_next_vars[str(i) + "_" + str(j)] + weighted_coherence, "Objective" + str(i) + "_" + str(j)
    if verbose:
        print("Objective constraints created.")
        print("--- %s seconds ---" % (time() - start_time))

    if previous_varsdict:
        current_names = [v.name for v in prob.variables() if "node_act" in v.name]
        if verbose:
            print("Generated list of names.")
            print("--- %s seconds ---" % (time() - start_time))
        for k, v in previous_varsdict.items():
            if "node_act" in k and k in current_names:
                node_act_vars[k.replace("node_act_", "")].setInitialValue(v)

    if verbose:
        if previous_varsdict:
            print("Used previous solution as starting point.")
            print("--- %s seconds ---" % (time() - start_time))
        else:
            print("No previous solution available.")
            print("--- %s seconds ---" % (time() - start_time))
    # The problem data is written to an .lp file
    return prob

def extract_varsdict(prob):
    # We get all the node_next variables in a dict.
    varsdict = {}
    for v in prob.variables():
        if "node_next" in v.name or "node_act" in v.name:
            varsdict[v.name] = np.clip(v.varValue, 0, 1) # Just to avoid negative rounding errors.
    return varsdict


def build_graph_df_multiple_starts(query, varsdict, prune=None, threshold=0.01, cluster_dict={}, start_nodes=[]):
    n = len(query)
    if 'bias' in query.columns:
        graph_df = pd.DataFrame(columns=['id', 'adj_list', 'adj_weights', 'date', 'publication','title', 'text', 'url', 'bias', 'coherence'])
    else:
        graph_df = pd.DataFrame(columns=['id', 'adj_list', 'adj_weights', 'date', 'publication','title', 'text', 'url', 'coherence'])

    already_in = []
    for i in range(0, n):
        prob = []
        coherence = varsdict["node_act_" + str(i)]
        if coherence < threshold:
            continue
        coherence_list = []
        index_list = []
        for j in window_i_j[i]:
            name = "node_next_" + str(i) + "_" + str(j)
            prob.append(varsdict[name])
            coherence_list.append(varsdict["node_act_" + str(j)])
        idx_list = [window_i_j[i][idx] for idx, e in enumerate(prob) if round(e,8) != 0 and e >= threshold and coherence_list[idx] >= threshold] # idx + i + 1
        nz_prob = [e for idx, e in enumerate(prob) if round(e,8) != 0 and e >= threshold and coherence_list[idx] >= threshold]
        if prune:
            if len(idx_list) > prune:
                top_prob_idx = sorted(range(len(nz_prob)), key=lambda k: nz_prob[k])[-prune:]
                idx_list = [idx_list[j] for j in top_prob_idx]
                nz_prob = [nz_prob[idx] for idx in top_prob_idx]
        sum_nz = sum(nz_prob)
        nz_prob = [nz_prob[j] / sum_nz for j in range(0, len(nz_prob))]
        # If we haven't checked this one before we add it to the graph.
        url = str(query.iloc[i]['url'])
        if i in already_in or sum_nz > 0:
            if len(url) > 0:
                url = urlparse(url).netloc
            if not (graph_df['id'] == i).any():
                title = query.iloc[i]['title']
                for key, value in cluster_dict.items():
                    if str(i) in value:
                        title = "[" + str(key) + "] " + title
                outgoing_edges = [idx_temp for idx_temp in idx_list]
                #coherence = varsdict["node_act_" + str(i)]
                if 'bias' in query.columns:
                    graph_df.loc[len(graph_df)] = [i, outgoing_edges, nz_prob, query.iloc[i]['date'], query.iloc[i]['publication'],
                                                   title, '', query.iloc[i]['url'], query.iloc[i]['bias'], coherence]
                else:
                    graph_df.loc[len(graph_df)] = [i, outgoing_edges, nz_prob, query.iloc[i]['date'], query.iloc[i]['publication'],
                                                   title, '', query.iloc[i]['url'], coherence]

            already_in += [i] + idx_list
    return graph_df

def check_extra_starts(graph_df, start_nodes):
    print("Start nodes")
    print(start_nodes)
    print(type(start_nodes[0]))
    # Start nodes should be properly defined at this point.
    all_nodes = set()
    has_incoming = set()
    for i, row in graph_df.iterrows():
        node_id = graph_df.at[i, 'id']
        all_nodes.add(node_id)
        adj_list = graph_df.at[i, 'adj_list']
        for j in adj_list:
            has_incoming.add(j)

    print("No incoming")
    print(list(all_nodes.difference(has_incoming)))
    if len(list(all_nodes.difference(has_incoming))) > 0:
        print(type(list(all_nodes.difference(has_incoming))[0]))

    extra_starts = all_nodes.difference(has_incoming).difference(set(start_nodes))
    return extra_starts

def graph_clean_up(graph_df, start_nodes=[]):
    if start_nodes is None:
        return graph_df # No changes because no start nodes were defined.
    if len(start_nodes) == 0:
        return graph_df # No changes because no start nodes were defined.
    extra_starts = check_extra_starts(graph_df, start_nodes)
    while len(extra_starts) > 0:
        graph_df = graph_df.drop(graph_df[graph_df['id'].isin(extra_starts)].index)
        extra_starts = check_extra_starts(graph_df, start_nodes)
    return graph_df


def compute_sim(query):#, is_WMDS=False):
    doc_list = []
    for index, article in query.iterrows():
        doc_list.append(article['embed'])

    X = np.array(doc_list)
    cluster_size_est = np.sqrt(len(query.index))/2
    cluster_size_est = 5 * round(cluster_size_est / 5) # Round to nearest multiple of 5

    n_neighbors = 2
    if len(query.index) > 40:
        n_neighbors = 10
    elif len(query.index) > 120:
        n_neighbors = cluster_size_est

    #if is_WMDS:
    #    clusterable_embedding, X_scaled = WMDS(X)
    #else:
    random_state = 42
    clusterable_embedding = umap.UMAP(n_neighbors=n_neighbors, random_state=np.random.RandomState(random_state)).fit_transform(X)

    cosine_sim = False
    if cosine_sim:
        similarities = np.clip(cosine_similarity(clusterable_embedding), -1 , 1)
        # Force normalize
        sim_table = (1 - np.arccos(similarities) / pi)
        sim_table = (sim_table - np.amin(sim_table)) / (np.amax(sim_table) - np.amin(sim_table))
    else:
        dist_mat = distance.cdist(clusterable_embedding, clusterable_embedding, 'euclidean')
        dist_mat = (dist_mat - np.min(dist_mat))/(np.ptp(dist_mat) - np.min(dist_mat))
        sim_table = np.clip(1 - dist_mat, -1, 1)

    return sim_table

def compute_temp_distance_table(query, dataset):
    n = len(query.index)
    filename = dataset + '_temp.npy'
    if os.path.isfile(filename):
        #print("Temporal distance table already exists.")
        temporal_distance_table = np.load(filename)
        if n == temporal_distance_table.shape[0]: # Must match, otherwise regenerate.
            return temporal_distance_table

    temporal_distance_table = np.zeros((n, n))
    for i in range(0, n - 1):
        for j in range(i, n):
            temporal_distance_table[i, j] = (query.iloc[j]['date'] - query.iloc[i]['date']).days # We are using days as the basic temporal unit.
    temporal_distance_table = np.maximum(temporal_distance_table, temporal_distance_table.T)

    np.save(filename, temporal_distance_table)
    return temporal_distance_table

def compute_sim_with_t(query, dataset, sigma_t=30):#, is_WMDS=False): # Assume no temporal similarity by default.
    sim_table = compute_sim(query)#, is_WMDS)
    if sigma_t == 0:
        return sim_table
    # Compute temporal distances
    temporal_distance_table = compute_temp_distance_table(query, dataset)
    exp_temp_table = np.exp(-temporal_distance_table / sigma_t)
    sim_table_with_t = sim_table * exp_temp_table
    return sim_table_with_t

def jaccard(x, y):
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    if union_cardinality == 0:
        return 0.0
    return intersection_cardinality / float(union_cardinality)

def get_entity_table(query, dataset):
    n = len(query.index)
    filename = dataset + '.npy'
    if os.path.isfile(filename):
        print("Entity table already exists.")
        ent_table = np.load(filename)
        with open(dataset + '.pickle', 'rb') as handle:
            ent_doc_list = pickle.load(handle)
        if n == len(ent_doc_list):
            return ent_table, ent_doc_list

    print("Entity table does not exist.")
    doc_list = []
    ent_doc_list = []
    ent_table = np.zeros((n, n))

    for index, article in query.iterrows():
        doc = nlp(article["title"])
        doc_list.append(doc)
        ent_doc_list.append([ent for ent in doc.ents]) # X.label_

    ent_doc_list_temp = [[ent.as_doc() for ent in ent_doc_ele] for ent_doc_ele in ent_doc_list]
    with open(dataset + '.pickle', 'wb') as handle:
        pickle.dump(ent_doc_list_temp, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Created entity-event list.")

    for i in range(0, n):
        ents_i = ent_doc_list[i]
        for j in range(0, n):
            if i != j:
                ents_j = ent_doc_list[j]
                pairwise_similarities = []
                for ent_i in ents_i:
                    ent_i_unrolled = [token.text for token in ent_i if not token.text in all_stopwords]
                    for ent_j in ents_j:
                        ent_j_unrolled = [token.text for token in ent_j if not token.text in all_stopwords]
                        final_sim = jaccard(ent_i_unrolled, ent_j_unrolled)
                        pairwise_similarities.append(final_sim)
                if len(pairwise_similarities) != 0:
                    ent_table[i,j] = np.max(pairwise_similarities)
                else:
                    ent_table[i,j] = 0.0
            else:
                ent_table[i, j] = 1.0
    print("Entity table has been created and saved.")
    np.save(filename, ent_table)
    return ent_table, ent_doc_list

def solve_LP_from_query(query, dataset,
    operations=[], focus_query="",
    window_time=None, K = 6, mincover=0.20,
    min_samples=2, min_cluster_size=2,
    n_neighbors=2, min_dist=0.0,
    sigma_t = 30,
    cred_check=False, start_nodes=[],
    verbose=True, random_state=42,
    force_cluster=True,
    use_entities=True, use_temporal=True, strict_start=False):#, is_WMDS=False):

    global start_time
    start_time = time()
    # Min-cluster-size = 10, n_neighbors=2
    doc_list = []
    for index, article in query.iterrows():
        doc_list.append(article['embed'])
    X = np.array(doc_list)

    # Find cluster list
    cluster_dict = {}
    for idx, op in enumerate(operations):
        if "ACL" in op:
            op_ref = op.split(":",1)[1]
            # Remove _out (this case only happens if you mark a story node before adding it to the map.)
            node_set = set([int(node.replace('_out', '')) for node in literal_eval(op_ref.split("-",1)[0])])
            cluster_id = int(op_ref.split("-",1)[1])
            if cluster_id not in cluster_dict:
                cluster_dict[cluster_id] = node_set
            else:
                for key, value in cluster_dict.items():
                    if key != cluster_id:
                        cluster_dict[key] = cluster_dict[key].difference(node_set) # Make sure we only keep the latest interaction.
                cluster_dict[cluster_id] = cluster_dict[cluster_id].union(node_set)
    cluster_list = []
    for key, value in cluster_dict.items():
        cluster_list.append([int(node) for node in value])
    if verbose:
        print("Cluster List (base) " + str(cluster_list))

    #if is_WMDS:
    #    clusterable_embedding, X_scaled = WMDS(X)
    #else:
    clusterable_embedding = umap.UMAP(n_neighbors=n_neighbors, random_state=np.random.RandomState(random_state)).fit_transform(X)
    #print(clusterable_embedding)
    if len(cluster_list) > 0:
        #if is_WMDS:
        #    subset_X_2d, subset_X_scaled = simulate_interaction(clusterable_embedding, X_scaled, cluster_list)
        #    weights = np.ones(subset_X_scaled.shape[1]) / subset_X_scaled.shape[1]
        #    weights, objectives, gradients, n_iter = optimize_weights(subset_X_2d, subset_X_scaled, weights, alpha=0.001, max_iter=10000, epsilon=1e-5, L=0.0, threshold=1e-5)
        #    weights = weights / np.sum(weights)
        #    clusterable_embedding, X_scaled = WMDS(X, weights)
        #else:
        target = -np.ones(X.shape[0])
        for clust_idx, cluster in enumerate(cluster_list):
            target[cluster] = clust_idx
        clusterable_embedding = umap.UMAP(n_neighbors=n_neighbors, random_state=np.random.RandomState(random_state)).fit_transform(X, y=target)
    if verbose:
        print("Computed projection.")
        print("--- %s seconds ---" % (time() - start_time))

    cosine_sim = False
    if cosine_sim:
        similarities = np.clip(cosine_similarity(clusterable_embedding), -1 , 1)
        # Force normalize
        sim_table = (1 - np.arccos(similarities) / pi)
        mask = np.ones(sim_table.shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        max_value = sim_table[mask].max()
        min_value = sim_table[mask].min()
        sim_table = (sim_table - min_value) / (max_value - min_value)
    else:
        dist_mat = distance.cdist(clusterable_embedding, clusterable_embedding, 'euclidean')

        mask = np.ones(dist_mat.shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        max_value = dist_mat[mask].max()
        min_value = dist_mat[mask].min()
        dist_mat = (dist_mat - min_value)/(max_value - min_value)
        sim_table = np.clip(1 - dist_mat, -1, 1)
    if verbose:
        print("Computed similarities.")
        print("--- %s seconds ---" % (time() - start_time))
    #with open('embed_' + dataset + '.npy', 'wb') as f:
    #    np.save(f, sim_table, allow_pickle=True)

    hdbscan_model = hdbscan.HDBSCAN(min_samples=min_samples,min_cluster_size=min_cluster_size,prediction_data=True)
    labels = hdbscan_model.fit_predict(clusterable_embedding)
    membership_vectors = hdbscan.prediction.all_points_membership_vectors(hdbscan_model)
    numclust = 1
    clust_sim = np.zeros((membership_vectors.shape[0], membership_vectors.shape[0]))
    if verbose:
        print("Computed clustering.")
        print("--- %s seconds ---" % (time() - start_time))
    #if 'cluster_vec' in query.columns:
    #    print("Pre-computed clusters.")
    #    cluster_vec_list = []
    #    for index, article in query.iterrows():
    #        cluster_vec_list.append(article['cluster_vec'])
    #    membership_vectors = np.array(cluster_vec_list)

    if len(membership_vectors.shape) > 1:
        numclust = membership_vectors.shape[1]
        membership_vectors[membership_vectors < 1/numclust] = 0
        #np.set_printoptions(threshold=sys.maxsize)
        # If any row is full of zeros replace with uniform distribution.
        membership_vectors[np.all(membership_vectors == 0, axis=1)] = np.ones(numclust) / numclust
        row_sums = membership_vectors.sum(axis=1)
        membership_vectors = membership_vectors / row_sums[:, np.newaxis]
        clust_sim = distance.cdist(membership_vectors, membership_vectors, lambda u,v:distance.jensenshannon(u,v,base=2.0))
    else:
        membership_vectors = np.ones((membership_vectors.shape[0], 1)) # No clusters special case (all noise according to HDBSCAN)

    clust_sim_table = 1 - clust_sim
    if verbose:
        print("Computed clustering similarities.")
        print("--- %s seconds ---" % (time() - start_time))

    n = len(query.index)
    # Compute temporal distances

    varsdict_filename = 'varsdict_' + dataset  + "_" + str(n) + '.pickle'

    temporal_distance_table = compute_temp_distance_table(query, dataset)

    if sigma_t != 0 and use_temporal:
        exp_temp_table = np.exp(-temporal_distance_table / sigma_t)
    else:
        exp_temp_table = np.ones(temporal_distance_table.shape)

    if verbose:
        print("Computed temporal distance table.")
        print("--- %s seconds ---" % (time() - start_time))

    window_time = None
    if sigma_t != 0 and use_temporal:
        window_time = sigma_t * 3 # Days

    if window_time is None:
        for i in range(0, n):
            window_i_j[i] = list(range(i + 1, n))
        for j in range(0, n):
            window_j_i[j] = list(range(0, j))
    else:
        for j in range(0, n):
            window_j_i[j] = []
        for i in range(0, n):
            window_i_j[i] = []
        for i in range(0, n - 1):
            window = 0
            for j in range(i + 1, n):
                if temporal_distance_table[i,j] <= window_time:
                    window += 1
            window = max(min(5, n - i), window)
            window_i_j[i] = list(range(i + 1, min(i + window, n)))
            for j in window_i_j[i]:
                window_j_i[j].append(i)

    if verbose:
        print("Computed temporal windows.")
        print("--- %s seconds ---" % (time() - start_time))

    ent_table, ent_doc_list = get_entity_table(query, dataset)
    if verbose:
        print("Computed entity similarities.")
        print("--- %s seconds ---" % (time() - start_time))
    if not use_entities:
        ent_table = np.zeros(ent_table.shape)

    relevance_table = [1.0] * membership_vectors.shape[0] # Create a vector full of 1s.
    if focus_query:
        #torch.cuda.empty_cache()
        model = SentenceTransformer('all-MiniLM-L6-v2')
        q = model.encode(focus_query)
        q = q.reshape(1, -1)
        euc_dist = distance.cdist(q, X, metric='euclidean')
        euc_dist = (euc_dist - np.min(euc_dist)) / (np.max(euc_dist) - np.min(euc_dist))
        sim_list = 1 - euc_dist
        relevance_table = sim_list[0] # Remove one dimension.

    if verbose:
        print("Computed query relevance table.")
        print("--- %s seconds ---" % (time() - start_time))

    cred = {'default': 1.0}
    pol_orientation = {'default': 'default'}
    has_start = False
    if start_nodes is not None:
        has_start = (len(start_nodes) > 0)
    if verbose:
        print("Creating LP...")

    # Read previous solution and feed to LP. If none there is no previous solution.
    previous_varsdict = None
    if os.path.isfile(varsdict_filename):
        with open(varsdict_filename, 'rb') as handle:
            previous_varsdict = pickle.load(handle)

    prob = create_LP(query, sim_table, membership_vectors, clust_sim_table, exp_temp_table, ent_table, numclust, relevance_table,
        K=K, mincover=mincover, sigma_t=sigma_t,
        operations=operations, cluster_list=cluster_list,
        has_start=has_start, has_end=False,
        window_time=window_time, start_nodes=start_nodes, verbose=verbose, force_cluster=force_cluster, previous_varsdict=previous_varsdict)
    if verbose:
        print("Saving model...")
        print("--- %s seconds ---" % (time() - start_time))
    prob.writeLP("left_story.lp")
    if verbose:
        print("Solving model...")
        print("--- %s seconds ---" % (time() - start_time))
    prob.solve(PULP_CBC_CMD(mip=True, warmStart=True))#(GLPK_CMD(path = 'C:\\glpk-4.65\\w64\\glpsol.exe', options = ["--tmlim", "180"]))


    varsdict = extract_varsdict(prob)
    # Overwrite last solution.
    with open(varsdict_filename, 'wb') as handle:
        pickle.dump(varsdict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    graph_df = build_graph_df_multiple_starts(query, varsdict, prune=ceil(sqrt(K)), threshold=0.1/K, cluster_dict=cluster_dict)

    if verbose:
        print("Graph data frame construction...")
        print("--- %s seconds ---" % (time() - start_time))

    if strict_start:
        graph_df = graph_clean_up(graph_df, start_nodes)

    if verbose:
        print("Graph clean up...")
        print("--- %s seconds ---" % (time() - start_time))

    cluster_assignment = np.argmax(membership_vectors, axis=1)
    scatter_df = pd.DataFrame(data=clusterable_embedding, columns=['X', 'Y'])
    scatter_df['id'] = query['id']
    scatter_df['cluster_id'] = cluster_assignment
    scatter_df['cluster_id'] = scatter_df['cluster_id'].astype(str)
    scatter_df['title'] = query['title']

    if verbose:
        print("Scatter data frame construction...")
        print("--- %s seconds ---" % (time() - start_time))

    return [graph_df, (numclust, LpStatus[prob.status]), scatter_df, sim_table, clust_sim_table, ent_table, ent_doc_list, cluster_assignment]