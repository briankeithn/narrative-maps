import pandas as pd
import numpy as np
import networkx as nx
from math import log, exp, sqrt
from sklearn.metrics import euclidean_distances

def build_graph(graph_df):
    G = nx.DiGraph()
    for index, row in graph_df.iterrows():
        G.add_node(str(row['id']), coherence=max(-log(row['coherence']),0))
        for idx, adj in enumerate(row['adj_list']):
            G.add_edge(str(row['id']), str(adj), weight=max(-log(row['adj_weights'][idx]),0))
    return G

def graph_stories(G):
    # Base case, return the nodes if there is 1 or fewer nodes left.
    if len(G.nodes()) == 0:
        return []
    if len(G.nodes()) == 1:
        return [list(G.nodes())]
    # Base case, return node singletons if there are no edges left.
    if len(G.edges()) == 0:
        return [[node] for node in G.nodes()]
    # Main case.
    # Get maximum likelihood chain.
    mlc = get_shortest_path(G)
    # Remove all nodes and adjacent edges to the maximum likelihood chain.
    H = G.copy()
    for node in mlc:
        H.remove_node(node)
    # Normalize outgoing edges to sum up to 1.
    H = normalize_graph(H)
    # Recursive call.
    return [mlc] + graph_stories(H)

def get_shortest_path(G):
    sources = [node for node, in_degree in G.in_degree() if in_degree == 0]
    targets = [node for node, out_degree in G.out_degree() if out_degree == 0]
    best_st = (sources[0], targets[0])
    try:
        best_val = nx.shortest_path_length(G, best_st[0], best_st[1], weight='weight') + G.nodes[best_st[0]]['coherence'] # Check? + vs *
    except nx.NetworkXNoPath:
        best_val = 100000
    for s in sources:
        for t in targets:
            try:
                current_val = nx.shortest_path_length(G, s, t, weight='weight') + G.nodes[s]['coherence']
            except nx.NetworkXNoPath:
                current_val = 100000
            if current_val < best_val:
                best_st = (s, t)
                best_val = current_val
    sp = nx.shortest_path(G, best_st[0], best_st[1], weight='weight')
    return sp

def normalize_graph(G):
    for node in G.nodes():
        llhs = [edge[2]['weight'] for edge in G.out_edges(node, data=True)]
        probabilities = [exp(-llh) for llh in llhs]
        sum_prob = sum(probabilities)
        probs = [prob / sum_prob for prob in probabilities]
        for idx,edge in enumerate(G.out_edges(node)):
            attrs = {edge: {'weigth': llhs[idx] , 'prob': probs[idx]}}
            nx.set_edge_attributes(G, attrs)
    return G

def maximum_antichain(antichain_list, antichain_op):
    m = len(max(antichain_list, key=len))
    idx_list = [i for i, j in enumerate(antichain_list) if len(j) == m]
    if antichain_op == 0:
        max_list = min([antichain_list[i] for i in idx_list], key=sum)
    else:
        max_list = max([antichain_list[i] for i in idx_list], key=sum)
    max_length = len(max_list)

    return max_list, max_length

def get_representative_landmarks(G, storylines, query, mode="ranked"):
    antichain = []
    # first is default.
    if mode == "last":
        antichain = [story[-1] for story in storylines] # get the last element in the story
    elif mode == "degree":
        degree_story = [[v for k, v in G.degree(story)] for story in storylines]
        max_degree_idx_list = [degrees.index(max(degrees)) for degrees in degree_story]
        antichain = [storylines[idx][max_degree_idx] for idx, max_degree_idx in enumerate(max_degree_idx_list)]
    elif mode == "centrality":
        explanation = []
        antichain = []
        num_landmarks = len([story for story in storylines if len(story) > 1]) # Count non-singleton storylines.
        #g_distance_dict = {(e1, e2): 1 / weight for e1, e2, weight in G.edges(data='weight')}
        #nx.set_edge_attributes(g, g_distance_dict, 'distance')
        #centrality = nx.closeness_centrality(G, distance='weight')
        centrality = nx.degree_centrality(G)
        centrality_df = pd.DataFrame.from_dict({'node': list(centrality.keys()), 'centrality': list(centrality.values())})
        centrality_df = centrality_df.sort_values('centrality', ascending=False)
        for idx in range(num_landmarks): # Get the top N landmarks based on centrality, where N = num_landmarks
            antichain.append(centrality_df.iloc[idx]['node'])
            explanation.append("This event was marked as important due to its position as a relevant -hub- in the map.")
    elif mode == "centroid":
        explanation = []
        antichain = []
        for story in storylines:
            if len(story) > 1: # We exclude singleton storyline (no relevant landmarks)
                node_embedding_list = [query.loc[query['id'] == node]['embed'].item() for node in story]
                node_embeddings = np.stack(node_embedding_list)
                centroid = node_embeddings.mean(axis=0)
                l1_distance = np.linalg.norm(node_embeddings - centroid, axis=1)
                idx_closest_node = np.argmin(l1_distance)
                antichain.append(story[idx_closest_node]) # need the ID, not the index
                explanation.append("This event was marked as important due to its -content- being representative of its corresponding storyline.")
    elif mode == "ranked":
        centrality = nx.betweenness_centrality(G, weight='weight')
        explanation = []
        for story in storylines:
            if len(story) > 1: # We exclude singleton storyline (no relevant landmarks)
                node_embedding_list = [query.loc[query['id'] == node]['embed'].item() for node in story]
                node_embeddings = np.stack(node_embedding_list)
                centroid = node_embeddings.mean(axis=0)
                l1_distance = np.linalg.norm(node_embeddings - centroid, axis=1)
                temp = l1_distance.argsort()
                ranks_dist = np.empty_like(temp)
                ranks_dist[temp] = np.arange(len(l1_distance))
                #idx_closest_node = np.argmin(l1_distance)
                #antichain.append(story[idx_closest_node])
                centrality_array = np.array([centrality[node] for node in story])
                temp = centrality_array.argsort()
                ranks_centrality = np.empty_like(centrality_array)
                ranks_centrality[temp] = np.arange(len(centrality_array))

                average_ranks = np.average([ranks_dist, ranks_centrality], axis=0)
                idx_min = np.where(average_ranks == average_ranks.min())[0]
                lowest_dist_rank_idx = idx_min[0]
                lowest_dist_rank = ranks_dist[lowest_dist_rank_idx]
                # This doesn't matter if there's only one min node.
                # If there are many we give priority to dist to break ties.
                for idx in idx_min:
                    if lowest_dist_rank > ranks_dist[idx]:
                        lowest_dist_rank_idx = idx
                        lowest_dist_rank = ranks_dist[idx]
                antichain.append(story[lowest_dist_rank_idx])
                if ranks_dist[lowest_dist_rank_idx] < ranks_centrality[lowest_dist_rank_idx]:
                    explanation.append("This event was marked as important due to its -content- being representative of its corresponding storyline.")
                elif ranks_dist[lowest_dist_rank_idx] == ranks_centrality[lowest_dist_rank_idx]:
                    explanation.append("This event was marked as important based on its -content- being representative of its corresponding storyline and acting as a relevant -hub- in the map.")
                else:
                    explanation.append("This event was marked as important due to its position as a relevant -hub- in the map.")
    else: # first
        antichain = [story[0] for story in storylines] # get the first element in the story
    return antichain

def force_layout(query, positions_dot, sim_table, top_k=0):
    k = None
    fixed_keys = None
    positions_dot_int = None
    if positions_dot: # If dictionary is not empty.
        positions_dot_int = {int(k): v for k,v in positions_dot.items()}
        fixed_keys = positions_dot_int.keys()
        fixed_values = list(positions_dot_int.values())

    unraveled_set = set()
    if top_k > 0 and fixed_keys is not None:
        sim_table[np.diag_indices_from(sim_table)] = 0.0

        top_k_inds = np.argsort(sim_table)[:, -1:-top_k - 1:-1]
        filter_fixed = top_k_inds[list(fixed_keys)]
        unraveled_set = set(filter_fixed.ravel())

        sim_table[:,list(fixed_keys)] = 0.0 # Remove fixed keys from computation, otherwise we get no results (since all similar stuff is already in).
        top_k_inds = np.argsort(sim_table)[:, -1:-top_k - 1:-1]
        filter_fixed = top_k_inds[list(fixed_keys)]
        unraveled_set = set(filter_fixed.ravel())

        n = len(unraveled_set) + len(fixed_keys)
        n_map = len(fixed_keys)
        ratio = min(1 - sqrt(n_map / n), 0.4)
        max_dist = np.max(euclidean_distances(np.array(fixed_values)))
        k = max_dist * ratio

    full_G = nx.convert_matrix.from_numpy_matrix(sim_table)
    full_G.remove_nodes_from(set(range(len(query.index))) - unraveled_set - set(fixed_keys))

    pos = nx.drawing.layout.spring_layout(full_G, k = k, pos=positions_dot_int, fixed=fixed_keys, weight='weight', seed=42)


    final_pos = dict()
    for k, v in pos.items():
        final_pos[k] = v

    return final_pos, unraveled_set