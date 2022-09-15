import pandas as pd
import numpy as np
import truecase
import math
import networkx as nx
import json
from math import log, exp, pi
from packages.globals import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import scipy.sparse as sp
from nltk.corpus import wordnet as wn
from nltk.tokenize import WordPunctTokenizer

# Taken from SHAP library decision_plot code.
def change_shap_base_value(base_value, new_base_value, shap_values) -> np.ndarray:
    """Shift SHAP base value to a new value. This function assumes that `base_value` and `new_base_value` are scalars
    and that `shap_values` is a two or three dimensional array.
    """
    # matrix of shap_values
    if shap_values.ndim == 2:
        return shap_values + (base_value - new_base_value) / shap_values.shape[1]

    # cube of shap_interaction_values
    main_effects = shap_values.shape[1]
    all_effects = main_effects * (main_effects + 1) // 2
    temp = (base_value - new_base_value) / all_effects / 2  # divided by 2 because interaction effects are halved
    shap_values = shap_values + temp
    # Add the other half to the main effects on the diagonal
    idx = np.diag_indices_from(shap_values[0])
    shap_values[:, idx[0], idx[1]] += temp
    return shap_values

def preprocess_text(documents):
    cleaned_documents = [doc.lower() for doc in documents]
    cleaned_documents = [doc.replace("\n", " ") for doc in cleaned_documents]
    cleaned_documents = [doc.replace("\t", " ") for doc in cleaned_documents]
    return cleaned_documents

def process_documents(text_list):
    processed_documents = []
    for text in text_list:
        doc = nlp(text)
        processed_documents.append(doc)
    return processed_documents

def generate_candidates(processed_documents):
    noun_phrases = []
    for doc in processed_documents:
        noun_phrases += list(doc.noun_chunks)
    return noun_phrases

def remove_verbs(candidates):
    filtered_candidates = []
    for candidate in candidates:
        has_verb = False
        for token in candidate:
            if token.pos_ == "VERB":
                has_verb = True
        if not has_verb:
            filtered_candidates.append(candidate)
    return filtered_candidates

def get_maximal_candidates(candidates):
    candidates = sorted(candidates, key=len, reverse=True)
    filtered_candidates = []
    for candidate in candidates:
        current_candidates = [filter_candidate.text.lower() for filter_candidate in filtered_candidates]
        if not any(candidate.text.lower() in s for s in current_candidates):
            filtered_candidates.append(candidate)
    return filtered_candidates


def proper_noun_score(noun_phrase):
    for token in noun_phrase:
        if token.pos_ == "PROPN":
            return 1
    else:
        return 0

def common_noun_score(noun_phrase):
    for token in noun_phrase:
        if token.pos_ == "NOUN":
            return 1
    else:
        return 0

def np_frequency_score(noun_phrase, text_list):
    # Note that this frequency is always at least 1, so log can't be undefined.
    freq = 0
    for text in text_list:
        if noun_phrase.text in text:
            freq += 1
    return freq

def word_frequency_score(noun_phrase, text_list):
    freq = 0
    for text in text_list:
        for token in noun_phrase:
            if token.text in text:
                freq += 1
    return freq

def word_abstractness(token):
    # We only care about nouns.
    if token.pos_ not in ["NOUN", "PROPN"]:
        return 0 # Won't count
    word = token.text
    synset_list = wn.synsets(word)
    concrete_freq = 0
    total_freq = 0
    for synset in synset_list:
        hypernym_path_list = synset.hypernym_paths()
        total_freq += len(hypernym_path_list)
        for hypernym_path in hypernym_path_list:
            if 'physical_entity' in [hypernym.name().split('.')[0] for hypernym in hypernym_path]:
                concrete_freq += 1
    if word.lower() == "coronavirus": # Need an augmented dictionary with special domain words.
        return 1
    if total_freq == 0:
        return 0 # Won't count.
    return 1 - concrete_freq / total_freq

def abstractness_score(noun_phrase):
    return max([word_abstractness(token) for token in noun_phrase])

def length_score(noun_phrase):
    return len(noun_phrase)

def sim_score(noun_phrase, processed_documents):
    np_doc = noun_phrase
    all_sims = []
    if not np_doc.has_vector:
        return 0.0
    for doc in processed_documents:
        all_sims.append(np_doc.similarity(doc))
    return sum(all_sims) / len(all_sims)

def overlap_score(noun_phrase, top_previous_list):
    if not top_previous_list:
        return 1
    # Subset case
    lower_case_list = [top.lower() for top in top_previous_list]
    if any(noun_phrase.text.lower() in s for s in lower_case_list):
        return -1 # It's a subset! Perfect overlap even if similarity says otherwise!
    # Non-subset case.
    np_doc = noun_phrase
    if not np_doc.has_vector:
        return 0.5 # Flip a coin when there is no info.
    all_sims = []
    for top in top_previous_list:
        doc = nlp(top)
        all_sims.append(np_doc.similarity(doc))
    return 1 - max(all_sims)

def pos_score(noun_phrase):
    return (1 - noun_phrase.start / len(noun_phrase.doc)) ** 2

def evaluate_candidates(candidates, text_list, processed_documents, top_previous_list=[],
                        proper_noun_weight=10,common_noun_weight=10,np_frequency_weight=5,
                        word_frequency_weight=8,abstractness_weight=10,length_weight=1,
                        similarity_weight=5, overlap_weight=30, pos_weight=5):
    # Special case of single event.
    #similarity_weight * sim_score(candidate, processed_documents) + \
    #similarity_weight +\
    if len(text_list) == 1:
        scored_candidates = []
        for candidate in candidates:
            pos_weight = max(40, pos_weight * 2)
            score = proper_noun_weight * proper_noun_score(candidate) + \
                    common_noun_weight * common_noun_score(candidate) + \
                    abstractness_weight * abstractness_score(candidate) + \
                    length_weight * length_score(candidate) + \
                    overlap_weight * overlap_score(candidate, top_previous_list) + \
                    pos_weight * pos_score(candidate)
            weight_sum = proper_noun_weight + common_noun_weight + \
                         abstractness_weight + length_weight + \
                         overlap_weight + pos_weight
            score /= weight_sum
            scored_candidates.append((candidate, score, overlap_score(candidate, top_previous_list)))
        return sorted(scored_candidates, key=lambda x: x[1], reverse=True)

    scored_candidates = []
    for candidate in candidates:
        score = proper_noun_weight * proper_noun_score(candidate) + \
                common_noun_weight * common_noun_score(candidate) + \
                np_frequency_weight * np_frequency_score(candidate, text_list) + \
                word_frequency_weight * word_frequency_score(candidate, text_list) + \
                abstractness_weight * abstractness_score(candidate) + \
                length_weight * length_score(candidate) + \
                similarity_weight * sim_score(candidate, processed_documents) + \
                overlap_weight * overlap_score(candidate, top_previous_list) + \
                pos_weight * pos_score(candidate)
        weight_sum = proper_noun_weight + common_noun_weight + \
                     np_frequency_weight + word_frequency_weight + \
                     abstractness_weight + length_weight + \
                     similarity_weight + overlap_weight + \
                     pos_weight
        score /= weight_sum
        scored_candidates.append((candidate, score))
    return sorted(scored_candidates, key=lambda x: x[1], reverse=True)


def generate_names(G, storylines, topn=3):
    story_names = []
    top_previous_list = []
    for story in storylines:
        if len(story) == 1: # Singleton storyline.
            story_names.append(["Singleton storyline"])
        else:
            text_list = G[G['id'].isin([int(n) for n in story])]['title'].tolist()#[G.nodes[node_id]['label'] for node_id in story]
            text_list = [truecase.get_true_case(text) for text in text_list]
            processed_documents = process_documents(text_list)
            candidates = generate_candidates(processed_documents)
            #candidates = remove_verbs(candidates)
            candidates = get_maximal_candidates(candidates)
            sorted_candidates = evaluate_candidates(candidates, text_list, processed_documents, top_previous_list)
            if len(sorted_candidates) > 0: # If list is not empty.
                top_previous_list.append(sorted_candidates[0][0].text) # Get text from top candidate.
            final_candidate_list = []
            for final_candidate in sorted_candidates[:topn]:
                final_candidate_list.append(truecase.get_true_case(str(final_candidate[0]))) # Capitalize before finishing.
            if len(final_candidate_list) == 0:
                final_candidate_list.append("No valid name was found for this storyline.")
            story_names.append(final_candidate_list)
    return story_names

def jaccard(x, y):
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    if union_cardinality == 0:
        return 0.0
    return intersection_cardinality / float(union_cardinality)

def sim_explanation(s1, s2):
    # Get the similarity score between the sentences
    #torch.cuda.empty_cache()
    sts_explain_model = ExplainableSTS('minilm', MAX_LENGTH=30)
    sim_score = sts_explain_model(s1, s2)
    # Get the explanation
    values = sts_explain_model.explain(s1, s2, plot=False)
    return values


def add_connection_types(query, graph_df, sim_table, clust_sim_table, ent_table, ent_doc_list, cluster_assignment):
    graph_df["connection_type"] = np.empty((len(graph_df), 0)).tolist()
    graph_df["entity_description"] = np.empty((len(graph_df), 0)).tolist()
    graph_df["topical_description"] = np.empty((len(graph_df), 0)).tolist()
    graph_df["similarity_description"] = np.empty((len(graph_df), 0)).tolist()
    graph_df["event_topic"] = np.empty((len(graph_df), 0)).tolist()

    cluster_description = []
    # https://stackoverflow.com/questions/64743583/which-10-words-has-the-highest-tf-idf-value-in-each-document-total

    # Global TF-IDF
    origin_documents = preprocess_text(query["title"].values.tolist())

    print("Computing cluster descriptions.")
    wt = WordPunctTokenizer().tokenize
    for cluster in range(np.max(cluster_assignment) + 1): # Number of columns is number of clusters.
        idx_list = np.where(cluster_assignment == cluster)[0]
        cluster_only = query.loc[idx_list]["title"]
        if len(cluster_only) > 0:
            # Count the words in a cluster
            documents_per_topic = cluster_only.values.tolist()
            concatenated_documents = preprocess_text(documents_per_topic)
            vectorizer_model = CountVectorizer(ngram_range=(1, 3), tokenizer=wt, stop_words=all_stopwords)
            vectorizer_model.fit(concatenated_documents)
            X_per_cluster = vectorizer_model.transform(concatenated_documents)
            X_origin = vectorizer_model.transform(origin_documents)
            words = vectorizer_model.get_feature_names_out()

            tfidf_global_transformer = TfidfTransformer()
            _global_tfidf = tfidf_global_transformer.fit_transform(X_origin)
            global_df = pd.DataFrame(_global_tfidf.toarray())
            global_df['Topic'] = cluster_assignment

            avg_global_df = global_df.groupby(['Topic'], as_index=False).mean()
            avg_global_df = avg_global_df.drop(labels='Topic', axis=1)
            _avg_global_tfidf = avg_global_df.values[cluster, :]

            # Local TF-IDF
            local_tfidf_transformer = TfidfTransformer()
            local_tfidf_transformer.fit_transform(X_per_cluster)
            _idfi = local_tfidf_transformer.idf_

            scores = _avg_global_tfidf * _idfi
            score_list = list(scores)
            score_word = sorted(zip(words, score_list), key=lambda x: x[1], reverse=True)
            no_stop_words = [(word, score) for word, score in score_word if word not in all_stopwords]

            #scores = normalize(scores, axis=1, norm='l1', copy=False)
            #scores = sp.csr_matrix(scores)
            top_n_max = 10
            top_n = min(top_n_max, len(no_stop_words))
            final_word_list = [word_score[0] + " (" + str(round(word_score[1], 2)) + ")" for word_score in no_stop_words[:top_n]]
            results = ", ".join(final_word_list)
            cluster_description.append(results)#[w for w in top_n if w not in all_stopwords])
        else:
            cluster_description.append(["N/A"])

    print("Generating explanations.")
    for idx, row in graph_df.iterrows():
        i = int(row['id'])
        common_ent_frags = set()
        cluster_i = cluster_assignment[i]
        set_i = set(row["title"].lower().split(' ')) | set(row["text"].lower().split(' '))
        ent_list_i = ent_doc_list[i]
        graph_df.at[idx,'event_topic'] = "Topic keywords and importance: " + cluster_description[cluster_i]#', '.join(cluster_description[cluster_i])
        for idx_adj, j in enumerate(row["adj_list"]):
            j = int(j)
            cluster_j = cluster_assignment[j]
            set_j = set(graph_df[graph_df["id"] == j]["title"].values[0].lower().split(' ')) | set(graph_df[graph_df["id"] == j]["text"].values[0].lower().split(' '))
            shared_tokens =  set_i & set_j
            ent_list_j = ent_doc_list[j]
            connection_type = 'Temporal'
            ent_string = ''

            # If coherence is high enough, we evaluate whether it's a similarity or topical connection.
            # If it's a similarity connection, we check whether it's specifically an entity-based connection.
            sim_ij = sim_table[i,j]
            clust_ij = clust_sim_table[i,j]
            connection_type = "Similarity"# [" + str(graph_df.at[idx,'similarity_description']) + "]" # Assume similarity by default.
            if clust_ij >= 0.5:
                connection_type = "Topical"# [" +  str(graph_df.at[idx,'topical_description']) + "]"
            if ent_table[i,j] >= 0.5:
                common_ent_frags = set()
                for ent_i in ent_list_i:
                    ent_i_unrolled = [token.text for token in ent_i if not token.text in all_stopwords]
                    for ent_j in ent_list_j:
                        ent_j_unrolled = [token.text for token in ent_j if not token.text in all_stopwords]
                        inter = set.intersection(*[set(ent_i_unrolled), set(ent_j_unrolled)])
                        if len(inter) > 0:
                            common_ent_frags.update(inter)
                connection_type += ", Entity" #[" + ent_string + "]"

            # If nothing else works, then it's simply a temporal connection.
            graph_df.at[idx,'connection_type'].append(connection_type)
            if cluster_i == cluster_j:
                graph_df.at[idx,'topical_description'].append("Source and Target share the same cluster/topic. Topic keywords and importance: " + cluster_description[cluster_i])
            else:
                graph_df.at[idx,'topical_description'].append("Source topic keywords and importance: " + cluster_description[cluster_i] + " Target topic keywords and importance: " + cluster_description[cluster_j])
            #graph_df.at[idx,'similarity_description'].append(', '.join([t for t in shared_tokens if t not in all_stopwords and t != '']))
            common_ent_frags = [ent for ent in common_ent_frags if ent.strip()]
            graph_df.at[idx,'entity_description'].append(', '.join(common_ent_frags))


    return graph_df, cluster_description