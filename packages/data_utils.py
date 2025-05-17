import pandas as pd
import numpy as np
from ast import literal_eval
import re
import os

# Handling dates
import datetime
from datetime import date, datetime, timezone
from dateutil.parser import parse

from packages.globals import *

def clean_publication(graph_df):
    publication_dict = {'www.bbc.com': 'bbc',
                    'www.aljazeera.com': 'ajz',
                    'www.nytimes.com': 'nyt',
                    'www.theguardian.com': 'tgn',
                    'edition.cnn.com': 'cnn',
                    'www.cnn.com': 'cnn',
                    'Reuters': 'reu',
                    'www.reuters.com': 'reu',
                    'Business Insider': 'bin',
                    'CNN': 'cnn',
                    'The Hill': 'hill',
                    'The New York Times': 'nyt',
                    'CNBC': 'cnbc',
                    'Breitbart': 'brb',
                    'www.breitbart.com': 'brb',
                    'Fox News': 'fox',
                    'www.foxnews.com': 'fox',
                    'CIA': 'cia',
                    'FBI': 'fbi',
                    'Army CID': 'cid',
                    'INS': 'ins',
                    'Sanctioned Intercepts': 'sai',
                    'NSA': 'nsa',
                    'abcnews.go.com': 'abc',
                    'apnews.com': 'apn',
                    'www.firstpost.com': 'fpost',
                    'theconversation.com': 'tcon',
                    'nypost.com': 'nyp',
                    'newscomworld.com': 'nwc',
                    'havanatimes.org': 'hvt',
                    'www.nbcnews.com': 'nbc',
                    'www.local10.com': 'l10',
                    'www.trtworld.com': 'trt',
                    'www.washingtonpost.com': 'wapo',
                    'www.xinhuanet.com': 'xinhua'
                    }
    graph_df['publication'] = graph_df['publication'].map(publication_dict).fillna('default')
    return graph_df

def is_path_edge(i, j, val_list):
    if len(val_list) == 0:
        return False
    for start, end in zip(val_list, val_list[1:]):
        if str(start) == str(i) and str(end) == str(j):
            return True
    return False

def centroid_tuples(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length


def generate_elements(graph_df, antichain, hub_nodes, sp, low_dim_projection=None,
    query=None, graph_precomputed_positions=[], story_list=[], story_names=[], unraveled_set=set(),
    elements=[], label_length=450):
    element_list = []

    has_query = (query is not None)

    for story_idx, story in enumerate(story_list):
        story_name_str = [""]
        if len(story_names) > 0:
            story_name_str = story_names[story_idx] # Use the idx to get the names.
        if story_name_str[0] == "Singleton storyline":
            story_name_str[0] = ""
            continue
        cur_dict = {"data": {"id": "story_" + str(story_idx), "label": story_name_str[0], "story": True}, "classes": 'story', "selectable": False}
        element_list.append(cur_dict)
    # Generate map elements.
    for index, row in graph_df.iterrows():
        date = pd.to_datetime(str(row['date'])).strftime("%m/%d/%Y")
        story_data = [(idx, story) for idx, story in enumerate(story_list) if str(row['id']) in story]
        story_data_list = []
        story_name_str = ["Unassigned"]
        story_data_tuple = story_data[0] # Get the only element (the index of the relevant story)
        story_data_list = story_data_tuple[1] # Get the story, not the idx.
        if len(story_names) > 0:
            story_name_str = story_names[story_data_tuple[0]] # Use the idx to get the names.
        if story_name_str[0] == "Unassigned":
            story_data_list = [] # Make sure it does not get assigned.
        label = date + ' - ' + row['title']
        if len(label) > label_length:
            label = label[:label_length]
        source = "/static/" + str(row['publication']) + ".svg"
        cur_dict = {"data": {"id": str(row['id']), "parent": "story_" + str(story_data_tuple[0]),"label": label, "full_text": row['title'], "date": date, "source": source, "type": "in_map", "story":story_data_list, "storyname":story_name_str[0]}, "classes": "n"}
        if low_dim_projection:
            cur_dict["position"] = {'x': graph_precomputed_positions[str(row['id'])][0], 'y': graph_precomputed_positions[str(row['id'])][1]}

        if str(row['id']) in antichain and str(row['id']) in hub_nodes:
            cur_dict['classes'] = "hub ac"
            cur_dict['data']['explanation'] = "This event is representative of its storyline and acts as a hub in the map."
        elif str(row['id']) in antichain:
            cur_dict['classes'] = "ac"
            cur_dict['data']['explanation'] = "This event is representative of its storyline."
        elif str(row['id']) in hub_nodes:
            cur_dict['classes'] = "hub"
            cur_dict['data']['explanation'] = "This event is structurally important, it acts as a hub in the map."

        for node in elements:
            if node.get('data').get('id') == str(row['id']):
                cluster_id = node['data'].get('cluster')
                if cluster_id is not None:
                    cur_dict['data']['cluster'] = int(cluster_id)
                    cur_dict['classes'] += (' ' + color_cluster[int(cluster_id) - 1])
                break
        element_list.append(cur_dict)

        for idx, adj in enumerate(row['adj_list']):
            weight = row['adj_weights'][idx]
            if "connection_type" not in graph_df:
                edge_label = ""
            else:
                edge_label = row["connection_type"][idx]
            if is_path_edge(row['id'], adj, sp):
                edge_info = {"data": { "id": str(row['id']) + "-" + str(adj), "source": str(row['id']),
                                              "target": str(adj), "label": edge_label, "weight": weight, "width": 1.2 + 1.8 * weight}, "classes": "sp"}
                element_list.append(edge_info)
            else:
                edge_info = {"data": { "id": str(row['id']) + "-" + str(adj), "source": str(row['id']),
                                              "target": str(adj), "label": edge_label, "weight": weight, "width": 0.2 + 1.8 * weight}, "classes": ""}
                element_list.append(edge_info)
    # Generate non-map elements
    if has_query:
        centroid = centroid_tuples(np.array(list(graph_precomputed_positions.values())))
        for index, row in query.iterrows():
            # If it's not in the map, we add this to the layout.
            if str(index) not in graph_precomputed_positions:
                date = pd.to_datetime(str(row['date'])).strftime("%m/%d/%Y")
                label = date + ' - ' + row['title']
                if len(label) > label_length:
                    label = label[:label_length]
                #if 'cluster' in query.columns:
                #    label += " [" + row['cluster'] + "]"
                source = "/static/" + str(row['publication']) + ".svg"
                if int(index) in unraveled_set:
                    cur_dict = {"data": {"id": str(index) + "_out", "label": label, "full_text": row['title'], "date": date, "source": source,
                            "type": "out_map", "story":[], "storyname":"Unassigned", "size": 1},
                            'position': {'x': low_dim_projection[index][0], 'y': low_dim_projection[index][1]},
                            "classes": "n"}
                else:
                    cur_dict = {"data": {"id": str(index) + "_out", "label": label, "full_text": row['title'], "date": date, "source": source,
                            "type": "out_map_hidden", "story":[], "storyname":"Unassigned", "size": 1},
                            'position': {'x': centroid[0], 'y': centroid[1]},
                            "classes": "n"}
                element_list.append(cur_dict)
    return element_list

def add_execution_id(element_list, execution_id):
    modified_exec_id = "_" + str(execution_id) + "I"
    for ele in element_list:
        ele['data']['id'] += modified_exec_id
        if "-" in ele['data']['id']: # edge, update target and source
            ele['data']['target'] += modified_exec_id
            ele['data']['source'] += modified_exec_id
            ele['data']['id'] = ele['data']['source'] + "-" + ele['data']['target']
        if 'parent' in ele['data']:
            ele['data']['parent'] += modified_exec_id
    return element_list

def remove_execution_id(element_list, execution_id):
    modified_exec_id = "_" + str(execution_id) + "I"
    for ele in element_list:
        ele['data']['id'] = ele['data']['id'][:-len(modified_exec_id)]
        if "-" in ele['data']['id']: # edge, update target and source
            ele['data']['target'] = ele['data']['target'][:-len(modified_exec_id)]
            ele['data']['source'] = ele['data']['source'][:-len(modified_exec_id)]
            ele['data']['id'] = ele['data']['source'] + "-" + ele['data']['target']
        if 'parent' in ele['data']:
            ele['data']['parent'] = ele['data']['parent'][:-len(modified_exec_id)]
    return element_list


def read_query(dataset, start_date=None, end_date=None, partial=False):
    if partial:
        columns_to_skip = 'embed'
        base_dir = os.path.dirname(__file__)  # Gets the directory of data_utils.py
        file_path = os.path.join(base_dir, "..", "data", f"{dataset}.csv")
        query = pd.read_csv(file_path, usecols=lambda x: x not in columns_to_skip)
    else:
        base_dir = os.path.dirname(__file__)  # Gets the directory of data_utils.py
        file_path = os.path.join(base_dir, "..", "data", f"{dataset}.csv")
        query = pd.read_csv(file_path)
    query = clean_publication(query)

    # Inferring datatime format (this might bring issues depending on the data!)
    query['date'] = pd.to_datetime(query['date'], infer_datetime_format=True) #8/22/2019 12:15

    if start_date is not None:
        query = query[(query['date'] >= pd.to_datetime(start_date, format='%Y-%m-%d'))]

    if end_date is not None:
        query = query[(query['date'] <= pd.to_datetime(end_date, format='%Y-%m-%d'))]

    if not partial:
        embed_list = ['embed']
        query[embed_list] = query[embed_list].replace(r'( )+', ',', regex=True)
        query[embed_list] = query[embed_list].replace(r'\[,', '[', regex=True)
        query[embed_list] = query[embed_list].replace(r',\]', ']', regex=True)
        query[embed_list] = query[embed_list].map(literal_eval).applymap(np.array)

        if 'cluster_vec' in query.columns: # Predefined clusters!
            query[['cluster_vec']] = query[['cluster_vec']].applymap(literal_eval).applymap(np.array)

    query.reset_index(inplace=True)
    query['id'] = [str(i) for i in list(query.index)]

    return query

def read_query_search(dataset, start_date=None, end_date=None):
    base_dir = os.path.dirname(__file__)  # Gets the directory of data_utils.py
    file_path = os.path.join(base_dir, "..", "data", f"{dataset}.csv")
    query = pd.read_csv(file_path)
    query = clean_publication(query)


    if dataset == 'cuba': # This is date time.
        query['date'] = pd.to_datetime(query['date'], format='%m/%d/%Y %H:%M') #8/22/2019 12:15
    else:
        query['date'] = pd.to_datetime(query['date'], format='%m/%d/%Y').dt.date

    if start_date is not None:
        query = query[(query['date'] >= pd.to_datetime(start_date, format='%Y-%m-%d'))]

    if end_date is not None:
        query = query[(query['date'] <= pd.to_datetime(end_date, format='%Y-%m-%d'))]

    embed_list = ['embed']
    query[embed_list] = query[embed_list].replace(r'( )+', ',', regex=True)
    query[embed_list] = query[embed_list].replace('\[,', '[', regex=True)
    query[embed_list] = query[embed_list].replace(',]', ']', regex=True)
    query[embed_list] = query[embed_list].applymap(literal_eval).applymap(np.array)

    if 'cluster_vec' in query.columns: # Predefined clusters!
        query[['cluster_vec']] = query[['cluster_vec']].applymap(literal_eval).applymap(np.array)
    query.reset_index(inplace=True)
    query['id'] = [str(i) for i in list(query.index)]

    return query