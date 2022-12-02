import pandas as pd
import numpy as np
from ast import literal_eval
import re

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
        query = pd.read_csv("../data/" + str(dataset) + '.csv', usecols=lambda x: x not in columns_to_skip)
    else:
        query = pd.read_csv("../data/" + str(dataset) + '.csv')
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
        query[embed_list] = query[embed_list].replace('\[,', '[', regex=True)
        query[embed_list] = query[embed_list].replace(',]', ']', regex=True)
        query[embed_list] = query[embed_list].applymap(literal_eval).applymap(np.array)

        if 'cluster_vec' in query.columns: # Predefined clusters!
            query[['cluster_vec']] = query[['cluster_vec']].applymap(literal_eval).applymap(np.array)

    query.reset_index(inplace=True)
    query['id'] = [str(i) for i in list(query.index)]

    return query

def read_query_search(dataset, start_date=None, end_date=None):
    query = pd.read_csv("../data/" + str(dataset) + '.csv')
    query = clean_publication(query)


    # Inferring datatime format (this might bring issues depending on the data!)
    query['date'] = pd.to_datetime(query['date'], infer_datetime_format=True) #8/22/2019 12:15

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