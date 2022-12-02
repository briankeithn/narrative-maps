import spacy
import string
from nltk.corpus import stopwords
from packages.sts_pair_explainer import ExplainableSTS

# Initialize spacy.
nlp = spacy.load("en_core_web_sm")
#nlp.add_pipe('sentence_bert', config={'model_name': 'all-MiniLM-L6-v2'})

all_stopwords = stopwords.words('english')
all_stopwords.append("\'")#nlp.Defaults.stop_words
all_stopwords += list(string.punctuation)
execution_id = 0

# Initialise the explainable sts metric
#sts_explain_model = ExplainableSTS('minilm', MAX_LENGTH=30)

color_cluster = ["blue","red", "orange", "purple", "yellow", "green"]#, "blue", "green", "red", "orange", "purple"]
color_id = ["#a6cee3", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"]#, "#b2df8a", "#1f78b4", "#33a02c", "#e31a1c", "#ff7f00","#6a3d9a"]

# Stylesheets for node and edges.
cyto_stylesheet = [
            # Group Selectors
            {'selector': '[type = "in_map"]',
             'style': { 'label': 'data(label)',
                       'text-wrap': 'wrap',
                       'text-max-width': 150,
                       'text-valign': 'center',
                       'text-halign': 'right',
                       'font-family': 'candara',
                       'font-size': '8px',
                       'background-color': '#ffffff',
                       'background-opacity': 0.0,
                       'background-image': 'data(source)',
                       'background-fit': 'contain'
                      }
            },
            {'selector': '[type = "out_map"]',
             'style': {
                       'background-color': '#d3d3d3',
                       'background-opacity': 0.5,
                      }
            },
            {'selector': '[type = "out_map_hidden"]',
             'style': {
                       'background-color': '#d3d3d3',
                       'background-opacity': 0.5,
                       'display': 'none'
                      }
            },
            {'selector': '[search = "T"]',
             'style': {
                        'label': 'data(label)',
                        'font-size': '8px',
                        'background-color': '#969696',
                        'line-color': '#969696',
                        'background-opacity': 0.7,
                        'text-wrap': 'wrap',
                        'text-max-width': 150,
                        'text-valign': 'center',
                        'text-halign': 'right',
                      }
            },
            {'selector': 'edge',
             'style': {'label': 'data(label)',
                       'line-style': 'solid',
                       'width':'data(width)',
                       'font-size': '7px',
                       'curve-style': 'bezier',
                       'target-arrow-color': 'black',
                       'target-arrow-shape': 'triangle',
                       'arrow-scale': 0.7,
                       'text-rotation': 'autorotate',
                       'text-margin-y': '5px',
                       'text-margin-x': '5px'}
            },
            # Class selectors
            {
                'selector': '.ac',
                'style': {
                    #'background-color': '#C7F6B6',
                    'shape': 'round-rectangle',
                    'border-color': 'red',
                    'border-width': 2,
                    'background-opacity': 0.7
                }
            },
            {
                'selector': '.hub',
                'style': {
                    #'background-color': '#C7F6B6',
                    'shape': 'round-rectangle',
                    'border-color': 'blue',
                    'border-width': 2,
                    'background-opacity': 0.7
                }
            },
            {
                'selector': '.hub.ac',
                'style': {
                    #'background-color': '#C7F6B6',
                    'shape': 'round-rectangle',
                    'border-color': 'purple',
                    'border-width': 2,
                    'background-opacity': 0.7
                }
            },
            {
                'selector': '.story',
                'style': {
                    'label': 'data(label)',
                    'font-size': '12px',
                    'text-halign':'center',
                    'text-valign':'top',
                }
            },
            {
                'selector': '.sp',
                'style': {
                    'line-color': 'blue',
                    'line-style': 'dashed'
                }
            },
            {'selector': '.lblue', 'style': { 'background-color': '#a6cee3', 'background-opacity': 0.7}},
            {'selector': '.blue', 'style': { 'background-color': '#1f78b4', 'background-opacity': 0.7}},
            {'selector': '.lgreen', 'style': { 'background-color': '#b2df8a', 'background-opacity': 0.7}},
            {'selector': '.green', 'style': { 'background-color': '#33a02c', 'background-opacity': 0.7}},
            {'selector': '.lred', 'style': { 'background-color': '#fb9a99', 'background-opacity': 0.7}},
            {'selector': '.red', 'style': { 'background-color': '#e31a1c', 'background-opacity': 0.7}},
            {'selector': '.lorange', 'style': { 'background-color': '#fdbf6f', 'background-opacity': 0.7}},
            {'selector': '.orange', 'style': { 'background-color': '#ff7f00', 'background-opacity': 0.7}},
            {'selector': '.lpurple', 'style': { 'background-color': '#cab2d6', 'background-opacity': 0.7}},
            {'selector': '.purple', 'style': { 'background-color': '#6a3d9a', 'background-opacity': 0.7}},
            {'selector': '.lyellow', 'style': { 'background-color': '#ffff99', 'background-opacity': 0.7}},
            {
                'selector': ":selected",
                'style': {
                    'label': 'data(label)',
                    'font-size': '8px',
                    'background-color': '#969696',
                    'line-color': '#CC0000',
                    'background-opacity': 0.7,
                    'text-wrap': 'wrap',
                    'text-max-width': 150,
                    'text-valign': 'center',
                    'text-halign': 'right',
                }
            }
        ]

styles = {
    'json-output': {
        'overflowY': 'scroll',
        'height': 'calc(50% - 25px)',
        'border': 'thin lightgrey solid'
    },
    'tab': {'height': 'calc(98vh - 115px)', 'padding-left': 10}
}


base_layout = {
    'name': 'preset',
    'fit': False
}
base_layout_fit = {
    'name': 'preset',
    'fit': True
}