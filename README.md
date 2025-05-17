# Narrative Maps 2.0.0
## Overview
This repository contains the following elements:

1. Narrative Maps Visualization Tool ("narratives" folder).
2. Testing data sets ("data" folder).
3. Custom data preparation ("preprocessing" folder, see below for details).
4. Tutorial and examples ("tutorial" folder), see the detailed tutorial [here](https://github.com/briankeithn/narrative-maps/blob/main/tutorial/TUTORIAL.md).

This is a major release that required updating several parts of the code to handle outdated libraries and dependency issues. Note that older versions of the repository contained some additional experiments and validations. For simplicity in the "release" version, I decided to simplify the structure. 

## Important: This code is provided AS IS under a MIT License.

## Scientific Publications associated with this work:
The following articles are based upon our work in this functional prototype:

- [Keith and Mitra (2020)](https://dl.acm.org/doi/abs/10.1145/3432927). This article provides further details on the Narrative Maps framework and the basic extraction algorithm.
- [Keith, Mitra, and North (2021)](https://journals.sagepub.com/doi/abs/10.1177/14738716221079593). This article provides information on the design guidelines that we followed in the implementation of our system.
- [Keith, Mitra, and North (2023)](https://dl.acm.org/doi/abs/10.1145/3581641.3584076). This article presents the semantic interaction model of our system and evaluations based on simulations and expert feedback.
- [Keith, German, Krokos, Joseph, and North (2025)](https://ceur-ws.org/Vol-3964/paper1.pdf). This article presents the XAI components of our work and an associated user study.

Further research is being developed based on this interactive prototype. However, this version is deemed to be appropriate for release for testing purposes. If you wish to purely test the algorithms, I would recommend the following work and associated repository:

- [German, Keith, and North](https://ceur-ws.org/Vol-3964/paper2.pdf). This work presents an alternative extraction algorithm and compares it with a (simplified) version of Narrative Maps. It also has an associated [GitHub Repository](https://ceur-ws.org/Vol-3964/paper2.pdf).

## Running locally
In the "narratives" directory, run the following command:
```
python NMVT.py
```

## Running with custom data
I added a specific option to run with Custom data in your local version. Simply create a "custom.csv" file, place it in the "data" folder, and select the "Custom" option in the data selection drop down in the main menu.

The CSV file must have the following structure (columns):
- id (numerical id, assumed integer, assumed 0-indexed in increasing order by date, so 0 is the first document in temporal order)
- title (title, used for display in the map)
- url (can be empty, but needs to exist as a column, if it exists a link will be added in the event details tab)
- date (date of the document, can be a regular date or a date time)
- publication (source of the document, in our case it's mostly news outlets either as a string or as a partial URL, but it can be anything in practice, if no specific logo is available for the source a blank marker is used in the map)
- full_text (the full text of the document)
- embedding (we recommend using an embedding with the all-MiniLM-L6-v2 model from SentenceTransformers) - note that we can easily swap the model to a different one, but it requires generating all the pre-computed embeddings again for all data sets.

If you have a data set with the first 6 columns (id, title, url, date, publication, full_text), you can generate the embeddings using the Jupyter notebook "Prepare_Embeddings.ipynb" available in the "preprocessing" folder. The Jupyter notebook has the following requirements:
```
ftfy==6.0.3
numpy==1.21.2
pandas==1.2.4
sentence_transformers==2.2.2
```

**Please note that the data set must be sorted in ascending order by time and the id column must also align with that, as the system makes assumptions on the temporal order of the documents based on their listing order in the data set.**

## Reproducibility
- The seed is fixed for reproducibility, but the results will likely be different in each system (although consistent within it). I'm not sure if this can be fixed so that the same results can be obtained in all systems.
- Sometimes the same result may look different due to the layout engine producing different results (probably due to random initliazation, not sure if that can be fixed, but it is not a major issue usually).

## Performance and computational cost
- The first execution of the narrative extraction method is probably going to be slow.
- The first time you run a specific data set requires extra computation as it needs to precompute some similarity tables and entity information that is then saved for later runs.
- After that, the performance should stabilize.

## Important libraries
- Narrative maps are visualized using [dash-cytoscape](https://github.com/plotly/dash-cytoscape).
- Optimization is done via linear programming using [PuLP](https://github.com/coin-or/pulp).
- The connection explanation and event comparison functions use [SHAP](https://github.com/slundberg/shap) to generate explanations. This approach was adapted from [this implementation of text similarity explainable metrics](https://github.com/yg211/explainable-metrics). 

## Requirements
Here is the list of requirements for the project. Please note that everything has been implemented using Python 3.8. You may also need to install some NLTK packages (`stopwords`, `punkt`, and `wordnet`). Also make sure to have GraphViz properly installed (needed to use NetworkX's `graphviz_layout` and `pygraphviz`).
```
dash==3.0.4
dash_bootstrap_components==2.0.2
dash_cytoscape==1.0.1
dash_daq==0.6.0
dash_extensions==2.0.4
Flask==3.0.3
ftfy==6.0.3
hdbscan==0.8.40
matplotlib==3.10.3
networkx==3.4.2
nltk==3.8.1
numpy==2.2.5
pandas==2.2.3
plotly==6.1.0
PuLP==2.5.0
python_dateutil==2.8.2
scikit_learn==1.6.0
scipy==1.15.3
sentence_transformers==4.1.0
shap==0.47.2
spacy==3.8.4
transformers==4.51.3
truecase==0.0.14
umap_learn==0.5.1
torch==2.7.0
pygraphviz==1.14
en-core-web-md @ https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.8.0/en_core_web_md-3.8.0-py3-none-any.whl
```

## Issues and Bugs
- Complex Maps: When the maps become too big or too complex to handle by the layout engine, there will be issues with the bounding boxes of the storyline (e.g., a big gray box that engulfs everything else). This seems to be an issue with the GraphViz/DOT engine that's running behind the scenes. It doesn't look like a limitation of Cytoscape. But I might be wrong.
- Cluster Coloring: In some rare cases (I haven't been able to reproduce this yet), nodes that have been assigned to a specific cluster (e.g. "blue") get their color removed in the next iteration, despite still belonging to that cluster and still being used by the semantic interaction code as such. This is only a display issue, but it might be annoying.

Please let me know if you find anything else. 
