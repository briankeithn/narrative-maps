# Narrative Maps 1.1.1
## Overview
This repository contains the following elements:

1. Narrative Maps Visualization Tool ("narratives" folder).
2. Search-based Baseline Tool ("search" folder).
3. Testing data sets ("data" folder).
4. Custom data preparation ("preprocessing" folder, see below for details).
5. Tutorial and examples ("tutorial" folder), see the detailed tutorial [here](https://github.com/briankeithn/narrative-maps/blob/main/tutorial/TUTORIAL.md).
6. Simulation-based evaluations ("tests" folder), see the notebooks for details.

## Important: This code is provided as is and will not be maintained.

## Citing our work
Please cite the following publication: *Mixed Multi-Model Semantic Interaction for Graph-based Narrative Visualizations*, B. Keith, T. Mitra, C. North. In Proceedings of ACM IUI 2023 (DOI: TBA)

For further details on the narrative maps framework and the extraction algorithm, please refer to [Keith and Mitra (2020)](https://dl.acm.org/doi/abs/10.1145/3432927). For further design guidelines that we followed in the implementation of our system, please refer to [Keith, Mitra, and North (2021)](https://journals.sagepub.com/doi/abs/10.1177/14738716221079593). 

## Installation
Requires Python 3.12+ and Poetry.

1. Run `poetry install`. 
2. Run `poetry shell` to activate the virtual environment and then run `spacy download en_core_web_sm` to download the English language model for spaCy.
3. Install PyTorch (see [here](https://pytorch.org/get-started/locally/) for details).
4. Install PyGraphViz (see [here](https://pygraphviz.github.io/documentation/stable/install.html) for details).


## Running locally
In the "narratives" directory, run the following command:
```
python NMVT.py
```
In the "search" directory, run the following command:
```
python NMVT_Baseline.py
```

## Running with custom data
I added a specific option to run with Custom data in your local version (not available in the PythonAnywhere version!). Simply create a "custom.csv" file, place it in the "data" folder, and select the "Custom" option in the data selection drop down in the main menu.

The CSV file must have the following structure (columns):
- id (numerical id, assumed integer, assumed 0-indexed in increasing order by date, so 0 is the first document in temporal order)
- title (title, used for display in the map)
- url (can be empty, but needs to exist as a column, if it exists a link will be added in the event details tab)
- date (date of the document, can be a regular date or a date time)
- publication (source of the document, in our case it's mostly news outlets either as a string or as a partial URL, but it can be anything in practice, if no specific logo is available for the source a blank marker is used in the map)
- full_text (the full text of the document)
- embedding (an embedding using the all-MiniLM-L6-v2 model from SentenceTransformers) - note that we can easily swap the model to a different one, but it requires generating all the pre-computed embeddings again for all data sets.

If you have a data set with the first 6 columns (id, title, url, date, publication, full_text), you can generate the embeddings using the Jupyter notebook "Prepare_Embeddings.ipynb" available in the "preprocessing" folder. 

**Please note that the data set must be sorted in ascending order by time and the id column must also align with that, otherwise there might be some issues.**

## Reproducibility
- The seed is fixed for reproducibility, but the results will likely be different in each system (although consistent within it). I'm not sure if this can be fixed so that the same results can be obtained in all systems.
- Sometimes the same result may look different due to the layout engine producing different results (probably due to random initliazation, not sure if that can be fixed)

## Performance and computational cost
- The first execution of the narrative extraction method is probably going to be slow.
- The first time you run a specific data set requires extra computation as it needs to generates some similarity tables and entity information that is then saved for later runs.
- After that, the performance should stabilize.

## Important libraries
- Narrative maps are visualized using [dash-cytoscape](https://github.com/plotly/dash-cytoscape).
- Optimization is done via linear programming using [PuLP](https://github.com/coin-or/pulp).
- The connection explanation and event comparison functions use [SHAP](https://github.com/slundberg/shap) to generate explanations. This approach was adapted from [this implementation of text similarity explainable metrics](https://github.com/yg211/explainable-metrics).

## Issues and Bugs
- Complex Maps: When the maps become too big or too complex to handle by the layout engine, there will be issues with the bounding boxes of the storyline (e.g., a big gray box that engulfs everything else). This seems to be an issue with the GraphViz/DOT engine that's running behind the scenes. It doesn't look like a limitation of Cytoscape. But I might be wrong.
- Cluster Coloring: In some rare cases (I haven't been able to reproduce this yet), nodes that have been assigned to a specific cluster (e.g. "blue") get their color removed in the next iteration, despite still belonging to that cluster and still being used by the semantic interaction code as such. This is only a display issue, but it might be annoying.

Please let me know if you find anything else. 
