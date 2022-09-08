# Narrative Maps 1.0.0
## Overview
This repository contains the following elements:

1. Narrative Maps Visualization Tool ("narratives" folder).
2. Search-based Baseline Tool ("search" folder).
3. Testing data sets ("data" folder).

The narrative maps algorithm and system is implemented based on the algorithm of [Keith and Mitra (2020)](https://dl.acm.org/doi/abs/10.1145/3432927) and follows the design guidelines of [Keith, Mitra, and North (2021)](https://journals.sagepub.com/doi/abs/10.1177/14738716221079593).

## Notes
- Narrative maps are visualized using [dash-cytoscape](https://github.com/plotly/dash-cytoscape).
- Optimization is done via linear programming using [PuLP](https://github.com/coin-or/pulp).
- The connection explanation and event comparison functions use [SHAP](https://github.com/slundberg/shap) to generate explanations. This approach was adapted from [this implementation of text similarity explainable metrics](https://github.com/yg211/explainable-metrics). 

## Requirements
Here is the list of requirements for the project.
```
dash==2.6.0
dash_bootstrap_components==1.0.0b3
dash_cytoscape==0.3.0
dash_daq==0.5.0
dash_extensions==0.1.5
Flask==1.1.2
ftfy==6.0.3
hdbscan==0.8.27
matplotlib==3.3.4
networkx==2.5
nltk==3.6.1
numpy==1.21.2
pandas==1.2.4
plotly==5.3.1
PuLP==2.5.0
python_dateutil==2.8.2
scikit_learn==1.1.2
scipy==1.6.2
sentence_transformers==2.2.2
shap==0.41.0
spacy==3.1.3
torch==1.12.1
transformers==4.21.0
truecase==0.0.14
umap==0.1.1
umap_learn==0.5.1
torch==1.10.2
```
