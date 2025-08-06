FROM condaforge/mambaforge:latest
WORKDIR /app

# Layer 1: System dependencies and build tools (rarely change)
RUN mamba install -y python=3.12.11 \
    graphviz \
    c-compiler \
    cxx-compiler \
    pkg-config \
    pip \
    setuptools && \
    pip install --upgrade "pip>=24.0"

# Layer 2: Core scientific Python stack (changes infrequently)
RUN mamba install -y \
    numpy=2.2.5 \
    scikit-learn=1.6.0 \
    matplotlib=3.10.3 \
    pandas=2.2.3 \
    networkx=3.4.2 && \
    mamba install -y -c defaults scipy=1.15.3

# Layer 3: Data visualization and graph tools
RUN mamba install -y \
    pygraphviz=1.14 \
    python-graphviz

# Layer 4: NLTK and other NLP basics (before the heavy ML packages)
RUN mamba install -y nltk=3.8.1 && \
    python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"

# Layer 5: Other analysis tools
RUN mamba install -y \
    umap-learn=0.5.8 \
    plotly=6.1.0 \
    dash=3.0.4

# Layer 6: Dash ecosystem (might update together)
RUN pip install --no-cache-dir \
    dash_bootstrap_components==2.0.2 \
    dash_cytoscape==1.0.1 \
    dash_daq==0.6.0 \
    dash_extensions==2.0.4

# Layer 7: Flask and web-related packages
RUN mamba install -y flask==3.0.3 python-dateutil==2.8.2 -c conda-forge || \
    pip install Flask==3.0.3 python_dateutil==2.8.2

# Layer 8: Clustering and other algorithms
RUN pip install hdbscan==0.8.39
RUN mamba install -y shap==0.47.2 -c conda-forge || pip install shap==0.47.2

# Layer 9: Text processing utilities
RUN pip install \
    ftfy==6.0.3 \
    truecase==0.0.14

# Layer 10: PyTorch (large download, separate layer)
# Try mamba first for faster download, fallback to pip for exact version
RUN mamba install -y pytorch==2.7.0 -c pytorch -c conda-forge || \
    pip install torch==2.7.0

# Layer 11: Transformers and sentence-transformers (depend on torch)
# Try mamba first for faster downloads
RUN mamba install -y transformers==4.51.3 -c huggingface -c conda-forge || \
    pip install transformers==4.51.3
RUN mamba install -y sentence-transformers==4.1.0 -c conda-forge || \
    pip install sentence_transformers==4.1.0

# Layer 12: SpaCy and its model (separate due to size)
RUN mamba install -y spacy==3.8.4 -c conda-forge || pip install spacy==3.8.4
RUN pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.8.0/en_core_web_md-3.8.0-py3-none-any.whl

# Layer 13: CBC solver and PuLP (moved here so you can change it)
RUN mamba install -y coincbc=2.10.12 pulp=2.8.0 -c conda-forge && \
    chmod +x /opt/conda/bin/cbc && \
    # Copy cbc to where PuLP expects it
    cp /opt/conda/bin/cbc /usr/bin/cbc && \
    chmod +x /usr/bin/cbc

# Clear any potentially corrupted bytecode (this is important to avoid issues)
RUN find /opt/conda -name "*.pyc" -delete 2>/dev/null || true

# Layer 14: Copy application code (changes most frequently)
COPY . .

ENV PATH="/opt/conda/bin:${PATH}"
EXPOSE 8050
CMD ["python", "NMVT.py"]