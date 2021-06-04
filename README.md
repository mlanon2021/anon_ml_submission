# Simple (not cleanest) way to install:
1. install graph_tool globally for python3.x 
    - e.g. via adding https://downloads.skewed.de/apt to sources and apt-get install python3-graph-tool (as described in https://git.skewed.de/count0/graph-tool/-/wikis/installation-instructions)
2. setup virtualenv with pip and requirements.txt 
    - make the global site-packages available to virtualenv by either "--system-site-packages"
    
# To reproduce the main experiments of the paper
python3 -m querying.cut_finding_experiments

# To reproduce the convexity tests on community detection networks
1. Download data from snap http://snap.stanford.edu/data/index.html "Networks with ground-truth communities".
2. Put it into dataset/communities (dblp is already contained in there, as an example dataset)
3. python3 -m querying.cut_finding_experiments

