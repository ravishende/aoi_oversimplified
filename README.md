An oversimplified recreation of AoI
- Real version is hosted at [http://aoi.nrp-nautilus.io/](http://aoi.nrp-nautilus.io/)

There are several differences between this demo and AoI, but the main steps are similar and outlined in graph.py. 
This demo relies on LLM calls slightly more frequently than AoI because the amount of code for some of those deterministic steps is too large and would bloat the limited code that is meant for understanding the process.

Because this is an oversimplified demo, certain behaviors are different.
For example:
 1. The analyze_dataset step only works for the "Iris" dataset or image classification datasets
    - it returns "tabular_classification" for Iris and "image_classification" for everything else.
 2. There are no baseline models injected, so any recommendations are only based on the papers search, rather than including common classical models (e.g. RandomForest)

# Setup

1. Clone the repo
`git clone https://github.com/ravishende/aoi_oversimplified.git`

2. create a .env file 
`cp .env.example .env`

3. Open the .env file and fill in NRP_LLM_API_KEY with your API key.

4. Install required packages (instructions below) 

## MacOS
```
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

## Windows
```
py -m venv venv
venv\Scripts\Activate.ps1
pip install -e .
```

# Running
`chainlit run app.py`

Example prompt:
"Recommend a model for tabular classification on the Iris dataset at ./data/iris"