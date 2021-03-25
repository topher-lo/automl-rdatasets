# automl-rdatasets
[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/topher-lo/automl-rdatasets/main/app.py)

🔎🧙 Contextual search and autoML on R datasets using spacy and tpot. Served as a Streamlit web app.
Find inspiration (and starter code) for your next ML project.
Search for a dataset in your domain of interest from R's extensive 
[collection of data](https://vincentarelbundock.github.io/Rdatasets/index.html).
Run AutoML to find and generate Python code for an optimised ML pipeline!

![automl-rdatasets demo](https://github.com/topher-lo/automl-rdatasets/blob/main/media/recording.gif)

## 🎬 Live Demo
You can find the live demo [here](https://share.streamlit.io/topher-lo/automl-rdatasets/main/app.py).

### Using the app:
1. Find relevant R datasets using the search bar
2. Select a R dataset
3. Select an outcome variable and supervised ML task (regression or classification) to perform 
4. Press the "Run AutoML" button to perform AutoML and generate Python code for the best ML pipeline

Note: The running time for pipeline optimization is limited to to 5 minutes max. In practice, AutoML with TPOT should be run with multiple instances in parallel
for much longer (hours or days). You can modify this limit by modifying the "Maximum running time" Streamlit slider in `app.py`.

## Install
automl-rdatasets has been tested on Python 3.8 and depends on the following packages:
- `sklearn`
- `spacy`
- `streamlit`
- `streamlit-pandas-profiling`
- `missingno`
- `numpy`
- `numba`
- `pandas`
- `tpot`

To use run the demo locally, you must first clone this repo:
```bash
git clone git@github.com:topher-lo/automl-rdatasets.git
cd [..path/to/repo]
```
Then install its dependencies using either pip:
```bash
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```
Or run it as a containerised application using docker and docker-compose.
First create a file `.env` with the line:
```
PORT=8501
```
to specify which port to publish the data app to.
Then run the following commands on your CLI:
```bash
docker-compose build
docker-compose up -d
```

## 🚀 How to run this demo
In your virtual environment, run the following command from the `automl-rdatasets` dir:
```bash
streamlit run app.py
```
The web app will be available at http://localhost:8501/
Otherwise, if you are using docker, the web app will be available at whichever port you specified in the `.env` file.

## 🦾 Tech stack
- Uses `spaCy`'s pre-trained Word2Vec word embeddings and cosine similarity to perform contextual search
- Generates the data profiling report using `pandas-profiling`
- Generates missing value plots using `missingno`
- Performs AutoML using `TPOT`
- UI inspired by [Traingenerator](https://github.com/jrieke/traingenerator)

## Getting in touch
If you are having a problem with this demo, please raise a GitHub issue. For anything else, you can reach me at: lochristopherhy@gmail.com
