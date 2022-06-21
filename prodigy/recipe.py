import prodigy
from prodigy.components.preprocess import add_tokens
import requests
import spacy
import json
from prodigy.components.loaders import JSONL
import pandas as pd

@prodigy.recipe(
    "annotation",
    dataset=("The dataset to save to", "positional", None, str),
    file_path=("Path to texts", "positional", None, str),
)

def block_ner(dataset,file_path, lang="en"):
    # We can use the blocks to override certain config and content, and set
    # "text": None for the choice interface so it doesn't also render the text
    blocks = [
        {"view_id": "ner_manual", "text": None},
        {"view_id": "choice", "text": None},
        {"view_id": "html", "html_template": "<p>Please access your confidence: <span id='conf'></span></p> <input type='range' id='slider' min='1' max='5' onChange='handleChange(this)' />", "javascript":"var slider = document.getElementById('slider'); function handleChange(slider) {window.prodigy.update({value: slider.value })}"}
    ]
    #stream = JSONL(source) # set up the stream
    #nlp = spacy.blank(lang)
    #stream = add_tokens(nlp, stream)  # tokenize the stream for ner_manual
    def get_data():
       # f = open(file_path)
        #res = json.load(f)
        #for fact in res:
         #   yield {"text": fact["text"]}
            
        res = pd.read_csv(file_path)
        for idx, fact in res.iterrows():
            yield {"text": fact["sentences"]}
            
    nlp = spacy.blank(lang)           # blank spaCy pipeline for tokenization
    stream = get_data()             # set up the stream
    stream = add_tokens(nlp, stream)  # tokenize the stream for ner_manual
    stream = add_options(stream)  # add options to each task
    
    return {
        "dataset": dataset,          # the dataset to save annotations to
        "view_id": "blocks",         # set the view_id to "blocks"
        "stream": stream,            # the stream of incoming examples
        "config": {
            "labels": ["Causal", "Effect", "Prev-C", "Enab-C"],  # the labels for the manual NER interface
            "batch_size": 20,
            "blocks": blocks         # add the blocks to the config
        }
    }

def add_options(stream):
    # Helper function to add options to every task in a stream
    options = [
        {"id": 1, "text": "Causal"},
        {"id": 0, "text": "Not Causal"}
    ]
    for task in stream:
        task["options"] = options
        yield task
    
