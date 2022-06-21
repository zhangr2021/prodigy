# https://support.prodi.gy/t/saving-user-ratings-to-a-task-from-html-inputs/4468
# https://support.prodi.gy/t/is-there-a-slider-interface/3987
import prodigy
from prodigy.components.preprocess import add_tokens
import requests
import spacy
import json
from prodigy.components.loaders import JSONL
import pandas as pd

with open('monitor.js') as txt:
    monitor = txt.read()
    
custom_css = """
/* style settings for custom slider */
.slidecontainer {
    margin: auto;
    width: 100%; /* Width of the outside container */
}

.slider {
    -webkit-appearance: none; /* Override default look */
    width: 100%; /* Set a specific slider handle width */
    height: 25px; /* Slider handle height */
    background: #d3d3d3; /* slider background */
    outline: none; /* Remove outline */
    opacity: 0.7; /* Set transparency (for mouse-over effects on hover) */
    -webkit-transition: .2s; /* 0.2 seconds transition on hover */
    transition: opacity .2s;
    cursor: pointer; /* Cursor on hover */
}

.slider:hover {
    opacity: 1; /* Fully shown on mouse-over */
}
"""

slider_html = """
<p>Please access your confidence: <span id='conf'></span></p>
<div class="slidecontainer">
    <input type="range" list="tickmarks" min="1"
    max="5" class="slider" id="score_slider" step=1>
    
    <datalist id="tickmarks">
        <option value="1" label="0%"></option>
        <option value="2"></option>
        <option value="3" label="50%"></option>
        <option value="4"></option>
        <option value="5" label="100%"></option>
    </datalist>
</div>
"""

custom_javascript1 = """
if (!selected.length && 'choice' in window.prodigy.viewId){
    prodigy.addEventListener('prodigyanswer', event => {
        const selected = event.detail.task.accept || []
        if (!selected.length) {
            alert('Task with no selected options submitted.')
        }
    })
}
"""

custom_javascript2 = """
document.addEventListener('prodigyanswer', event => {
    const { task, answer } = event.detail
    // Perform your checks here and show an alert
    if(answer != 'ignore' && task.accept.length == 0) {
        alert ('Select at least one option. Please undo and correct your annotation.')
    }
})
"""

@prodigy.recipe(
    "annotation",
    dataset=("The dataset to save to", "positional", None, str),
    file_path=("Path to texts", "positional", None, str),
)

def block_ner(dataset, file_path: str, lang="en"):
    """
    Rating pairwise model outputs with a preference slider
    """
    # define blocks for multi tasks
    blocks = [
        {"view_id": "ner_manual", "text": None},
        {"view_id": "choice", "text": None},
        {"view_id": "text_input", "field_id": "confidence", "field_placeholder": "Your Confidence", "field_suggestions": [1, 2, 3, 4, 5]
}
    ]
    # stream in lines from JSONL file yielding a
    # dictionary for each example in the data.
    def get_data():
        res = pd.read_csv(file_path)
        for idx, fact in res.iterrows():
            yield {"text": fact["sentences"]}

    nlp = spacy.blank(lang)           # blank spaCy pipeline for tokenization
    stream = get_data()             # set up the stream
    stream = add_tokens(nlp, stream)  # tokenize the stream for ner_manual
    stream = add_options(stream)
    
    return {
        "view_id": "blocks",
        "dataset": dataset,  # Name of dataset to save annotations
        "stream": stream,  # Incoming stream of examples
        "config": {
            "blocks": blocks,
            "labels": ["Causal", "Effect", "Prev-C", "Enab-C"],  # the labels for the manual NER interface
            "batch_size": 20,
            #"global_css": custom_css,
            "javascript": monitor, 
            },
        }

def add_options(stream):
    # Helper function to add options to every task in a stream
    options = [
        {"id": 1, "text": "Causal"},
        {"id": 0, "text": "Not Causal"}
    ]
    for task in stream:
        task["options"] = options
        #task["score"] = 0
        yield task
    
