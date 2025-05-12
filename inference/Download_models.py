# %% [markdown]
# # Libabry

# %%
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import os

# %% [markdown]
MODEL_DIR = "./models/pretrained_model"
os.makedirs(MODEL_DIR, exist_ok=True)

# %% [markdown]
def download_pipeline(model_name, save_dir):
    print(f"Downloading {model_name}...")
    pipe = pipeline("sentiment-analysis", model=model_name)
    pipe.model.save_pretrained(save_dir)
    pipe.tokenizer.save_pretrained(save_dir)

def download_model(model_name, save_dir):
    print(f"Downloading {model_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

# %% [markdown]
download_pipeline("cardiffnlp/twitter-roberta-base-sentiment", f"{MODEL_DIR}/twitter-roberta-base-sentiment")
download_pipeline("unitary/toxic-bert", f"{MODEL_DIR}/toxic-bert")
download_model("nateraw/bert-base-uncased-emotion", f"{MODEL_DIR}/bert-base-uncased-emotion")
download_model("facebook/roberta-hate-speech-dynabench-r4-target", f"{MODEL_DIR}/roberta-hate-speech")
