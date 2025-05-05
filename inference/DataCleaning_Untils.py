# %% [markdown]
# # Libabry

# %%
import os
import pandas as pd
import re
import emoji

from bs4 import BeautifulSoup
from textblob import TextBlob

# %%
words_map = {
    "u": "you",
    "ur": "your",
    "pls": "please",
    "plz": "please",
    "idk": "I don't know",
    "im": "I am",
    "lol": "laugh out loud",
    "lmao": "laughing my ass off",
    "wtf": "what the heck",
    "fyi": "for your information",
    "diy": "do it yourself",
    "omg": "oh my god",
    "btw": "by the way",
    "tbh": "to be honest",
    "ded": "dead"
}

# Blob correction
def blob_correct(text):
    if not isinstance(text, str):
        return ""
    
    corrected_word = []

    for word in text.split():
        if len(word) <= 4:
            word = str(TextBlob(word).correct())
        corrected_word.append(word)
    return " ".join(corrected_word)
        
def clean_comment(text):
    if not isinstance(text, str):
        return ""
    
    #Remove html markup
    text = BeautifulSoup(text, "html.parser").get_text()
    # Lower case
    text = text.lower()

    # Normalize emoji
    text = emoji.demojize(text)

    # Remove URL of all kind
    text = re.sub(r"http:\S+|www\S+", "", text)

    # Remove mention and hashtag
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)

    # Handle acronym words first phase
    words = text.split()
    words = [words_map.get(word, word) for word in words]
    text = " ".join(words)

    #Correct small typo
    text = blob_correct(text)

    # Normalize repeated character. Ex: yeeeees -> yees
    text = re.sub(r"(.)\1{2,}", r"\1", text)

    # Remove non-alphanumeric except emoji alias
    text = re.sub(r"[^a-zA-Z0-9\s_:]", "", text)
    
    # Clean up extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    #textblob = TextBlob(text)
    #text = textblob.correct()

    return text
