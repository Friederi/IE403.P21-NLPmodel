# %% [markdown]
# # Libabry

# %%
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

    text = BeautifulSoup(text, "html.parser").get_text()
    text = text.lower()

    text = emoji.demojize(text)

    text = re.sub(r"http:\S+|www\S+", "", text)

    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)

    words = text.split()
    words = [words_map.get(word, word) for word in words]
    text = " ".join(words)

    text = blob_correct(text)

    text = re.sub(r"(.)\1{2,}", r"\1", text)

    text = re.sub(r"[^a-zA-Z0-9\s_:]", "", text)
    
    text = re.sub(r"\s+", " ", text).strip()

    return text
