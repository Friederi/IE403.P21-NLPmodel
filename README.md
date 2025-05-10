This repostiory is for IE403.P21-NLPmodel project, which is a FastAPI-based REST API that annotates user comments using multiple pre-trained NLP models (sentiment, emotion, toxicity, hate score) and a custom fine-tuned BERT classifier. 
It is containerized via Docker for easy deployment and sharing.

For issues or contributions, open a GitHub issue or PR!
Contract Email: Khoa11042003@gmail.com
---

## Features

- Pretrained Models:
  - `cardiffnlp/twitter-roberta-base-sentiment`
  - `unitary/toxic-bert`
  - `nateraw/bert-base-uncased-emotion`
  - `facebook/roberta-hate-speech-dynabench-r4-target`
- Fine-tuned BERT model with scalar features
- FastAPI serving and Dockerized setup
- Volume mounting support to reduce image size

---

## Requirements

- Git large file storage (for downloading fine-tuned model in repository)
- Docker installed
- Python (for downloading pretrained models, optional if using pre-mounted volumes)

---

## Follow below instructions:

### 1. Clone the Repository

```
git lfs install
git clone https://github.com/Friederi/IE403.P21-NLPmodel.git
cd IE403.P21-NLPmodel
```

Note: You are not required to run requirements-dev.txt nor inference/requirements.txt. However, if you want to test in local development environment, run `pip install -r requirements-dev.txt`.

### 2. Download pretrained models (Only once)

```
cd ./inference
python Download_Models.py
```

### 3. Pull Docker Image from Docker Hub (or build locally, your choice!)

```
docker push freiderich/bert-fastapi-image:1.0.0
```
or build image yourself:
```
docker build -t  bert-fastapi-image .
```

###  4. Run the Docker Container

Make sure your local model/ directory contains:
 + best_model.pt (fine-tuned model)
 + pretrained_model/ (contains 4 Hugging Face models)

Then run the container with a volume mount:
```
docker run -p 8000:8000 -v "$(pwd)/model:/app/model" --name bert-fastapi-container bert-fastapi-image
```

### 4.1. (Optionally) Run locally

Once you download all packages in requirements-dev.txt, in inference folder run:
```
uvicorn app:app --reload --port 8000
```

### 5. Test the API with curl

```
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "you suck and should quit!"}'
```
