from fastapi import FastAPI
from transformers import pipeline

app = FastAPI(title="API FastAPI + HuggingFace Demo", description="Practica IA MLOps")

# Cargar pipelines HF solo una vez al arrancar el servidor
sentiment_analysis = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

@app.get("/")
def root():
    return {"mensaje": "¡API FastAPI funcionando!"}

@app.get("/hello")
def hello(name: str = "Beatriz"):
    return {"saludo": f"¡Hola {name}!"}

@app.get("/sentiment")
def sentiment(text: str):
    res = sentiment_analysis(text)
    return {"sentimiento": res}

@app.get("/zero-shot")
def zero_shot(text: str, candidate_labels: str):
    # candidate_labels = "deporte,tecnologia,comida"
    etiquetas = [e.strip() for e in candidate_labels.split(",")]
    res = zero_shot_classifier(text, etiquetas)
    return {"resultado": res}

@app.get("/length")
def length(text: str):
    return {"longitud_texto": len(text)}

@app.get("/reverse")
def reverse(text: str):
    return {"texto_invertido": text[::-1]}
