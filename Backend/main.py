from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os

# ==========================================
# Create FastAPI App
# ==========================================

app = FastAPI(
    title="AI Sentence Classifier API",
    description="Predict whether a sentence is Good or Bad using Machine Learning",
    version="1.0"
)

# ==========================================
# Enable CORS
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# Load Saved Model
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join("saved_model", "model.pkl")
vectorizer_path = os.path.join("saved_model", "vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# ==========================================
# Request Model
# ==========================================

class InputText(BaseModel):
    text: str

# ==========================================
# Home Route
# ==========================================

@app.get("/")
def home():
    return {
        "message": "AI Sentence Classification API is Running 🚀"
    }

# ==========================================
# Prediction Route
# ==========================================

@app.post("/predict")
def predict(data: InputText):

    sentence = data.text

    # Convert sentence into TF-IDF vector
    sentence_vector = vectorizer.transform([sentence])

    # Predict class
    prediction = model.predict(sentence_vector)[0]

    # Predict probabilities
    probability = model.predict_proba(sentence_vector)[0]

    # Confidence of predicted class
    confidence = probability[prediction] * 100

    if prediction == 0:
        result = "Good"
    else:
        result = "Bad"

    return {
        "sentence": sentence,
        "prediction": result,
        "confidence": round(confidence, 2)
    }