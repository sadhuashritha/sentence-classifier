from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os

from scipy.sparse import hstack


# ============================================================
# CREATE FASTAPI APP
# ============================================================

app = FastAPI(
    title="AI Sentence Classifier API",
    description="Predict whether a sentence is Good or Bad using Machine Learning",
    version="1.0"
)


# ============================================================
# ENABLE CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MODEL PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(
    BASE_DIR,
    "saved_model"
)

model_path = os.path.join(
    MODEL_DIR,
    "model.pkl"
)

word_vectorizer_path = os.path.join(
    MODEL_DIR,
    "word_vectorizer.pkl"
)

char_vectorizer_path = os.path.join(
    MODEL_DIR,
    "char_vectorizer.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model from:", model_path)
print(
    "Loading word vectorizer from:",
    word_vectorizer_path
)
print(
    "Loading character vectorizer from:",
    char_vectorizer_path
)


model = joblib.load(model_path)

word_vectorizer = joblib.load(
    word_vectorizer_path
)

char_vectorizer = joblib.load(
    char_vectorizer_path
)


print("Model loaded successfully!")


# ============================================================
# REQUEST MODEL
# ============================================================

class InputText(BaseModel):
    text: str


# ============================================================
# HOME ROUTE
# ============================================================

@app.get("/")
def home():

    return {
        "message": "AI Sentence Classification API is Running 🚀"
    }


# ============================================================
# PREDICTION ROUTE
# ============================================================

@app.post("/predict")
def predict(data: InputText):

    sentence = data.text

    # --------------------------------------------------------
    # Word-level TF-IDF
    # --------------------------------------------------------

    word_features = word_vectorizer.transform(
        [sentence]
    )

    # --------------------------------------------------------
    # Character-level TF-IDF
    # --------------------------------------------------------

    char_features = char_vectorizer.transform(
        [sentence]
    )

    # --------------------------------------------------------
    # Combine Word + Character Features
    # --------------------------------------------------------

    combined_features = hstack([
        word_features,
        char_features
    ])

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(
        combined_features
    )[0]

    # --------------------------------------------------------
    # Prediction Probability
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        combined_features
    )[0]

    confidence = (
        probabilities[int(prediction)] * 100
    )

    # --------------------------------------------------------
    # Convert Label
    #
    # 0 = Bad / Negative
    # 1 = Good / Positive
    # --------------------------------------------------------

    if prediction == 0:
        result = "Bad"
    else:
        result = "Good"

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {
        "sentence": sentence,
        "prediction": result,
        "confidence": round(
            confidence,
            2
        )
    }