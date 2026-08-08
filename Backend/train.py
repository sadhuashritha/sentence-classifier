import os
import joblib
import pandas as pd

from scipy.sparse import hstack

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "Backend", "saved_model")

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# DATASETS
# ============================================================

files = [
    "amazon_cells_labelled.txt",
    "imdb_labelled.txt",
    "yelp_labelled.txt"
]


# ============================================================
# LOAD DATASETS
# ============================================================

print("\n==========================================")
print("LOADING DATASETS")
print("==========================================\n")

dataframes = []

for filename in files:

    file_path = os.path.join(DATASET_DIR, filename)

    if not os.path.exists(file_path):
        print(f"Dataset not found: {filename}")
        continue

    temp_df = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=["text", "label"]
    )

    dataframes.append(temp_df)

    print(f"Loaded {filename}: {len(temp_df)} samples")


if not dataframes:
    raise FileNotFoundError(
        "No dataset files found inside the dataset folder."
    )


# ============================================================
# COMBINE DATASETS
# ============================================================

df = pd.concat(dataframes, ignore_index=True)

print("\nTotal samples:", len(df))


# ============================================================
# CLEAN DATA
# ============================================================

df = df.dropna(subset=["text", "label"])

df["text"] = df["text"].astype(str).str.strip()

df = df[df["text"] != ""]

df["label"] = pd.to_numeric(
    df["label"],
    errors="coerce"
)

df = df.dropna(subset=["label"])

df["label"] = df["label"].astype(int)

df = df[df["label"].isin([0, 1])]

# Remove duplicate sentences
df = df.drop_duplicates(subset=["text"])

df = df.reset_index(drop=True)

print("Samples after cleaning:", len(df))


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n==========================================")
print("CLASS DISTRIBUTION")
print("==========================================")

print(df["label"].value_counts().sort_index())

print("\n0 = Bad / Negative")
print("1 = Good / Positive")


# ============================================================
# FEATURES AND LABELS
# ============================================================

X = df["text"]
y = df["label"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n==========================================")
print("TRAIN / TEST SPLIT")
print("==========================================")

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# WORD-LEVEL TF-IDF
# ============================================================

print("\n==========================================")
print("WORD-LEVEL TF-IDF")
print("==========================================")

word_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=20000,
    sublinear_tf=True,
    min_df=1
)

X_train_word = word_vectorizer.fit_transform(X_train)

X_test_word = word_vectorizer.transform(X_test)

print(
    "Word features:",
    len(word_vectorizer.get_feature_names_out())
)


# ============================================================
# CHARACTER-LEVEL TF-IDF
# ============================================================

print("\n==========================================")
print("CHARACTER-LEVEL TF-IDF")
print("==========================================")

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    max_features=30000,
    sublinear_tf=True,
    min_df=2
)

X_train_char = char_vectorizer.fit_transform(X_train)

X_test_char = char_vectorizer.transform(X_test)

print(
    "Character features:",
    len(char_vectorizer.get_feature_names_out())
)


# ============================================================
# COMBINE WORD + CHARACTER FEATURES
# ============================================================

print("\n==========================================")
print("COMBINING FEATURES")
print("==========================================")

X_train_combined = hstack([
    X_train_word,
    X_train_char
])

X_test_combined = hstack([
    X_test_word,
    X_test_char
])

print(
    "Combined training shape:",
    X_train_combined.shape
)

print(
    "Combined testing shape:",
    X_test_combined.shape
)


# ============================================================
# TRAIN LOGISTIC REGRESSION
# ============================================================

print("\n==========================================")
print("TRAINING MODEL")
print("==========================================")

model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    C=1.0,
    random_state=42
)

model.fit(
    X_train_combined,
    y_train
)

print("Model trained successfully.")


# ============================================================
# PREDICTIONS
# ============================================================

predictions = model.predict(X_test_combined)


# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n==========================================")
print("MODEL EVALUATION")
print("==========================================")

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Bad / Negative",
            "Good / Positive"
        ],
        digits=4
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    predictions
)

print("\nConfusion Matrix:\n")

print(cm)


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print("\n==========================================")
print("SAMPLE PREDICTIONS")
print("==========================================")

sample_sentences = [
    "I don't hate you",
    "Have a wonderful day",
    "bad food items",
    "You are an amazing person",
    "I hate this product",
    "This product is terrible",
    "I really enjoyed this",
    "The service was horrible",
    "I love this",
    "This is the worst experience",
    "I don't like this",
    "I really don't like this product",
    "This was an excellent experience",
    "The product is absolutely amazing",
    "The service was disappointing"
]


sample_word = word_vectorizer.transform(
    sample_sentences
)

sample_char = char_vectorizer.transform(
    sample_sentences
)

sample_combined = hstack([
    sample_word,
    sample_char
])

sample_predictions = model.predict(
    sample_combined
)

sample_probabilities = model.predict_proba(
    sample_combined
)


for sentence, prediction, probability in zip(
    sample_sentences,
    sample_predictions,
    sample_probabilities
):

    confidence = max(probability) * 100

    if prediction == 1:
        result = "Good"
    else:
        result = "Bad"

    print("\n" + sentence)
    print(f"Prediction : {result}")
    print(f"Confidence : {confidence:.2f}%")


# ============================================================
# SAVE MODEL
# ============================================================

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


joblib.dump(
    model,
    model_path
)

joblib.dump(
    word_vectorizer,
    word_vectorizer_path
)

joblib.dump(
    char_vectorizer,
    char_vectorizer_path
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n==========================================")
print("MODEL SAVED SUCCESSFULLY")
print("==========================================")

print(f"\nModel            : {model_path}")
print(f"Word Vectorizer  : {word_vectorizer_path}")
print(f"Char Vectorizer  : {char_vectorizer_path}")

print("\n==========================================")
print("TRAINING COMPLETE")
print("==========================================")