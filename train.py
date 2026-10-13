
import joblib
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

print("Loading dataset...")

train = fetch_20newsgroups(
    subset="train",
    remove=("headers", "footers", "quotes")
)

test = fetch_20newsgroups(
    subset="test",
    remove=("headers", "footers", "quotes")
)

print("Creating model...")

model = make_pipeline(
    TfidfVectorizer(),
    MultinomialNB()
)

print("Training model...")
model.fit(train.data, train.target)

print("Testing model...")
predictions = model.predict(test.data)

accuracy = accuracy_score(test.target, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")

joblib.dump(model, "20newsgroups_model.joblib")

print("Model saved successfully!")
