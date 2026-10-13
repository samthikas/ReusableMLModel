
import joblib

print("Loading saved model...")

model = joblib.load("20newsgroups_model.joblib")

print("Model loaded successfully!")

texts = [
    "I enjoy playing football and watching sports.",
    "My computer has a graphics card problem."
]

predictions = model.predict(texts)

for text, prediction in zip(texts, predictions):
    category = model.classes_[prediction]
    print("\nText:", text)
    print("Predicted category ID:", category)
