import json
import random
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Load intents
with open("intents.json") as file:
    data = json.load(file)

sentences = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        sentences.append(pattern)
        labels.append(intent["tag"])

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)

model = LogisticRegression()
model.fit(X, labels)

print("Smart Chatbot 🤖 (type 'exit' to stop)")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Bot: Goodbye!")
        break

    input_vector = vectorizer.transform([user_input])
    prediction = model.predict(input_vector)[0]

    for intent in data["intents"]:
        if intent["tag"] == prediction:
            print("Bot:", random.choice(intent["responses"]))