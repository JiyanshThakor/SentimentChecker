import joblib as jb

model = jb.load("sentiment_model.pkl")

text_to_predict = ["yay!!!!1"]

pred = model.predict(text_to_predict)

mapping = {0: "negative", 1: "neutral", 2: "positive"}
sentiment = mapping.get(pred[0], "unknown")

print(f"Label: {pred[0]}")
print(f"Sentiment: {sentiment}")