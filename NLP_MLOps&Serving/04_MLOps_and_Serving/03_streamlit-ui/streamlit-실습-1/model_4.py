import random

class MyModel:
    def __init__(self):
        print("Model init")

    def predict(self, input_text, keywords):
        if any(keyword in input_text for keyword in keywords):
            prediction_score = 1
            reason = "Spam Keyword Found"
        else:
            prediction_score = 0
            reason = "Nothing Found"
        return {
            "input": input_text,
            "prediction_score": prediction_score,
            "is_spam": prediction_score > 0,
            "reason": reason
        }