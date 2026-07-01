import joblib
import pandas as pd

pipeline = joblib.load("pipeline.pkl")

sample = pd.DataFrame(
    {
        "age": [35],
        "sex": ["male"],
        "bmi": [27.5],
        "children": [2],
        "smoker": ["no"],
        "region": ["southwest"],
    }
)

prediction = pipeline.predict(sample)

print("Predicted Insurance Cost:", prediction[0])