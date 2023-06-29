from flask import Flask, request, jsonify, render_template
import torch
from model import BiLSTM
# from dataset import TimeSeriesInferenceBatchDataset

model = BiLSTM(2, 10)
_ = model.eval()

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    # request_type_str = request.method
    # print(request_type_str)
    # return f"<p>Hello, World! {request_type_str} </p>"
    return render_template("index.html")

@app.post("/predict")
def predict():
    data = request.json

    try:
        X = data["input"]
        X = torch.Tensor(X).unsqueeze(0)
    except KeyError:
        return jsonify({"error": "No input sent"})

    with torch.no_grad():
        y_pred = model(X)

    try:
        result = jsonify({
            "X": X.tolist()[0],
            "y_pred": y_pred.tolist()[0]
        })
    except TypeError as e:
        result = jsonify({'error': str(e)})

    return result

if __name__ == "__main__":
    app.run(debug=True)