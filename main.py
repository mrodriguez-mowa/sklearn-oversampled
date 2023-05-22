from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/api/trained-model', methods=['POST'])
def predict():
    vectorizer = joblib.load('vectorizer.joblib')
    model = joblib.load('model.joblib')

    body = request.get_json()
    response = []

    for element in body:
        message = element['textMessage']
        X = vectorizer.transform([message.lower()])
        y_pred = model.predict(X)
        element['qualification'] = y_pred[0]
        element['model'] = 'Sklearn'
        response.append(element)

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)