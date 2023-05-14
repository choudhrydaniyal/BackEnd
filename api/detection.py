
import joblib


def detecting_fake_news(var):
    load_model = joblib.load(open(
        'model.sav', 'rb'))
    prediction = load_model.predict([var])
    prob = load_model.predict_proba([var])
    pred = prediction[0]*100
    result = ""

    return (prob[0][1])
