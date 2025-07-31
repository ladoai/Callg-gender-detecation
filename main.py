from flask import Flask, request, jsonify
import os
import requests
from gender_predictor import predict_gender

app = Flask(__name__)

@app.route('/predict_gender', methods=['POST'])
def predict():
    data = request.get_json()
    audio_url = data.get('audio_url')

    if not audio_url:
        return jsonify({'error': 'Audio URL missing'}), 400

    try:
        filename = "temp_audio.wav"
        r = requests.get(audio_url)
        with open(filename, 'wb') as f:
            f.write(r.content)

        gender = predict_gender(filename)
        os.remove(filename)

        return jsonify({'gender': gender})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)
