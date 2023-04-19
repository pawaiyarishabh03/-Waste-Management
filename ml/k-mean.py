import pandas as pd
from flask_cors import CORS, cross_origin
import pickle
from flask import Flask, jsonify, request
import numpy as np
import sklearn
import random

app = Flask(__name__)
cors = CORS(app)

model = pickle.load(open('days_between_requests_model.pkl', 'rb'))


@app.route('/predict', methods=['POST'])
def predict():
  latitude = request.json["latitude"]
  longitude = request.json["longitude"]
  print(longitude)
  print(latitude)
  filename = 'days_between_requests_model.pkl'
  kmeans = pickle.load(open(filename, 'rb'))
  cluster_averages = pickle.load(open('cluster_averages.pkl', 'rb'))

  print(cluster_averages)

  # Define a new location's latitude and longitude
  new_location = np.array([[latitude, longitude]])

  # Compute the cluster label for the new location
  cluster_label = kmeans.predict(new_location)[0]
  print(cluster_label)

  # Look up the average number of days between requests for the cluster
  prediction = cluster_averages[cluster_label]

  return jsonify({
    'prediction': round(prediction),
  })


if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0', port=5000)
