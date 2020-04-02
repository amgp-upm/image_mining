### In requires the following packages
### conda install -c conda-forge flask-restful
### conda install pillow

from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from PIL import Image
import numpy as np
import io
import os

# if you have used sklearn for generating the model
from sklearn.externals import joblib

app = Flask(__name__) # Server
api = Api(app) # api-rest

#  Loading the pre-trained model
model = joblib.load('svm_model_mnist_pixels.joblib') 

def feature_extraction(image):
    # preprocessing
    im = image.astype(np.float32) / 255.
    # TODO: real feature extraction
    im = np.reshape(im, (1, np.prod(im.shape))) # we need a vector [1 x num_features] (depends on classifier)
    features = im
    return features

# We create a resource for the api
class Prediction(Resource):
    @staticmethod
    def post():
        data = {"success": False}
        # Check if an image was posted
        if request.files.get('image'):
            im = request.files["image"].read()
            im = Image.open(io.BytesIO(im))
            im = np.array(im)

            features = feature_extraction(im)
            
            y_hat = model.predict(features)

            labels = {0:'zero', 1:'one', 2:'two', 3:'three', 4:'four', 
                      5:'five', 6:'six', 7:'seven', 8:'eight', 9:'nine'}
            
            data['prediction'] = labels[y_hat[0]]
            data['success'] = True # Indicate that the request was a sucess

            return jsonify(data) # Response

api.add_resource(Prediction, '/predict')

if __name__ == "__main__":
    print('Loading model and Flask starting server...')
    print('please wait until server has fully started')

    app.run(debug=True, host='0.0.0.0', port=5000) # Debug mode and open to all connections in port 5000
