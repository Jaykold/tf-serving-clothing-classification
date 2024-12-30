import tensorflow as tf
import numpy as np
import grpc

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from keras_image_helper import create_preprocessor

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from proto import np_to_protobuf

app = FastAPI(debug=True)

class InputData(BaseModel):
    url: Optional[str] = None

host = 'localhost:8500'

channel = grpc.insecure_channel(host)

stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)


preprocessor = create_preprocessor('xception', target_size=(299,299))

classes = [
        'dress',
        'hat',
        'longsleeve',
        'outwear',
        'pants',
        'shirt',
        'shoes',
        'shorts',
        'skirt',
        't-shirt'
        ]

def shai_hulud(data: np.ndarray):
    grpc_request = predict_pb2.PredictRequest()
    grpc_request.model_spec.name = 'clothing-model'
    grpc_request.model_spec.signature_name = 'serving_default'

    grpc_request.inputs['input_2'].CopyFrom(np_to_protobuf(data))
    return grpc_request

def predict(url:str, timeout=20.0):
    X = preprocessor.from_url(url)
    grpc_request = shai_hulud(X)
    grpc_response = stub.Predict(grpc_request, timeout=timeout)
    preds = grpc_response.outputs['dense_1'].float_val
    
    return dict(zip(classes, preds))

@app.get('/predict')
@app.post('/predict')
def predict_endpoint(data: Optional[InputData] = None, url: Optional[str] = None):
    if data:
        url = data.url
    elif not url:
        return {"error": "URL parameter is required"}

    try:
        prediction = predict(url)
        return prediction
    except Exception as e:
        return {"error": str(e)}

if __name__ == '__main__':
    # url = "https://bit.ly/al-gaib"
    # response = predict(url)
    # print(response)
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8001)