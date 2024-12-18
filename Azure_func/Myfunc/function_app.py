import json
import logging
import azure.functions as func
from tflite_runtime.interpreter import Interpreter
from keras_image_helper import create_preprocessor

app = func.FunctionApp()

interpreter = Interpreter(model_path='model/clothing-model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

preprocessor = create_preprocessor('xception', target_size=(299, 299))

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

def predict(url):
    X = preprocessor.from_url(url)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    return dict(zip(classes, preds[0].tolist()))

@app.route(route="clothing_model", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST", "GET"])
def clothing_model(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    try:
        url = req.params.get('url')
        if not url:
            req_body = req.get_json()
            url = req_body['url']
        logging.info("Received URL: %s", url)

    except KeyError:
        return func.HttpResponse(
            "Invalid request. JSON body with 'url' field required.",
            status_code=400
        )
    
    # Get predictions
    predictions = predict(url)

    return func.HttpResponse(
        json.dumps(predictions),
        mimetype="application/json",
        status_code=200
    )