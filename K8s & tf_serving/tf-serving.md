Using IPython on your terminal, run this commands

```
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('xception_v1_03_0.839.h5')

tf.saved_model.save(model, 'clothing-model')
```

We can inspect what's inside the saved model using the utility (saved_model_cli) from TensorFlow

```
saved_model_cli show --dir clothing-model --all
```

You can build the docker images using this code

```
docker build -t zoomcamp-10-model:xception-v1-03 -f image-model.dockerfile .

docker build -t zoomcamp-10-gateway -f image-gateway.dockerfile .
```

Docker compose

```
docker-compose up

docker ps: see the running container
```

Deploying to Kubernetes using Kind

- Use this [link](https://kind.sigs.k8s.io/docs/user/quick-start/) to download/install kind
- Use this [link](https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/) to install kubectl

Setup kubernetes clusters

- `kind create cluster` (default cluster name is kind)
- Configure kubectl to interact with kind: kubectl cluster-info --context kind-kind
- Check the running services to make sure it works: kubectl get service

Load the model image to kind
`kind load docker-image zoomcamp-10-model:xception-v1-03`

create model deployment & service
`kubectl apply -f model_deployment.yaml`
`kubectl apply -f model_service.yaml`

create gateway deployment & service
`kubectl apply -f gateway_deployment.yaml`
`kubectl apply -f gateway_service.yaml`

Test the gateway service
`kubectl port-forward service/gateway 8001:80` and run the test.py file to get predictions
