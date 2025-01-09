FROM tensorflow/serving:2.14.0

# Copy the model to the model directory
COPY clothing-model /models/clothing-model/1
ENV MODEL_NAME=clothing-model