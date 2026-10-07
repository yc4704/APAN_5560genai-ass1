# APAN 5560 — Applied Generative AI

This repository contains the cumulative coursework for APAN 5560.
Assignment 2 extends the FastAPI application developed in Assignment 1.

## Assignment Overview

| Assignment | Implementation | API Endpoints |
| --- | --- | --- |
| Assignment 1 | Bigram text generation and spaCy word embeddings | POST /generate, POST /embedding |
| Assignment 2 | CNN trained on CIFAR-10, with saved weights and Docker deployment | POST /predict-image |

For Assignment 2, see the "Assignment 2: CIFAR-10 Image Classification"
section below for the architecture, training results, and Docker commands.

## Assignment 1

A FastAPI application demonstrating bigram text generation and spaCy word embeddings.



\## Features



\- Generate text using a simple bigram language model.

\- Calculate a 300-dimensional embedding for an input word.

\- Validate request data with Pydantic.

\- Provide interactive API documentation with Swagger UI.

\- Run locally or inside a Docker container.



\## Project Structure



```text

APAN\_5560genai/

├── app/

│   ├── \_\_init\_\_.py

│   ├── bigram\_model.py

│   ├── embedding\_model.py

│   └── main.py

├── .dockerignore

├── .gitignore

├── .python-version

├── Dockerfile

├── pyproject.toml

├── README.md

└── uv.lock

```



\## Run Locally



Install the locked dependencies:



```bash

uv sync

```



Start the development server:



```bash

uv run fastapi dev app/main.py

```



Open the interactive documentation:



```text

http://127.0.0.1:8000/docs

```



\## API Endpoints



\- `GET /` returns basic API information.

\- `GET /health` checks whether the API is running.

\- `POST /generate` generates text with the bigram model.

\- `POST /embedding` returns a 300-dimensional spaCy word embedding.



\### Text Generation Request



```json

{

&#x20; "start\_word": "the",

&#x20; "length": 10

}

```



\### Word Embedding Request



```json

{

&#x20; "word": "apple"

}

```



\## Run with Docker



Build the Docker image:



```bash

docker build -t apan\_5560genai .

```



Run the container:



```bash

docker run -p 8000:80 apan\_5560genai

```



Then open:



```text

http://127.0.0.1:8000/docs

```



Press `Ctrl+C` in the terminal to stop the container.

## Assignment 2: CIFAR-10 Image Classification

This assignment extends the existing FastAPI application with a
CNN image classification endpoint. The text generation and word
embedding endpoints from Assignment 1 are retained.

### CNN Architecture

- Input: RGB image resized to 64 × 64
- Convolution: 3 input channels, 16 output channels, 3 × 3 kernel,
  stride 1, padding 1
- ReLU
- Max pooling: 2 × 2 kernel, stride 2
- Convolution: 16 input channels, 32 output channels, 3 × 3 kernel,
  stride 1, padding 1
- ReLU
- Max pooling: 2 × 2 kernel, stride 2
- Flatten: 32 × 16 × 16 = 8192 features
- Fully connected: 8192 → 100
- ReLU
- Fully connected: 100 → 10 class scores

### Training

The model was trained on the CIFAR-10 training set for 5 epochs,
using a batch size of 64, cross-entropy loss, and the Adam optimizer
with a learning rate of 0.001.

Preprocessing consists of resizing images to 64 × 64 and converting
them to tensors.

- Final training accuracy: 68.37%
- Test accuracy: 63.61%
- Saved weights: `models/cifar10_cnn.pth`

The trained weights are included, so retraining is not required
to run the API. Inference supports CPU execution.

### Project Files

- `helper_lib/model.py`: CNN architecture
- `helper_lib/data_loader.py`: CIFAR-10 loading and preprocessing
- `helper_lib/trainer.py`: training loop
- `helper_lib/evaluator.py`: test-set evaluation
- `train_cnn.py`: training and weight-saving script
- `app/cnn_service.py`: weight loading and image inference
- `app/main.py`: FastAPI endpoints

### Run with Docker

From the repository root, build the image:

```bash
docker build -t apan_5560genai:assignment2 .
```

Start the container:

```bash
docker run -p 127.0.0.1:8000:80 apan_5560genai:assignment2
```

Open http://127.0.0.1:8000/docs in a browser.

### Image Classification Endpoint

`POST /predict-image`

Upload an image using the multipart form field named `file`.
In Swagger UI, select **Try it out**, choose an image, and click
**Execute**.

Supported classes:

airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

Example response:

```json
{
  "filename": "test_cifar10.png",
  "class_index": 3,
  "class_name": "cat",
  "confidence": 0.6714
}
```

Confidence is the softmax score of the predicted class, not a
guarantee that the prediction is correct. Images outside the ten
CIFAR-10 classes are still assigned one of these classes.

### Verification

The Dockerized endpoint was tested with the first CIFAR-10 test
image, whose true label is `cat`. It returned HTTP 200 and predicted
`cat` with a confidence score of approximately 0.6714.

