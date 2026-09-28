\# APAN 5560 Assignment 1



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

