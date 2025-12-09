FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


# export IMAGE_TAG_VERSION=v0.1.0
# cd app
# podman build --arch amd64 -t 950310456124.dkr.ecr.us-east-1.amazonaws.com/polishbot/felp-polishbot:${IMAGE_TAG_VERSION} .
# podman push 950310456124.dkr.ecr.us-east-1.amazonaws.com/polishbot/felp-polishbot:${IMAGE_TAG_VERSION}

# podman run --arch amd64 -p 8000:8000 950310456124.dkr.ecr.us-east-1.amazonaws.com/polishbot/felp-polishbot:${IMAGE_TAG_VERSION}

