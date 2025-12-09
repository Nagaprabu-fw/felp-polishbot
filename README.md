# FastAPI Docker & Kubernetes Example

## Docker Usage

1. Export Image Tag Version:

   ```bash
   export IMAGE_TAG_VERSION=v1.0.0
   ```

2. Build the image:
   ```bash
   cd app
   podman build --arch amd64 -t polishbot/felp-polishbot:${IMAGE_TAG_VERSION} .
   ```

3. Run the container:
   ```bash
   podman run --arch amd64 -p 8000:8000 polishbot/felp-polishbot:${IMAGE_TAG_VERSION}
   ```

3. Push the Image:
   ```bash
   podman tag polishbot/felp-polishbot:${IMAGE_TAG_VERSION} 950310456124.dkr.ecr.us-east-1.amazonaws.com/polishbot/felp-polishbot:${IMAGE_TAG_VERSION}
   podman push 950310456124.dkr.ecr.us-east-1.amazonaws.com/polishbot/felp-polishbot:${IMAGE_TAG_VERSION}
   ```

## Kubernetes Deployment

See `k8s-deployment.yaml` and `k8s-service.yaml` for deployment and service manifests.

