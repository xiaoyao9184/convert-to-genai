# Convert to ONNX-GenAI

A Gradio Docker image built via GitHub Actions.

## Why

This project is similar to [onnx-community/convert-to-onnx](https://huggingface.co/spaces/onnx-community/convert-to-onnx),  
but it is based on [microsoft/onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai).

It uses GitHub Actions to build and publish Docker images, and to sync with Hugging Face Gradio Spaces.  
The goal is to keep the entire process clean and minimal, without custom configuration files.

## Spaces

The Hugging Face Space for this project is located at:  
👉 [xiaoyao9184/convert-to-genai](https://huggingface.co/spaces/xiaoyao9184/convert-to-genai)

## Tags

The Docker images are published to Docker Hub under:  
👉 [xiaoyao9184/convert-to-genai](https://hub.docker.com/r/xiaoyao9184/convert-to-genai)

Image tags are generated using the `commit_id` and branch name (`main`).  
See the tagging workflow in [docker-image-tag-commit.yml](./.github/workflows/docker-image-tag-commit.yml).

> **Note:** Currently, only the `linux/amd64` platform is supported.

## Change / Customize

You can fork this project and build your own image.  
You will need to provide the following secrets: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`, `HF_USERNAME`, and `HF_TOKEN`.

See [docker/login-action](https://github.com/docker/login-action#docker-hub) for more details.
