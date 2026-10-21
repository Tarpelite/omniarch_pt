# CPU image for the OmniArch mini demo. For NVIDIA GPUs, build with
#   --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128
# and run with --gpus all.
FROM python:3.11-slim
ARG PIP_INDEX=https://pypi.org/simple
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url ${TORCH_INDEX}
COPY pyproject.toml README.md ./
COPY omniarch ./omniarch
RUN pip install --no-cache-dir -i ${PIP_INDEX} -e .
COPY examples ./examples
# weights are mounted at runtime: -v /path/to/weights:/app/weights
EXPOSE 7860
CMD ["omniarch", "serve", "--host", "0.0.0.0", "--port", "7860"]
