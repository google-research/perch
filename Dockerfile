FROM python:3.11

# Environment variables
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV POETRY_VERSION=1.6.1

# Install poetry
RUN python -m pip install "poetry==$POETRY_VERSION"

# Install native dependencies
RUN apt-get -y update && apt-get -y install libsndfile1 ffmpeg

# Instal dependencies specified in the poetry configs
WORKDIR /app
ADD . /app
RUN poetry install

ENTRYPOINT [ "/usr/local/bin/poetry", "run" ]
