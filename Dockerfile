FROM python:3.10

# Install system dependancies
RUN apt-get update
RUN apt-get install libsndfile1 ffmpeg gsutil -y

RUN pip3 install poetry 

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false
RUN poetry install --no-root

COPY . ./

RUN gsutil -m cp -r "gs://gresearch/sound_separation/bird_mixit_model_checkpoints"

ENV PYTHONPATH "${PYTHONPATH}:/app"
