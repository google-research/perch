FROM python:3.10

# Install system dependancies
RUN apt-get update
RUN apt-get install libsndfile1 ffmpeg -y

RUN pip3 install poetry 

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false
RUN poetry install --no-root

COPY . ./

# Downloading gcloud package
RUN curl https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-444.0.0-linux-x86_64.tar.gz > /tmp/google-cloud-sdk.tar.gz

# Installing the package
RUN mkdir -p /usr/local/gcloud \
    && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
    && /usr/local/gcloud/google-cloud-sdk/install.sh

# Adding the package path to local
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin

# Download the bird classification models
RUN gsutil -m cp -r \
    "gs://chirp-public-bucket/birbsep_paper" .

# Download the bird separation models
RUN gsutil -m cp -r \
    "gs://gresearch/sound_separation/bird_mixit_model_checkpoints" .

# Update the pythonpath
ENV PYTHONPATH "${PYTHONPATH}:/app"
