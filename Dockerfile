FROM apache/beam_python3.10_sdk:2.46.0

# Install libsndfile1
RUN apt-get update \
  && apt-get install -y --no-install-recommends libsndfile1 ffmpeg

# Install Poetry
ENV POETRY_HOME="/opt/poetry"
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"
RUN poetry config virtualenvs.create false

# Install the BIRB codebase and its dependenciees
COPY ./ ./
RUN poetry install

# Set the entrypoint to the Apache Beam SDK launcher.
ENTRYPOINT ["/opt/apache/beam/boot"]
