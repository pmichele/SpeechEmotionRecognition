FROM python:3.7-stretch

# Install build tools
RUN apt-get update && \
	apt-get install -y gcc make apt-transport-https ca-certificates build-essential && \
	apt install unzip && \
	apt-get install -y supervisor && \
	apt-get -y install libsndfile-dev && \
    mkdir -p /var/log/supervisor

# Inspect python version
RUN python3 --version
RUN pip3 --version

WORKDIR /usr/src/SpeechEmotionRecognition

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the emo-db dataset and remove all unnecessary files
RUN mkdir /raw_data && cd /raw_data && \
    curl -o data.zip http://emodb.bilderbar.info/download/download.zip && \
    unzip data.zip && \
    mv wav .. && \
    rm -r * && \
    mv ../wav/* . && \
    rmdir ../wav

# Copy application files
COPY src/ /src/
COPY models/ /models/
RUN ls -la /src/*

# Set environment variables
ENV PYTHONPATH "${PYTHONPATH}:/"
ENV FLASK_APP /src/SpeechEmotionRecognition.py
ENV FLASK_RUN_HOST 0.0.0.0

# Produce the dataset for the flask model
RUN ["python3", "/src/data_preparation/DataPreparation.py"]

# Set up supervisord
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
RUN mkdir -p /var/log/supervisord/

WORKDIR ~
COPY Report_SER.ipynb .

CMD ["/usr/bin/supervisord"]