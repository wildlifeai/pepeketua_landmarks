FROM python:3.8.2
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY requirements.txt /code/requirements.txt
RUN pip install -r code/requirements.txt
COPY code /code
COPY docker/run_inferr.sh /run_inferr.sh
RUN mkdir /info
