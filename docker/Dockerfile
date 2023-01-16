FROM tensorflow/tensorflow
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY requirements.txt /code/requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r code/requirements.txt
COPY code /code
COPY docker/run_inferr.sh /run_inferr.sh
RUN mkdir /info
RUN chmod +x ./run_inferr.sh
