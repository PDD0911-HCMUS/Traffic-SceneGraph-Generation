FROM python:3
USER root

WORKDIR .
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install numpy
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
#RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118