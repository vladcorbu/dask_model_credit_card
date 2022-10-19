FROM ubuntu:latest

ARG USER=dutu

USER root

RUN apt-get -y update
RUN apt-get -y upgrade

RUN apt install -y python3-pip
RUN pip3 install --upgrade pip

RUN useradd -ms /bin/bash ${USER}
USER ${USER}
WORKDIR /home/${USER}
RUN mkdir -p /home/${USER}/.aws
COPY ./aws/credentials /home/${USER}/.aws
COPY ./aws/config /home/${USER}/.aws

COPY . .
RUN pip3 install -r requirements.txt

CMD [ "python3", "model.py"]