FROM python:3.8

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

RUN apt-get -y update && \
    apt-get install zsh -y && \
    apt-get install git-lfs

RUN git lfs install && \
    git submodule update --recursive --init

CMD ["/bin/bash"]
# EXPOSE 8080


