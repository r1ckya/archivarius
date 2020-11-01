FROM ubuntu:18.04

RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y python3.6 python3.6-dev python3-pip

RUN ln -sfn /usr/bin/python3.6 /usr/bin/python3 && ln -sfn /usr/bin/python3 /usr/bin/python && ln -sfn /usr/bin/pip3 /usr/bin/pip

RUN apt-get -y install tesseract-ocr libtesseract-dev wget git
RUN cd /usr/share/tesseract-ocr/4.00/tessdata \
  && wget https://raw.githubusercontent.com/tesseract-ocr/tessdata/master/rus.traineddata

RUN apt-get install -y ghostscript
RUN apt-get install -y poppler-utils

RUN pip install scikit-learn \
    Pillow \
    pre-commit \
    pdf2image \
    flask \
    XlsxWriter \
    pandas \
    walrus \
    seaborn \
    werkzeug \
    PyMySQL

RUN apt-get install -y language-pack-ru
ENV LANGUAGE ru_RU.UTF-8
ENV LANG ru_RU.UTF-8
ENV LC_ALL ru_RU.UTF-8
RUN locale-gen ru_RU.UTF-8 && dpkg-reconfigure locales



