FROM python:3.6-slim-stretch

# Update and install base dependencies
RUN apt-get --yes update
RUN apt-get --yes install -y curl gcc g++ git make cmake build-essential libboost-all-dev

# Install python packages for BigARTM
RUN apt-get --yes install python-numpy python-pandas python-scipy
RUN pip install protobuf tqdm wheel

# Clone the BigARTM repository, build and install
RUN git clone --branch=stable --depth=1 https://github.com/bigartm/bigartm.git
WORKDIR bigartm
RUN mkdir build && cd build && cmake -DCMAKE_INSTALL_PREFIX:PATH=/usr .. && make -j && make install
RUN cd 3rdparty/protobuf-3.0.0/python && setup.py build && python setup.py install
RUN cd python && python setup.py install
ENV ARTM_SHARED_LIBRARY=/tmp/bigartm/build/lib/libartm.so

# Create directory for this package

RUN ["mkdir", "-p", "/usr/local/src/deeppavlov_annotation_tools"]
ADD . /usr/local/src/deeppavlov_annotation_tools
WORKDIR /usr/local/src/deeppavlov_annotation_tools

# Install all Python dependencies of this package
RUN pip install -r requirements.txt

# Download the SpaCy model for English
RUN python -m spacy download en_core_web_lg

EXPOSE 80

CMD python ner -k model/oil_and_gas_keywords.txt --spacy en_core_web_lg --host 0.0.0.0 --port 80

HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -s -X POST -H 'Content-Type: application/json;charset=utf-8' -d '{"text": "Test test"}' http://localhost/ner || exit 1

