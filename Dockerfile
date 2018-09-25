FROM python:3.6-slim-stretch

RUN ["mkdir", "-p", "/usr/local/src/deeppavlov_annotation_tools"]
ADD . /usr/local/src/deeppavlov_annotation_tools
WORKDIR /usr/local/src/deeppavlov_annotation_tools

# Update and install base dependencies
RUN apt-get --yes update
RUN apt-get --yes install -y curl gcc g++ git make cmake build-essential libboost-all-dev

# Install python packages for BigARTM
RUN apt-get --yes install python-numpy python-pandas python-scipy
wget https://bootstrap.pypa.io/get-pip.py
python get-pip.py
pip install protobuf tqdm wheel

# Clone the BigARTM repository and build
git clone --branch=stable https://github.com/bigartm/bigartm.git
cd bigartm
mkdir build && cd build
cmake ..
make

# Install BigARTM
make install
export ARTM_SHARED_LIBRARY=/usr/local/lib/libartm.so

# Install all Python dependencies of this package
RUN pip install -r requirements.txt

# Download the SpaCy model for English
RUN python -m spacy download en_core_web_lg

EXPOSE 80

CMD python ner -k model/oil_and_gas_keywords.txt --spacy en_core_web_lg --host 0.0.0.0 --port 80

HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -s -X POST -H 'Content-Type: application/json;charset=utf-8' -d '{"text": "Test test"}' http://localhost/ner || exit 1

