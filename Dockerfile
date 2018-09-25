FROM python:3.6-slim-stretch

RUN ["mkdir", "-p", "/usr/local/src/deeppavlov_annotation_tools"]
ADD . /usr/local/src/deeppavlov_annotation_tools
WORKDIR /usr/local/src/deeppavlov_annotation_tools

RUN apt-get update
RUN apt-get install -y curl gcc g++

RUN pip install -r requirements.txt

RUN python -m spacy download en_core_web_lg

EXPOSE 80

CMD python console.py -k model/oil_and_gas_keywords.txt --spacy en_core_web_lg --host 0.0.0.0 --port 80

HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -s -X POST -H 'Content-Type: application/json;charset=utf-8' -d '{"text": "Test test"}' http://localhost/ner || exit 1

