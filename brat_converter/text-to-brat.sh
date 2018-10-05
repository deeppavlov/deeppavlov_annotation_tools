#!/bin/bash
#
# Usage: ./text-to-brat.sh ./data your.json OilAndGasTextPreprocessor ./model/oil_and_gas_keywords.txt brat-out
#
CONVERTER_FOLDER=brat_converter

python console.py training -s $1 -d $2 -p $3 -k $4 --spacy en_core_web_lg
python $CONVERTER_FOLDER/json2brat.py --source $2 --destination $5 --ann-conf $CONVERTER_FOLDER/annotation.conf

echo "Ready!"
