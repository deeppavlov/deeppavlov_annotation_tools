# deeppavlov_annotation_tools
Data annotation tools for NLP tasks

The project contains the whole pipeline for semi-automated data annotation: 
- term extraction
- initial data labelling
- model training
- automated data labelling
- manual label adjustment
...and so on (model-in-the-loop pipeline)

You can start a named entity recognizer (NER) by running the following command:

``` python -u console.py ner -k model/oil_and_gas_keywords.txt --spacy en_core_web_lg ```

Then you can send various texts for recognition, for example:

``` curl -X POST -d "text=What is the drill rig motor 5GEB22D voltage?"  http://localhost:5000/ner ```

For this example the NER will return something like this:

``` {"text": "What is the drill rig motor 5GEB22D voltage?", "named_entities": {"equipment": [[12, 17], [18, 21], [22, 27], [28, 35]], "operations": [], "brand": [], "properties": [], "property_values": []}} ```

#### Brat converter usage:
```
$brat_converter/text-to-brat.sh ./data output.json OilAndGasTextPreprocessor ./model/oil_and_gas_keywords.txt archive-name
```