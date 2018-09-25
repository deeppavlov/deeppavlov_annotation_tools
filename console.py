from argparse import ArgumentParser
import codecs
import importlib
import logging
import os
from typing import List

from flask import Flask
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS
import spacy

from keyword_extraction.tokenization import BaseTextPreprocessor
from keyword_extraction.keyword_extraction import KeywordExtractor
from ner.ner import NamedEntityRecognizer


def create_keywords_ner(args) -> NamedEntityRecognizer:
    keywords_list_name = os.path.normpath(args.destination_keywords_list)
    assert os.path.isfile(keywords_list_name), 'The file `{0}` does not exist!'.format(keywords_list_name)
    return NamedEntityRecognizer(spacy_model_name=args.spacy_lang, keywords_dictionary_name=keywords_list_name)


def ner_rest_api_service(cmd_args):

    class NerRestApi(Resource):
        recognizer = create_keywords_ner(cmd_args)

        def post(self):
            request_args = parser.parse_args()
            recognition_results = self.recognizer.recognize(request_args["text"])
            return {'text': request_args["text"], 'named_entities': recognition_results}

    app = Flask(__name__)
    cors = CORS(app, resources={r"/*": {"origins": "*"}})
    api = Api(app)
    parser = reqparse.RequestParser()
    parser.add_argument('text')
    api.add_resource(NerRestApi, '/ner')
    app.run(host = cmd_args.host_name, port = cmd_args.port_number)


def select_text_files(dir_name: str) -> List[str]:
    assert os.path.isdir(dir_name), 'A directory `{0}` does not exist!'.format(dir_name)
    return sorted(list(
        filter(
            lambda it3: not os.path.isdir(it3),
            map(
                lambda it2: os.path.join(dir_name, it2),
                filter(lambda it1: it1 not in {'.', '..'}, os.listdir(dir_name))
            )
        )
    ))


def create_preprocessor(preprocessor_class_name: str) -> BaseTextPreprocessor:
    module = importlib.import_module('keyword_extraction.tokenization')
    class_name = getattr(module, preprocessor_class_name)
    return class_name()


def select_keywords(args):
    keywords_list_name = os.path.normpath(args.destination_keywords_list)
    keywords_list_dir = os.path.dirname(keywords_list_name)
    if len(keywords_list_dir) > 0:
        assert os.path.isdir(keywords_list_dir), 'The directory `{0}` does not exist!'.format(keywords_list_dir)
    names_of_source_files = select_text_files(os.path.normpath(args.source_dir))
    assert len(names_of_source_files) > 0, 'Directory `{0}` is empty!'.format(args.source_dir)
    topic_model_name = os.path.normpath(args.topic_model_name)
    text_preprocessor = create_preprocessor(args.text_preprocessor)
    n_topics = args.topics_number
    assert n_topics > 1, '{0} is too small number of topics!'.format(n_topics)
    prob_threshold = args.probability_threshold
    assert prob_threshold > 0.0, '{0} is too small probability threshold!'.format(prob_threshold)
    assert prob_threshold < 1.0, '{0} is too large probability threshold!'.format(prob_threshold)
    extractor = KeywordExtractor(topic_model_name, n_topics, prob_threshold, args.use_nouns, args.use_verbs)
    spacy_nlp = spacy.load(args.spacy_lang)
    keywords = extractor.select_from_corpus(names_of_source_files, text_preprocessor, spacy_nlp)
    assert len(keywords) > 0, 'Keywords list is empty!'
    with codecs.open(keywords_list_name, mode='w', encoding='utf-8', errors='ignore') as fp:
        for cur_keyword in keywords:
            fp.write('{0}\n'.format(cur_keyword))


def use_ner(args):
    ner_rest_api_service(args)


def train_ner(args):
    pass


def main():
    main_parser = ArgumentParser()
    subparsers = main_parser.add_subparsers(dest='usage')

    parser_ner = subparsers.add_parser('ner')
    parser_training = subparsers.add_parser('training')
    parser_prepare_keywords = subparsers.add_parser('keywords')

    parser_prepare_keywords.add_argument('-s', '--src', dest='source_dir', type=str, required=True,
                                         help='A directory with source text files.')
    parser_prepare_keywords.add_argument('-d', '--dst', dest='destination_keywords_list', type=str, required=True,
                                         help='Name of text file into which a created keywords list will be written.')
    parser_prepare_keywords.add_argument('-n', '--name', dest='topic_model_name', type=str, required=True,
                                         help='Name of file into which a created topic model will be written.')
    parser_prepare_keywords.add_argument('-p', '--preprocessor', dest='text_preprocessor', type=str, required=True,
                                         help='Name of the text preprocessor class.')
    parser_prepare_keywords.add_argument('--topics', dest='topics_number', type=int, required=False, default=50,
                                         help='Number of topics.')
    parser_prepare_keywords.add_argument('--probability', dest='probability_threshold', type=float, required=False,
                                         default=1e-2, help='Minimal probability of keyword.')
    parser_prepare_keywords.add_argument('--spacy', dest='spacy_lang', type=str, required=False,
                                         default='en_core_web_lg', help='The SpaCy model name.')
    parser_prepare_keywords.add_argument('--nouns', dest='use_nouns', action='store_true', required=False,
                                         help='Do we want to use the noun phrases for keyword selection?')
    parser_prepare_keywords.add_argument('--verbs', dest='use_verbs', action='store_true', required=False,
                                         help='Do we want to use the root verbs for keyword selection?')

    parser_ner.add_argument('-k', '--keywords', dest='keywords_list', type=str, required=True,
                            help='Name of text file with keywords list.')
    parser_ner.add_argument('--spacy', dest='spacy_lang', type=str, required=False,
                            default='en_core_web_lg', help='The SpaCy model name.')

    args = main_parser.parse_args()
    if args.usage == 'keywords':
        select_keywords(args)
    elif args.usage == 'training':
        train_ner(args)
    elif args.usage == 'ner':
        use_ner(args)
    else:
        raise Exception("Error! `{0}` is unknown usage mode!".format(args.usage))


if __name__ == '__main__':
    logging.basicConfig(format='%(filename)s[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s] %(message)s',
                        level=logging.INFO)
    main()
