import codecs
import logging
import os
from typing import List, Union

import artm
from spacy.language import Language

from keyword_extraction.tokenization import BaseTextPreprocessor, SpaCyTokenizer


keyword_extraction_logger = logging.getLogger(__name__)


class KeywordExtractor:
    def __init__(self, topic_model_name: str, number_of_topics: int, probability_threshold: float=1e-2,
                 extract_noun_phrases: bool=True, extract_root_verbs: bool=False):
        self.number_of_topics = number_of_topics
        self.probability_threshold = probability_threshold
        self.extract_noun_phrases = extract_noun_phrases
        self.extract_root_verbs = extract_root_verbs
        self.topic_model_name = topic_model_name

    def select_from_corpus(self, list_of_files: List[str], preprocessor: BaseTextPreprocessor,
                           spacy_nlp: Language) -> List[str]:
        topic_model_name = os.path.normpath(self.topic_model_name.strip())
        if len(topic_model_name) == 0:
            raise ValueError('A topic model name is empty!')
        dir_name = os.path.dirname(topic_model_name)
        base_name = os.path.basename(topic_model_name)
        if len(dir_name) == 0:
            dir_name = os.path.curdir
        if len(base_name) == 0:
            raise ValueError('`{0}` is incorrect name for a topic model! Base name of file is empty!'.format(
                self.topic_model_name))
        if not os.path.isdir(dir_name):
            raise ValueError('`{0}` is incorrect name for a topic model! Directory `{1}` does not exist!'.format(
                self.topic_model_name, dir_name))
        collection_name = os.path.normpath(os.path.join(dir_name, base_name + '.collection'))
        collection_docword_name = os.path.normpath(os.path.join(dir_name, 'docword.' + base_name + '.collection'))
        collection_vocab_name = os.path.normpath(os.path.join(dir_name, 'vocab.' + base_name + '.collection'))
        if (not os.path.isfile(collection_docword_name)) or (not os.path.isfile(collection_vocab_name)):
            self.create_collection_as_bow_uci(list_of_files, preprocessor, spacy_nlp, collection_docword_name,
                                              collection_vocab_name)
        batches_path = os.path.normpath(os.path.join(dir_name, base_name + '.data_batches'))
        if os.path.isdir(batches_path):
            batch_vectorizer = artm.BatchVectorizer(data_path=batches_path, data_format='batches')
        else:
            batch_vectorizer = artm.BatchVectorizer(data_path=dir_name, data_format='bow_uci',
                                                    collection_name=collection_name, target_folder=batches_path)
        dictionary = artm.Dictionary()
        dictionary_name = os.path.normpath(topic_model_name + '.dictionary')
        if os.path.isfile(dictionary_name):
            dictionary.load(dictionary_name)
        else:
            dictionary.gather(data_path=batches_path)
            dictionary.save(dictionary_name)
        topic_model = self.load_topic_model(artm.ARTM(num_topics=self.number_of_topics, dictionary=dictionary,
                                                      cache_theta=False), topic_model_name)
        if topic_model is None:
            topic_model = self.create_topic_model(topic_model_name, batch_vectorizer, dictionary)
            if topic_model is None:
                raise ValueError('The trained topic model cannot be loaded from the file `{0}`!'.format(
                    topic_model_name))
        return self.select_keywords_from_topic_model(topic_model)

    def create_collection_as_bow_uci(self, list_of_files: List[str], preprocessor: BaseTextPreprocessor,
                                     spacy_nlp: Language, collection_docword_name: str, collection_vocab_name: str):
        documents = []
        global_token_frequencies = dict()
        for cur_name in list_of_files:
            for cur_text in preprocessor.get_texts_from_file(cur_name):
                cur_doc = spacy_nlp(cur_text)
                tokens = SpaCyTokenizer.tokenize_document(cur_doc, self.extract_noun_phrases, self.extract_root_verbs)
                if len(tokens) > 0:
                    token_frequencies = dict()
                    for cur_token in tokens:
                        token_frequencies[cur_token] = token_frequencies.get(cur_token.lower(), 0) + 1
                        global_token_frequencies[cur_token] = global_token_frequencies.get(cur_token, 0) + 1
                    documents.append(token_frequencies)
            keyword_extraction_logger.info('File `{0}` has been processed.'.format(cur_name))
        IDs_of_tokens = dict([
            (token_text, token_idx + 1) for token_idx, token_text in
            enumerate(sorted(list(global_token_frequencies.keys())))
        ])
        with codecs.open(collection_docword_name, mode='w', encoding='utf-8', errors='ignore') as fp:
            fp.write('{0}\n'.format(len(documents)))
            fp.write('{0}\n'.format(len(global_token_frequencies)))
            fp.write('{0}\n'.format(sum([global_token_frequencies[cur_token]
                                         for cur_token in global_token_frequencies])))
            for document_idx in range(len(documents)):
                for cur_token in documents[document_idx]:
                    fp.write('{0} {1} {2}\n'.format(document_idx + 1, IDs_of_tokens[cur_token],
                                                    documents[document_idx][cur_token]))
        with codecs.open(collection_vocab_name, mode='w', encoding='utf-8', errors='ignore') as fp:
            for cur_token in sorted(list(global_token_frequencies.keys())):
                fp.write('{0}\n'.format(cur_token))

    def select_keywords_from_topic_model(self, topic_model: artm.ARTM) -> List[str]:
        phi = topic_model.get_phi()
        all_words = phi.index
        n_words = all_words.shape[0]
        set_of_keywords = set()
        for topic_name in phi.columns:
            column = phi[topic_name]
            set_of_keywords |= set(
                map(
                    lambda keyword_and_probability: keyword_and_probability[0],
                    filter(
                        lambda value: value[1] >= self.probability_threshold,
                        map(lambda idx: (all_words[idx].replace('_', ' '), column[all_words[idx]]), range(n_words))
                    )
                )
            )
            del column
        return sorted(list(set_of_keywords))

    def create_topic_model(self, topic_model_name: str, batch_vectorizer: artm.BatchVectorizer,
                           dictionary: artm.Dictionary) -> artm.ARTM:
        topic_model = artm.ARTM(num_topics=self.number_of_topics, dictionary=dictionary, cache_theta=False)
        topic_model.scores.add(artm.PerplexityScore(name='perplexity_score', dictionary=dictionary))
        topic_model.scores.add(artm.SparsityPhiScore(name='sparsity_phi_score'))
        topic_model.scores.add(artm.SparsityThetaScore(name='sparsity_theta_score'))
        topic_model.num_document_passes = 5
        topic_model.num_processors = max(1, os.cpu_count() - 1)
        topic_model.regularizers.add(artm.SmoothSparsePhiRegularizer(name='sparse_phi_regularizer'))
        topic_model.regularizers.add(artm.SmoothSparseThetaRegularizer(name='sparse_theta_regularizer'))
        topic_model.regularizers.add(artm.DecorrelatorPhiRegularizer(name='decorrelator_phi_regularizer'))
        topic_model.regularizers['sparse_phi_regularizer'].tau = -1.0
        topic_model.regularizers['sparse_theta_regularizer'].tau = -0.5
        topic_model.regularizers['decorrelator_phi_regularizer'].tau = 1e+5
        best_score = None
        keyword_extraction_logger.info('epoch  perplexity_score  sparsity_phi_score  sparsity_theta_score')
        for restart_index in range(10):
            topic_model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=3)
            if best_score is None:
                best_score = topic_model.score_tracker['perplexity_score'].last_value
            else:
                if best_score > topic_model.score_tracker['perplexity_score'].last_value:
                    best_score = topic_model.score_tracker['perplexity_score'].last_value
                    self.save_topic_model(topic_model, topic_model_name)
            keyword_extraction_logger.info('{0:5}  {1:16.9}  {2:18.9}  {3:20.9}'.format(
                (restart_index + 1) * 3,
                topic_model.score_tracker['perplexity_score'].last_value,
                topic_model.score_tracker['sparsity_phi_score'].last_value,
                topic_model.score_tracker['sparsity_theta_score'].last_value
            ))
        del topic_model
        return self.load_topic_model(artm.ARTM(num_topics=self.number_of_topics, dictionary=dictionary,
                                               cache_theta=False), topic_model_name)

    @staticmethod
    def load_topic_model(topic_model: artm.ARTM, file_name: str) -> Union[artm.ARTM, None]:
        if (not os.path.isfile(file_name + '.p_wt')) or (not os.path.isfile(file_name + '.n_wt')):
            return None
        topic_model.load(os.path.join(file_name + '.p_wt'), 'p_wt')
        topic_model.load(os.path.join(file_name + '.n_wt'), 'n_wt')
        return topic_model

    @staticmethod
    def save_topic_model(topic_model: artm.ARTM, file_name: str):
        topic_model.save(os.path.join(file_name + '.p_wt'), 'p_wt')
        topic_model.save(os.path.join(file_name + '.n_wt'), 'n_wt')
