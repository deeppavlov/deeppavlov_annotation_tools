import codecs
from collections import OrderedDict
import logging
from typing import List, Union

import spacy
from spacy.matcher import PhraseMatcher


ner_logger = logging.getLogger(__name__)


class NamedEntityRecognizer:
    def __init__(self, spacy_model_name: str, keywords_dictionary_name: str=None):
        all_keywords = self.load_keywords_dictionary(keywords_dictionary_name)
        self.base_nlp = spacy.load(spacy_model_name)
        self.keyword_searcher = PhraseMatcher(self.base_nlp.vocab)
        self.keyword_searcher.add('ANY_KEYWORD', None, *[self.base_nlp(it) for it in all_keywords])
        del all_keywords
        ner_logger.info('NER has been initialized.')

    def recognize(self, text: str) -> OrderedDict:
        ner_logger.info('Recognition with NER is started.')
        doc = self.base_nlp(text)
        n_words = len(doc)
        entities = OrderedDict([
            ('equipment', []),
            ('operations', []),
            ('properties', []),
            ('property_values', [])
        ])
        if (n_words == 0) or all(map(lambda token: token.is_space, doc)):
            ner_logger.info('Recognition with NER is finished.')
            return entities
        used_words = [False for _ in range(n_words)]
        start_ne_pos = -1
        ne_type = None
        for token in doc:
            if token.ent_iob_ in {'O', 'B'}:
                if start_ne_pos >= 0:
                    prepared_ne_type = self.ontonotes_ne_to_our_ne(ne_type)
                    if len(prepared_ne_type) > 0:
                        for token_idx in range(start_ne_pos, token.i):
                            used_words[token_idx] = True
                        entities[prepared_ne_type].append((start_ne_pos, token.i))
                    if token.ent_iob_ == 'O':
                        start_ne_pos = -1
                        ne_type = None
                    else:
                        ne_type = token.ent_type_
                        start_ne_pos = token.i
                else:
                    if token.ent_iob_ == 'B':
                        ne_type = token.ent_type_
                        start_ne_pos = token.i
            else:
                if start_ne_pos >= 0:
                    if ne_type != token.ent_type_:
                        prepared_ne_type = self.ontonotes_ne_to_our_ne(ne_type)
                        if len(prepared_ne_type) > 0:
                            for token_idx in range(start_ne_pos, token.i):
                                used_words[token_idx] = True
                            entities[prepared_ne_type].append((start_ne_pos, token.i))
                        ne_type = token.ent_type_
                        start_ne_pos = token.i
                else:
                    ne_type = token.ent_type_
                    start_ne_pos = token.i
        if start_ne_pos >= 0:
            prepared_ne_type = self.ontonotes_ne_to_our_ne(ne_type)
            if len(prepared_ne_type) > 0:
                for token_idx in range(start_ne_pos, n_words):
                    used_words[token_idx] = True
                entities[prepared_ne_type].append((start_ne_pos, n_words))
        for cur_match in self.keyword_searcher(doc):
            start_ne_pos = cur_match[1]
            end_ne_pos = cur_match[2]
            can_add = True
            is_noun = False
            is_verb = False
            for token_idx in range(start_ne_pos, end_ne_pos):
                if used_words[token_idx]:
                    can_add = False
                    break
                if doc[token_idx].pos_ == 'NOUN':
                    is_noun = True
                elif doc[token_idx].pos_ == 'VERB':
                    is_verb = True
            if can_add:
                if is_verb:
                    can_add = False
                    for token_idx in range(start_ne_pos, end_ne_pos):
                        if doc[token_idx].dep_ == 'ROOT':
                            can_add = True
                            break
                    if can_add:
                        can_add = False
                        entities['operations'].append((start_ne_pos, end_ne_pos))
                        for token_idx in range(start_ne_pos, end_ne_pos):
                            used_words[token_idx] = True
                if is_noun and can_add:
                    entities['equipment'].append((start_ne_pos, end_ne_pos))
                    for token_idx in range(start_ne_pos, end_ne_pos):
                        used_words[token_idx] = True
        if len(entities['equipment']) > 0:
            for cur_entity_bounds in entities['equipment']:
                left_pos = cur_entity_bounds[0]
                right_pos = cur_entity_bounds[1]
                for token_idx in range(cur_entity_bounds[0], cur_entity_bounds[1]):
                    if doc[token_idx].left_edge.i < left_pos:
                        left_pos = doc[token_idx].left_edge.i
                    if (doc[token_idx].right_edge.i + 1) > right_pos:
                        right_pos = doc[token_idx].right_edge.i + 1
                if left_pos < cur_entity_bounds[0]:
                    can_add = False
                    for token_idx in range(left_pos, cur_entity_bounds[0]):
                        if doc[token_idx].pos_ in {'NOUN', 'PROPN'}:
                            can_add = True
                            break
                    if can_add:
                        start_ne_pos = left_pos
                        end_ne_pos = cur_entity_bounds[0]
                        if any(map(lambda token_idx: not doc[token_idx].is_stop, range(start_ne_pos, end_ne_pos))):
                            for token_idx in range(start_ne_pos, end_ne_pos):
                                if used_words[token_idx]:
                                    can_add = False
                                    break
                            if can_add:
                                entities['properties'].append((start_ne_pos, end_ne_pos))
                                for token_idx in range(start_ne_pos, end_ne_pos):
                                    used_words[token_idx] = True
                if right_pos > cur_entity_bounds[1]:
                    can_add = False
                    for token_idx in range(cur_entity_bounds[1], right_pos):
                        if doc[token_idx].pos_ in {'NOUN', 'PROPN', 'ADJ'}:
                            can_add = True
                            break
                    if can_add:
                        start_ne_pos = cur_entity_bounds[1]
                        end_ne_pos = right_pos
                        if any(map(lambda token_idx: not doc[token_idx].is_stop, range(start_ne_pos, end_ne_pos))):
                            for token_idx in range(start_ne_pos, end_ne_pos):
                                if used_words[token_idx]:
                                    can_add = False
                                    break
                            if can_add:
                                entities['properties'].append((start_ne_pos, end_ne_pos))
                                for token_idx in range(start_ne_pos, end_ne_pos):
                                    used_words[token_idx] = True
        set_of_digits = set('0123456789')
        for token in doc:
            if token.is_currency or ((len(set(token.text) & set_of_digits) > 0) and (not token.is_digit) and
                                     (not token.like_num)):
                if not used_words[token.i]:
                    used_words[token.i] = True
                    if self.is_measure(token.text):
                        entities['property_values'].append((token.i, token.i + 1))
                    else:
                        entities['equipment'].append((token.i, token.i + 1))
        ner_logger.info('Recognition with NER is finished.')
        for prepared_ne_type in entities:
            entities[prepared_ne_type] = sorted(
                list(map(
                    lambda it: (doc[it[0]].idx, doc[it[1] - 1].idx + len(doc[it[1] - 1].text)),
                    entities[prepared_ne_type]
                )),
                key=lambda it: (it[0], it[1])
            )
        return entities

    @staticmethod
    def is_measure(text: str) -> bool:
        start_idx = -1
        for idx in range(len(text)):
            if not text[idx].isdigit():
                start_idx = idx
                break
        if start_idx < 0:
            return True
        if start_idx == 0:
            return False
        res = True
        for idx in range(start_idx, len(text)):
            if text[idx].isdigit():
                res = False
                break
        return res

    @staticmethod
    def ontonotes_ne_to_our_ne(ontonotes_ne: Union[str, None]) -> str:
        if ontonotes_ne == 'ORG':
            return 'equipment'
        if ontonotes_ne == 'PRODUCT':
            return 'equipment'
        if ontonotes_ne == 'QUANTITY':
            return 'property_values'
        return ''


    @staticmethod
    def load_keywords_dictionary(file_name: str) -> List[str]:
        res = set()
        with codecs.open(file_name, mode='r', encoding='utf-8', errors='ignore') as fp:
            cur_line = fp.readline()
            while len(cur_line) > 0:
                prep_line = cur_line.strip().lower()
                if len(prep_line) > 0:
                    res.add(prep_line)
                    res.add(prep_line.upper())
                    res.add(' '.join(list(map(lambda it: it.title(), prep_line.split()))))
                    res.add(prep_line.title())
                cur_line = fp.readline()
        return sorted(list(res), key=lambda it: (-len(it), it))