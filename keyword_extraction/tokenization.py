import codecs
import re
from typing import Tuple, List

from spacy.tokens.doc import Doc

class SpaCyTokenizer:
    @staticmethod
    def strip_noun_phrase(doc: Doc, noun_phrase_start: int, noun_phrase_end: int) -> Tuple[int, int]:
        start_token_idx = noun_phrase_start
        while start_token_idx < noun_phrase_end:
            if doc[start_token_idx].is_stop or (doc[start_token_idx].pos_ in {'DET', 'ADP', 'PRON', 'PUNCT'}):
                start_token_idx += 1
            else:
                break
        if start_token_idx >= noun_phrase_end:
            return (-1, -1)
        end_token_idx = noun_phrase_end - 1
        while end_token_idx > start_token_idx:
            if doc[end_token_idx].is_stop or (doc[end_token_idx].pos_ in {'DET', 'ADP', 'PRON', 'PUNCT'}):
                end_token_idx -= 1
            else:
                break
        end_token_idx += 1
        return start_token_idx, end_token_idx

    @staticmethod
    def get_text_of_noun_phrase(doc: Doc, noun_phrase_start: int, noun_phrase_end: int) -> str:
        tokens = []
        for token_idx in range(noun_phrase_start, noun_phrase_end):
            if (not doc[token_idx].is_punct) and (not doc[token_idx].is_space):
                token_text = doc[token_idx].norm_.strip()
                if (len(token_text) > 0) and token_text.isalnum():
                    tokens.append(token_text)
        if all(map(lambda it: it.isdigit(), tokens)):
            return ''
        return '_'.join(tokens)

    @staticmethod
    def tokenize_document(doc: Doc, select_noun_phrases: bool, select_verbs: bool) -> List[str]:
        n_tokens = len(doc)
        used_words = [False for _ in range(n_tokens)]
        selected_phrases = list()
        ne_start_idx = -1
        for token_idx in range(n_tokens):
            if doc[token_idx].ent_iob_ == 'B':
                if ne_start_idx >= 0:
                    can_add = True
                    for token_idx_2 in range(ne_start_idx, token_idx):
                        if used_words[token_idx_2]:
                            can_add = False
                            break
                    if can_add:
                        phrase_bounds = SpaCyTokenizer.strip_noun_phrase(doc, ne_start_idx, token_idx)
                        if (phrase_bounds[0] >= 0) and (phrase_bounds[1] >= 0):
                            for token_idx in range(phrase_bounds[0], phrase_bounds[1]):
                                used_words[token_idx] = True
                            selected_phrases.append(phrase_bounds)
                ne_start_idx = token_idx
            elif doc[token_idx].ent_iob_ != 'I':
                if ne_start_idx >= 0:
                    can_add = True
                    for token_idx_2 in range(ne_start_idx, token_idx):
                        if used_words[token_idx_2]:
                            can_add = False
                            break
                    if can_add:
                        phrase_bounds = SpaCyTokenizer.strip_noun_phrase(doc, ne_start_idx, token_idx)
                        if (phrase_bounds[0] >= 0) and (phrase_bounds[1] >= 0):
                            for token_idx in range(phrase_bounds[0], phrase_bounds[1]):
                                used_words[token_idx] = True
                            selected_phrases.append(phrase_bounds)
                ne_start_idx = -1
        if ne_start_idx >= 0:
            can_add = True
            for token_idx_2 in range(ne_start_idx, n_tokens):
                if used_words[token_idx_2]:
                    can_add = False
                    break
            if can_add:
                phrase_bounds = SpaCyTokenizer.strip_noun_phrase(doc, ne_start_idx, n_tokens)
                if (phrase_bounds[0] >= 0) and (phrase_bounds[1] >= 0):
                    for token_idx in range(phrase_bounds[0], phrase_bounds[1]):
                        used_words[token_idx] = True
                    selected_phrases.append(phrase_bounds)
        if select_noun_phrases:
            for cur_phrase in doc.noun_chunks:
                phrase_bounds = SpaCyTokenizer.strip_noun_phrase(doc, cur_phrase.start, cur_phrase.end)
                if (phrase_bounds[0] >= 0) and (phrase_bounds[1] >= 0):
                    can_add = True
                    for token_idx_2 in range(phrase_bounds[0], phrase_bounds[1]):
                        if used_words[token_idx_2]:
                            can_add = False
                            break
                    if can_add:
                        selected_phrases.append(phrase_bounds)
                        for token_idx in range(phrase_bounds[0], phrase_bounds[1]):
                            used_words[token_idx] = True
        if select_verbs:
            for token_idx in range(n_tokens):
                if doc[token_idx].dep == 'ROOT':
                    if not used_words[token_idx]:
                        used_words[token_idx] = True
                        selected_phrases.append((token_idx, token_idx + 1))
        selected_phrases.sort()
        return list(filter(lambda it2: len(it2) > 0, map(lambda it1: SpaCyTokenizer.get_text_of_noun_phrase(
            doc, it1[0], it1[1]), selected_phrases)))


class BaseTextPreprocessor:
    def __init__(self):
        self.__re_for_tokenization = [
            re.compile(r'\w:\d', re.U), re.compile(r'\d%\w', re.U), re.compile(r'\w[\\/]\w', re.U),
            re.compile(r'.\w[\\/]', re.U), re.compile(r'\w\+\w', re.U), re.compile(r'.\w\+\S', re.U)
        ]

    def get_texts_from_file(self, file_name: str) -> List[str]:
        raise NotImplemented

    def tokenize_source_text(self, source_text: str) -> str:
        tokenized = source_text.replace('&quot;', '"').replace('&gt;', '>').replace('&lt;', '<')
        for cur_re in self.__re_for_tokenization:
            search_res = cur_re.search(tokenized)
            while search_res is not None:
                if (search_res.start() < 0) or (search_res.end() < 0):
                    search_res = None
                else:
                    tokenized = tokenized[:(search_res.start() + 2)] + ' ' + tokenized[(search_res.start() + 2):]
                    search_res = cur_re.search(tokenized, pos=search_res.end() + 1)
        return tokenized


class OilAndGasTextPreprocessor(BaseTextPreprocessor):
    def __init__(self):
        super().__init__()
        special_unicode_characters = {'\u00A0', '\u2003', '\u2002', '\u2004', '\u2005', '\u2006', '\u2009', '\u200A',
                                      '\u0000', '\r', '\n', '\t'}
        self.re_for_space = re.compile('[' + ''.join(special_unicode_characters) + ']+', re.U)
        self.re_for_unicode = re.compile(r'&#\d+;')
        self.min_characters_in_line = 20
        self.min_characters_in_text = 200

    def get_texts_from_file(self, file_name: str) -> List[str]:
        all_texts = []
        lines_of_text = []
        with codecs.open(file_name, mode='r', encoding='utf-8', errors='ignore') as fp:
            cur_line = fp.readline()
            while len(cur_line) > 0:
                prep_line = cur_line.strip()
                if len(prep_line) > 0:
                    prep_line = self.tokenize_source_text(prep_line).strip().replace('ﬁ', 'fl')
                    if len(prep_line) > 0:
                        prep_line = self.re_for_space.sub(' ', prep_line).strip()
                        if len(prep_line) > 0:
                            search_res = self.re_for_unicode.search(prep_line)
                            while search_res is not None:
                                if (search_res.start() < 0) or (search_res.end() < 0):
                                    search_res = None
                                else:
                                    unicode_value = int(prep_line[(search_res.start() + 2):(search_res.end() - 1)])
                                    prep_line = prep_line[:search_res.start()] + chr(unicode_value) + \
                                                prep_line[search_res.end():]
                                    search_res = self.re_for_unicode.search(prep_line)
                        if len(prep_line) > 0:
                            lines_of_text.append(prep_line)
                else:
                    if len(lines_of_text) > 0:
                        if all(map(lambda it: len(it) >= self.min_characters_in_line, lines_of_text)):
                            new_text = lines_of_text[0]
                            for new_line in lines_of_text[1:]:
                                if new_text.endswith('-') and (not new_text[-2].isspace()):
                                    new_text += new_line
                                else:
                                    new_text += (' ' + new_line)
                            if len(new_text) >= self.min_characters_in_text:
                                all_texts.append(new_text)
                    lines_of_text.clear()
                cur_line = fp.readline()
        return all_texts
