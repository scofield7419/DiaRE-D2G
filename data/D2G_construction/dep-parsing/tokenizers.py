#!/usr/bin/env python
# coding:utf-8

import re
from abc import ABCMeta, abstractmethod
from typing import List, Union

import en_core_web_sm
from pyxtension.streams import slist

__author__ = 'ASU'

from spacy.lang.en import English
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span

from spacy.tokens.token import Token

from .gramm_types import CTags
from .sentence_types import AbstractParsedSentence, AbstractSyntacticTree, GrammDep, SyntToken, \
    SyntWordToken

creReplaceNLs = re.compile(r'[\n\r\t\s]+')


class TextTokenizer(object, metaclass=ABCMeta):
    @abstractmethod
    def tokenizeText(self, text):
        pass


class NLTKTokenizer(TextTokenizer):
    def __init__(self):
        # Initializing TreeBank tokenizer from NLTK
        from nltk.tokenize import TreebankWordTokenizer
        self._tb_tokenizer = TreebankWordTokenizer().tokenize
        # Initializing Punkt Sentence Tokenizer from NLTK
        from nltk import data
        self._sent_detector = data.load('tokenizers/punkt/english.pickle')

    def tokenizeText(self, text: str) -> slist:
        """
        Uses a sentence tokenizer, and tokenize obtained sentences with a TreeBank tokenizer.
        Replace unnormal quotes.
        """
        sentences = self.__tokenizeToSentences(text)
        tokens = slist()
        for sent in sentences:
            sent = creReplaceNLs.sub(r' ', sent)
            tokens.extend(self._tb_tokenizer(
                sent))  # Tokenize sentences using TreeBank tokenizer initialized upper in the __init__ function
        return tokens

    def __tokenizeToSentences(self, text):
        text = re.sub(r'[`\x92\x91]', r"'", text)
        text = re.sub(r'[\x93\x94\x95\x96\x85\xE9]', r'"', text)
        text = re.sub(r'[\x80-\xFF]', r' ', text)
        text = re.sub(r"([:\s])\'(.+?)\'([\s\.])", r'\1"\2"\3', text)

        sentences = self._sent_detector.tokenize(text.strip())
        return sentences


class SpacySentence(AbstractParsedSentence):
    WORD_RE = re.compile(r"^[\w']+$", re.I)

    def __init__(self, tokens: List[Token], text: str) -> None:
        self._tokens = tokens
        self._text = text

    def __str__(self):
        return self._text

    def __repr__(self):
        return str({"instanceType": "SpacyParsedSentence", "val": self.__str__()})

    def getTaggedText(self) -> slist[Union[SyntToken, SyntWordToken]]:
        aText = slist()
        for i, token in enumerate(self._tokens):
            tag = CTags.fromString(token.tag_)
            if CTags.isWordType(tag):
                sn = SyntWordToken(text=token.text, tag=tag, stemmed=token.lemma_, idx=i)
            else:
                if token.is_punct or token.pos_ in ("SYM", "PUNCT"):
                    tag = CTags.SYM
                elif tag in (CTags.CD, CTags.SYM, CTags.LS, CTags.UH):
                    pass
                else:
                    tag = CTags.fromString("OTHER_TAG")
                sn = SyntToken(text=token.text, tag=tag, idx=i)
            aText.append(sn)
        return aText

    def getBestTree(self) -> AbstractSyntacticTree:
        raise NotImplementedError("Spacy doesn't produce classical syntactical tree of phrases")

    def getDependencies(self) -> slist[GrammDep]:
        raise NotImplementedError("Not implemented yet")


class SpacyTokenizer(TextTokenizer):
    def __init__(self) -> None:
        self._model: English = en_core_web_sm.load()

    def tokenizeText(self, text: str) -> List[SpacySentence]:
        """
        Uses a sentence tokenizer, and tokenize obtained sentences with Spacy tokenizer.
        Replace unnormal quotes.
        """
        text = re.sub(r'[`\x92\x91]', r"'", text)
        text = re.sub(r'[\x93\x94\x95\x96\x85\xE9]', r'"', text)
        text = re.sub(r'[\x80-\xFF]', r' ', text)
        text = re.sub(r"([:\s])\'(.+?)\'([\s\.])", r'\1"\2"\3', text)
        text = re.sub(r"\s+", r' ', text)
        text = text.strip()

        doc: Doc = self._model(text)
        tokenized = slist()
        sent: Span
        for sent in doc.sents:
            ss = SpacySentence(list(sent), sent.text)
            tokenized.append(ss)
        return tokenized
