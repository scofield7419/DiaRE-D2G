#!/usr/bin/env python
# coding:utf-8

from unittest import TestCase, main

from pyxtension.streams import stream

from tokenizers import SpacyTokenizer


class TestSpacyTokenizer(TestCase):
    def test_tokenize_text(self):
        tokenizer = SpacyTokenizer()
        sentences = tokenizer.tokenizeText("I am happy? All U.K. is happy after Brexit?")
        self.assertEqual(2, len(sentences))
        tagged_text = sentences[0].getTaggedText()
        jsoned = stream(tagged_text).map(lambda _: _.toJson()).toList()
        self.assertListEqual(
            [{'word': 'I', 'tag': 'PRP', 'stemmed': '-PRON-'}, {'word': 'am', 'tag': 'VBP', 'stemmed': 'be'},
             {'word': 'happy', 'tag': 'JJ', 'stemmed': 'happy'}, {'tag': 'SYM'}],
            jsoned)


if __name__ == '__main__':
    main()
