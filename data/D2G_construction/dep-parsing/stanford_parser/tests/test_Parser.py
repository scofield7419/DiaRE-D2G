from os.path import join, normpath
from unittest import TestCase, main

import stanford_parser.parser as parser
from tokenizers import NLTKTokenizer

__author__ = 'ASU'


class TestParser(TestCase):
    @classmethod
    def setUpClass(cls):
        jarPathName = normpath(join(parser.MODULE_PATH, "./CoreNLP"))
        cls._parser = parser.Parser(jarPathName, 'edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz')
    
    def test_nominalParse(self):
        text = "My dog likes sausages"
        tree = self._parser.parse(text)
        self.assertEqual(tree.toString(), '(ROOT (S (NP (PRP$ My) (NN dog)) (VP (VBZ likes) (NP (NNS sausages)))))')
    
    def test_ParseToDependencies(self):
        text = "Pick up the tire pallet next to the truck. He smelled that the price will go up."
        sentences = parser.StanfordParser(NLTKTokenizer(), self._parser).parse(text)
        
        dependencies = sentences[0].dependencies
        
        tupleResult = [(rel, gov.word, dep.word) for rel, gov, dep in dependencies]
        
        self.assertEqual(tupleResult,
                         [('root', '.', 'Pick'),
                          ('prt', 'Pick', 'up'),
                          ('det', 'pallet', 'the'),
                          ('nn', 'pallet', 'tire'),
                          ('dobj', 'Pick', 'pallet'),
                          ('det', 'truck', 'the'),
                          ('prep_next_to', 'pallet', 'truck')])


if __name__ == '__main__':
    main()
