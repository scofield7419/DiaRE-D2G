#!/usr/bin/env python
# coding:utf-8

# Stanford CoreNLP connector compatible with stanford-parser.jar version up to 2.0.5, build date 2013-04-05
# http://nlp.stanford.edu/software/stanford-corenlp-full-2013-04-04.zip

import logging
import os
import platform
import re
import sys

from pyxtension.Json import Json
from pyxtension.streams import slist

from gramm_types import DepType, STANFORD_WORD_TAGS, CTags, STEM_EXCEPTIONS
from sentence_types import AbstractSyntacticTree, GrammDep, AbstractParser, SyntWordNode, AbstractParsedSentence
from tokenizers import creReplaceNLs, TextTokenizer

_JAVA = platform.system() == "Java"


class ParserError(Exception):
    def __init__(self, *args, **margs):
        Exception.__init__(self, *args, **margs)


DEFAULT_MAX_SENTENCE_LENGTH = 125
MODULE_PATH = os.path.dirname(__file__)


def fReduceSpaces(x): return creReplaceNLs.sub(r' ', x)


class JClasses:
    MAIN_JAR_NAME = "stanford-parser.jar"
    MODELS_JAR_NAME = "stanford-parser-2.0.5-models.jar"
    
    def __init__(self):
        self._initialized = False
        self.JMaxentTagger = None
        self.JLexicalizedParser = None
        self.JMorphology = None
        self.JEnglishGrammaticalStructure = None
        self.JStringReader = None
        self.JDocumentProcessor = None
        self.JString = None
    
    def init(self, home, jvm_memory="20G"):
        if self._initialized:
            return
        
        if _JAVA:
            sys.path.append(os.path.join(home, self.MAIN_JAR_NAME))
            from edu.stanford.nlp.tagger.maxent import MaxentTagger as JMaxentTagger
            from edu.stanford.nlp.parser.lexparser import LexicalizedParser as JLexicalizedParser
            from edu.stanford.nlp.process import Morphology as JMorphology
            from edu.stanford.nlp.trees import EnglishGrammaticalStructure as JEnglishGrammaticalStructure
            from edu.stanford.nlp.process import DocumentPreprocessor as JDocumentProcessor
            from java.io import StringReader
            from java.lang import String
            self.JMaxentTagger = JMaxentTagger
            self.JLexicalizedParser = JLexicalizedParser
            self.JMorphology = JMorphology
            self.JEnglishGrammaticalStructure = JEnglishGrammaticalStructure
            self.JStringReader = StringReader
            self.JDocumentProcessor = JDocumentProcessor
            self.JString = String
        else:
            import jpype
            defJvmPath = jpype.getDefaultJVMPath()
            # try a fallback due to some Java installers bug
            if not os.path.exists(defJvmPath):
                defJvmPath = defJvmPath.replace('client', 'server')
            # no fallback worked
            if not os.path.exists(defJvmPath):
                raise Exception('Could not find a JVM')
            classPaths = ';'.join((os.path.normpath(os.path.join(home, self.MAIN_JAR_NAME)),
                                   os.path.normpath(os.path.join(home, self.MODELS_JAR_NAME))))
            jpype.startJVM(defJvmPath,
                           '-ea',
                           '-Xmx' + jvm_memory,
                           '-Djava.class.path={}'.format(classPaths))
            self.JLexicalizedParser = jpype.JClass("edu.stanford.nlp.parser.lexparser.LexicalizedParser")
            self.JMorphology = jpype.JClass("edu.stanford.nlp.process.Morphology")
            self.JEnglishGrammaticalStructure = jpype.JClass("edu.stanford.nlp.trees.EnglishGrammaticalStructure")
            self.JStringReader = jpype.JClass("java.io.StringReader")
            self.JDocumentProcessor = jpype.JClass("edu.stanford.nlp.process.DocumentPreprocessor")
            self.JString = jpype.JClass("java.lang.String")
            # self.JMaxentTagger = jpype.JClass("edu.stanford.nlp.tagger.maxent.MaxentTagger")
        
        self._initialized = True


_JClasses = JClasses()


class MaxEntTagger:
    """
    Not compatible with CoreNLP 2.x
    """
    TAGGER_MODEL_FILE_NAME = "english-bidirectional-distsim.tagger"
    
    def __init__(self, stanford_home):
        tagger_model_path = os.path.join(stanford_home, self.TAGGER_MODEL_FILE_NAME)
        _JClasses.init(stanford_home)
        try:
            self.tagger = _JClasses.JMaxentTagger(tagger_model_path)
        except Exception as e:
            logging.exception("Can not instantiate MaxentTagger with %s" % str(tagger_model_path))
            raise e
    
    # ----------------------------------------------------------------------
    def tagTokenizedString(self, toTag):
        """
        Tags the tokenized input string and returns the tagged version.
        This method requires the input to already be tokenized.
        The tagger wants input that is whitespace separated tokens, tokenized
        according to the conventions of the training data. (For instance,
        for the Penn Treebank, punctuation marks and possessive "'toTag" should
        be separated from words.)

        :param toTag: The untagged input String
        :type toTag: basestring
        :return: The same string with tags inserted in the form word/tag
        :rtype: basestring
        """
        return self.tagger.tagTokenizedString(toTag)
    
    def tagString(self, s):
        """
        Tags the input string and returns the tagged version.
        This method tokenizes the input into words in perhaps multiple sentences
        and then tags those sentences.  The default (PTB English)
        tokenizer is used.

        :param s: The untagged input String
        :type s: basestring
        :return: A String of sentences with tags inserted in the form word/tag
        :rtype: basestring
        """
        return self.tagger.tagString(s)


class Parser:
    def __init__(self, stanford_home, pcfg_model_fname='englishPCFG.ser.gz'):
        self.pcfg_model_fname = os.path.join(stanford_home, pcfg_model_fname)
        _JClasses.init(stanford_home)
        try:
            self.parser = _JClasses.JLexicalizedParser.loadModel(pcfg_model_fname, ["-retainTmpSubcategories"])
        except Exception:
            print(("Can not instantiate LexicalizedParser with %s" % str(pcfg_model_fname)))
            raise
        self.parser.setOptionFlags(["-retainTmpSubcategories"])
        self.morphology = _JClasses.JMorphology
        self.englishGrammaticalStructure = _JClasses.JEnglishGrammaticalStructure
    
    @staticmethod
    def segment_text(text):
        """
        Segment raw text into sentences using Stanford DocumentProcessor
        :type text: str|unicode
        :rtype: slist[basestring]
        """
        text = text.strip()
        reader = _JClasses.JStringReader(_JClasses.JString(text))
        dp = _JClasses.JDocumentProcessor(reader)
        iterator = dp.iterator()
        sentences = slist()
        while iterator.hasNext():
            sentence_array = next(iterator)
            tokens = []
            for idx in range(sentence_array.size()):
                token = sentence_array[idx].toString()
                
                old_token = ""
                while old_token != token:
                    old_token = token
                    token = token.replace("\\", "")
                    token = token.replace("\\/", "/")
                tokens.append(token)
            sentences.append(' '.join(tokens))
        return sentences
    
    def parse(self, sentence):
        """
        Parses the sentence string, returning the tokens, and the parse tree as a tuple.
        tokens, tree = parser.parse(sentence)
        :param sentence:
        :type sentence: basestring
        :rtype : jpype._jclass.edu.stanford.nlp.trees.LabeledScoredTreeNode
        """
        return self.parser.parse(sentence)


class StanfordParsedSentence(AbstractParsedSentence):
    def __init__(self, text, _wNodes, _tree, _dependency_wNodes):
        """
        :type text: unicode
        :param text: the string containing the text of the paper if text is an Integer,
            representing a text ID, than the constructor loads from the disk the serialized object of the paper
            with ID from first argument, ie 'text', and all further arguments are ignored
        :type _wNodes: slist
        :type _tree: AbstractSyntacticTree | SpinnedSyntacticTree
        :type _dependency_wNodes: list[GrammDep]
        """
        self._text = text
        self._tree = _tree
        self._wNodes = _wNodes
        self._dependency_wNodes = _dependency_wNodes
    
    @property
    def best_tree(self):
        return self.getBestTree()
    
    @property
    def tagged_text(self):
        """
        :rtype : slist[ SyntWordNode ]
        """
        return self._wNodes
    
    @property
    def dependencies(self):
        """
        Returns the list of dependencies for every sentence of the text
        :rtype : list[ GrammDep ]
        """
        return self.getDependencies()
    
    def getDependencies(self):
        """
        :rtype : list[ GrammDep ]
        """
        return self._dependency_wNodes
    
    def getTaggedText(self):
        return self._wNodes
    
    def getBestTree(self):
        """
        :return: Returns the list of best parsed syntactical trees of the paper
        :rtype: AbstractSyntacticTree
        """
        return self._tree
    
    def __str__(self):
        """
        When is called str(object) - this method is called
        :rtype : str
        """
        return self._text
    
    def __repr__(self):
        return str({"instanceType": "StanfordParsedSentence", "val": self.__str__()})
    
    @property
    def words(self):
        """
        Returns stream of strings representing only words from the text
        :return: stream[str]
        :rtype: stream[ str ]
        """
        
        return self.tagged_text.filter(lambda swn: swn.tag in STANFORD_WORD_TAGS).map(StanfordParsedSentence._toWord)
    
    def toJson(self):
        """
        :return:
        :rtype: Json
        """
        j = Json()
        j.text = self._text
        j.tree = self._tree.toJson()
        j.dependencies = self.dependencies.toJson()


class StanfordSyntacticTree(AbstractSyntacticTree):
    def __init__(self, jTree=None, sfp=None):
        """
        :type jTree: jpype._jclass.edu.stanford.nlp.trees.LabeledScoredTreeNode
        :type sfp: StanfordParser
        """
        super(StanfordSyntacticTree, self).__init__()
        if jTree is None:
            return
        isinstance(sfp, StanfordParser)
        (self._root, self._aWordNodes) = self.iterativeParse_jTree(jTree, sfp)
        egr = sfp.CEnglishGrammaticalStructure(jTree)
        for dependency in egr.typedDependenciesCCprocessed():
            gi = dependency.gov().index() - 1
            di = dependency.dep().index() - 1
            self._sent_dep_wNodes.append(
                GrammDep((DepType.fromString(str(dependency.reln())), self.aWordNodes[gi], self.aWordNodes[di])))
    
    def iterativeParse_jTree(self, jTree, sfp):
        """
        Parses the Tree object from JVM into Python VM
        Returns tuple (tree, dictionary of WordPositions)
        :type sfp: StanfordParser
        :type jTree: jpype._jclass.edu.stanford.nlp.trees.LabeledScoredTreeNode
        :rtype :  PhraseNode, slist[SyntWordNode]
        """
        level = 0
        tree_index = 0
        ptree = self._constructPhraseNode(CTags.fromString(str(jTree.value())), level, jTree, self, tree_index)
        """:type : CPhraseNode"""
        level += 1  # only root is level 0
        tree_index += 1
        
        q = []
        for c in jTree.children():
            q.append((c, ptree, level))
        q = list(reversed(q))
        
        hWordsPositions = {}
        while len(q):
            node = q.pop()
            wh_add = node[1]
            level = node[2]
            node = node[0]
            if node.isLeaf():
                raise ValueError("Malformed syntactic tree: unexpected leaf")
            elif node.isPreTerminal():
                tag = str(node.value())
                w_label = node.children()[0].label()
                v = w_label.value()
                word = re.sub(r'[\x80-\xFF]', r"_Unknown_char_", v)
                wtag = sfp.stem(word, tag)
                _wrd = wtag.word()
                if _wrd is None:
                    raise Exception("wtag returned by stemmer returns NULL word() for word %s and tag %s: %s %s" % (
                        word, tag, str(wtag), str(wtag.__class__)))
                
                if wtag.word().lower() in STEM_EXCEPTIONS:
                    stemmed = STEM_EXCEPTIONS[wtag.word().lower()]
                else:
                    stemmed = wtag.word().lower()
                
                w_pos = (int(w_label.beginPosition()), int(w_label.endPosition()))
                nt = self._constructWordNode(word, CTags.fromString(tag), stemmed, level, w_pos, node, self, tree_index,
                                             wh_add)
                tree_index += 1
                hWordsPositions[int(w_label.beginPosition())] = nt
                wh_add.addChild(nt)
            else:
                nt = self._constructPhraseNode(CTags.fromString(str(node.value())), level, node, self, tree_index,
                                               wh_add)
                tree_index += 1
                wh_add.addChild(nt)
                level += 1
                chldren = []
                for c in node.children():
                    chldren.append((c, nt, level))
                q.extend(reversed(chldren))
        wNodes = slist(hWordsPositions[p] for p in sorted(hWordsPositions.keys()))
        """:type : list[SyntWordNode]"""
        for i in range(len(wNodes)):
            wNodes[i].set_position(i)
        
        return ptree, wNodes


class StanfordParser(AbstractParser):
    """
    The class for parsing papers using Stanford Parser
    """
    
    # ----------------------------------------------------------------------
    def __init__(self, textTokenizer, parser_obj, maxSentenceLength=DEFAULT_MAX_SENTENCE_LENGTH):
        """
        :type textTokenizer: TextTokenizer
        :type parser_obj: Parser
        :type maxSentenceLength: int
        """
        self._MAX_SENTENCE_LENGTH = maxSentenceLength
        assert isinstance(textTokenizer, TextTokenizer)
        super(StanfordParser, self).__init__()
        assert isinstance(parser_obj, Parser), str(parser_obj.__class__)
        self._parser_obj = parser_obj
        self._tokenizer = textTokenizer
        self._CEnglishGrammaticalStructure = None
        self._morphology = None
    
    @property
    def _parser(self):
        return self._parser_obj
    
    @property
    def CEnglishGrammaticalStructure(self):
        if self._CEnglishGrammaticalStructure is None:
            self._CEnglishGrammaticalStructure = self._parser.englishGrammaticalStructure
        return self._CEnglishGrammaticalStructure
    
    @property
    def stem(self):
        if self._morphology is None:
            self._morphology = self._parser.morphology
            # callable
            assert callable(self._morphology.stemStaticSynchronized)
        return self._morphology.stemStaticSynchronized
    
    def parse(self, text):
        """
        :type text: str
        :rtype: list[StanfordParsedSentence]

        Parses the text, with ID 'tid' using stanford parser
        Returns a list of StanfordParsedSentence() objects
        """
        try:
            sentences = self._parser.segment_text(text)
        except Exception:
            logging.error("Stanford Sentence Segmentation failed. Used nltk instead.")
            sentences = self._tokenizer.tokenizeText(text)
        
        aSentences = []
        for sent in sentences:
            sent = fReduceSpaces(sent)
            sps = self.parse_sentence(sent)
            if sent:
                aSentences.append(sps)
        return aSentences
    
    def parse_sentence(self, sent):
        """
        :type sent: basestring
        :rtype StanfordParsedSentence:
        """
        sent = fReduceSpaces(sent)
        wrds = self._tokenizer.tokenizeText(sent)
        if len(wrds) > self._MAX_SENTENCE_LENGTH:
            logging.error("The sentence has %d words: %s" % (len(wrds), str(sent)))
            return None
        else:
            tree = self._parser.parse(sent)
            ptree = StanfordSyntacticTree(tree, self)
            assert isinstance(ptree, AbstractSyntacticTree)
            return StanfordParsedSentence(sent, ptree.aWordNodes, ptree, ptree.sent_dep_wNodes)
