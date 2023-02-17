#!/usr/bin/env python
# coding:utf-8

import copy
from abc import ABCMeta, abstractmethod
from collections import defaultdict

from pyxtension.Json import Json
from pyxtension.streams import slist

from gramm_types import DepType, TagType, CDepConst

__author__ = 'ASU'


class CWordDependencies:
    def __init__(self, syntWord, aDependencies):
        """
        :type syntWord: nlplib.spinlib.SyntWordNode
        :type aDependencies: list[GrammDep]
        """
        assert isinstance(syntWord, SyntWordNode)
        assert isinstance(aDependencies, list)
        self._aGovDeps = slist([dp for dp in aDependencies if dp.gov.position == syntWord.position])
        self._aSlaveDeps = slist([dp for dp in aDependencies if dp.slave.position == syntWord.position])
        self._hDeps = {'gov': defaultdict(list), 'slave': defaultdict(list), }
        for dep, gov, slave in self._aGovDeps:
            self._hDeps['gov'][dep].append(slave)
        ###for
        for dep, gov, slave in self._aSlaveDeps:
            self._hDeps['slave'][dep].append(gov)

    def getDeps(self, sDep, bGov=True):
        """
        returns
        :rtype : slist[  SyntWordNode ]
        :type sDep: DepType
        :type bGov: bool
        """
        dtype = ('slave', 'gov')[bGov]
        if sDep in self._hDeps[dtype]:
            return self._hDeps[dtype][sDep]
        return None

    def getEqDeps(self, depType, bGov=True):
        """
        :type depType: DepType
        :type bGov: bool
        :rtype : slist[  SyntWordNode ] | None
        """
        dtype = ('slave', 'gov')[bGov]
        hDeps = self._hDeps[dtype]

        aRet = slist()
        for t in depType.equivalent_names:
            if t in hDeps:
                aRet.extend(hDeps[t])
        return aRet or None

    @property
    def govDeps(self):
        """
        :rtype : slist[GrammDep]
        """
        return self._aGovDeps

    @property
    def slaveDeps(self):
        """
        :rtype : slist[GrammDep]
        """
        return self._aSlaveDeps


class SyntToken(object):

    def __init__(self, text: str, tag: TagType, idx: int):
        assert isinstance(tag, TagType)
        self._tag = tag
        self._text = text
        self._idx = idx

    @property
    def tag(self) -> TagType:
        return self._tag

    @property
    def text(self) -> str:
        return self._text

    @property
    def idx(self):
        return self._idx

    def toJson(self):
        return Json({'tag': self.tag, 'text': self.text})


class SyntWordToken(SyntToken):

    def __init__(self, text: str, tag: TagType, idx: int, stemmed: str):
        super().__init__(text, tag, idx)
        self._stemmed = stemmed

    @property
    def stemmed(self):
        return self._stemmed


class SyntNode(object):
    @property
    def tag(self) -> TagType:
        return self._tag

    @tag.setter
    def tag(self, value: TagType):
        assert isinstance(value, TagType)
        self._tag = value

    level = int(0)
    _tree_index = int(-1)
    _tree = None
    children = property(lambda self: self._children)
    tree = property(lambda self: self._tree)
    hasParent = property(lambda self: self._parent is not None)
    index = property(lambda self: self._tree_index)

    def set_children(self, value):
        self._children = value

    def isDescendantOf(self, parent_node):
        if 0: isinstance(parent_node, PhraseNode)
        p = self.parent
        if 0:   isinstance(p, SyntNode)
        while p is not None and p.hasParent:
            if p.index == parent_node.index:
                return True
            p = p.parent
        ##while
        return False

    def set_tree(self, value):
        self._tree = value

    @property
    def isLeaf(self):
        return self._isLeaf

    @property
    def isPhrasal(self):
        return self._isPhrasal

    @property
    def isUnaryRewrite(self):
        return self._isUnaryRewrite

    @property
    def isPreTerminal(self):
        return self._isPreTerminal

    def __init__(self, tag, level, jNode, tree, _tree_index, parent=None):
        """
        :type tag: TagType
        :type level: int
        :type jNode: jpype._jclass.edu.stanford.nlp.trees.LabeledScoredTreeNode or __builtin__.NoneType
        :type tree: AbstractSyntacticTree or SpinnedSyntacticTree
        :type _tree_index: int
        :type parent: unknown or PhraseNode
        """
        assert isinstance(tag, TagType)
        self._tag = tag
        self.level = level

        if jNode is not None:
            self._isLeaf = bool(jNode.isLeaf())
            self._isPhrasal = bool(jNode.isPhrasal())
            self._isUnaryRewrite = bool(jNode.isUnaryRewrite())
            self._isPreTerminal = bool(jNode.isPreTerminal())
        else:
            self._isLeaf = None
            self._isPhrasal = None
            self._isUnaryRewrite = None
            self._isPreTerminal = None

        self._tree = tree
        self._children = []
        if 0:   isinstance(parent, PhraseNode)
        self._parent = parent
        self._tree_index = _tree_index

    @property
    def parent(self):
        """"""
        if 0:   isinstance(self._parent, PhraseNode)
        return self._parent

    def addChild(self, node, index=-1):
        """"""
        # self._children.insert(index, node)
        self.tree.addChild(self, node, index)

    def indexOf(self, node):
        """
        Returns int() - index in children's list of node
        """
        for i in range(len(self.children)):
            if self.children[i] == node:
                return i
        return -1

    def __repr__(self):
        return repr((self.tag, self.level, self.children))

    def __str__(self):
        # return str(self.__class__.__bases__[0].__repr__(self[:2]))
        return str(repr((self.tag, self.level)))

    def __hash__(self):
        return hash(id(self))

    def __deepcopy__(self, memo):
        """"""
        if id(self) in memo:
            return memo[id(self)]
        cp = copy.copy(self)
        memo[id(self)] = cp
        for k, v in self.__dict__.items():
            setattr(cp, k, copy.deepcopy(v, memo))
        return cp

    def __getitem__(self, item):
        raise NotImplementedError("Not implemented getitem of SyntNode")

    def toJson(self):
        return Json({'tag': self.tag})


class PhraseNode(SyntNode):
    isPhrasal = property(lambda self: True)

    def __init__(self, tag, level, jNode, tree, _tree_index, parent=None):
        SyntNode.__init__(self, tag, level, jNode, tree, _tree_index, parent)
        self.level = level
        self._tree = tree

    @property
    def children(self):
        """"""
        return self._children

    @staticmethod
    def tag_normalize(swn):
        """"""
        isinstance(swn, SyntWordNode)
        hTransforms = {'-LRB-': '(', '-RRB-': ')'}
        if swn.tag in hTransforms:
            return hTransforms[swn.tag]
        else:
            return swn.word
            ##if

    def toPhrase(self):
        """Returns the list of string words in the  DF parse of the tree starting from Node"""
        s = []
        q = [self]
        while len(q):
            node = q.pop()
            if isinstance(node, SyntWordNode):
                s.append(self.__class__.tag_normalize(node))
            else:
                q.extend(reversed(node.children))
                ##if
        ##while
        return s

    def toTaggedPhrase(self):
        """Returns the list of WordNodes in the  DF parse of the tree"""
        s = []
        q = [self]
        while len(q):
            node = q.pop()
            if isinstance(node, SyntWordNode):
                s.append(node)
            elif node.isPhrasal:
                q.extend(reversed(node.children))
            else:
                raise Exception("Unknown NODE type")
                ##if
        ##while
        return s

    def findPhraseByTag(self, tag='NP'):
        """Returns a list of nodes(subtrees) that are of type <tag>"""
        rl = []
        q = [self]
        while len(q):
            node = q.pop()
            isinstance(node, SyntNode)
            if isinstance(node, SyntWordNode):
                pass
            elif node.isPhrasal:
                if node.tag == tag:
                    rl.append(node)
                q.extend(reversed(node.children))
                ##if
        ##while
        return rl

    def findPhraseContainingWordTag(self, tag):
        """Returns a list of nodes(subtrees) that contains Word tagged <tag> in direct children"""
        assert isinstance(tag, TagType)
        rl = []
        q = [self]
        while len(q):
            node = q.pop()
            if isinstance(node, SyntWordNode):
                pass
            else:
                if len([x for x in node.children if isinstance(x, SyntWordNode) and x.tag == tag]):
                    rl.append(node)
                q.extend(reversed([n for n in node.children if n.isPhrasal]))
                ##if
        ##while
        return rl

    def deleteChild(self, child):
        """ Remove a child node if found """
        i = self.indexOf(child)
        if i != -1:
            del self._children[i]
            return True
        else:
            return False

    def replaceChild(self, child, newNode):
        """ Replace the child node with the given one """
        i = self.indexOf(child)
        if i != -1:
            self._children[i] = newNode
            return True
        else:
            return False

    def insertChild(self, child, index=0):
        """ Insert a child node at the given index """
        self._children.insert(index, child)


class SyntWordNode(SyntNode):
    @property
    def word(self):
        '''
        :return: str
        :rtype: str
        '''
        return self._word

    @word.setter
    def word(self, val):
        if val is None:
            print("None!")
        self._word = val

    _stemmed = str()
    """:type : str"""

    @property
    def stemmed(self):
        return self._stemmed

    _position = int()
    """:type : int"""

    def set_position(self, val):
        self._position = val

    @property
    def position(self):
        return self._position

    isLeaf = property(lambda self: True)
    """:type : bool"""

    def __init__(self, word, tag, stemmed, level, position, jNode, tree, _tree_index, parent=None):
        """ Constructor
        :type word: basestring
        :type tag: TagType
        :type stemmed: basestring
        :type level: int
        :type position: int
        :type jNode: jpype._jclass.edu.stanford.nlp.trees.LabeledScoredTreeNode or __builtin__.NoneType
        :type tree: AbstractSyntacticTree or SpinnedSyntacticTree
        :type _tree_index: int
        :type parent: PhraseNode or SpinablePhraseNode or unknown
        """
        assert isinstance(tag, TagType)
        assert isinstance(word, str)
        assert word is not None
        SyntNode.__init__(self, tag, level, jNode, tree, _tree_index, parent)
        self._word = word
        self._position = position
        self._stemmed = stemmed
        self._myDeps = None  # lazy evaluation
        isinstance(self._myDeps, CWordDependencies)

    def getMyFirstPhraseOfType(self, sPhraseTag):
        """"""
        p = self.parent
        if 0:   isinstance(p, SyntNode)
        while p is not None and p.hasParent:
            if p.tag == sPhraseTag:
                return p
            p = p.parent
        else:
            return None

    def getMyPhrase(self):
        """"""
        # parent = self.parent
        if 0:   isinstance(self.parent, PhraseNode)
        return self.parent

    def getFirstNegation(self):
        """ Return the negation SyntWordNode or None if it's not negated """
        neg_deps = self.myDeps.getDeps(CDepConst.NEG, True)
        if not isinstance(neg_deps, list):
            return None
        # Get just the first one if there are multiple, very unlikely
        return neg_deps[0]

    def getModifiers(self):
        """ Return the list of adverbial modifiers (advmod) """
        advmod_deps = self.myDeps.getDeps(CDepConst.ADVMOD, True)
        if isinstance(advmod_deps, list):
            return advmod_deps

        return []

    def getFirstDeterminant(self):
        """ Return the determinant of the word or None """
        determinant = self.myDeps.getDeps(CDepConst.DET, True)
        if isinstance(determinant, list):
            # Get just the first one if there are multiple, very unlikely
            return determinant[0]

        return None

    def __repr__(self):
        return repr((self.word, self.tag, self.stemmed, self.level))

    def __str__(self):
        return self.toJson().toString()

    def toJson(self):
        return Json({'word': self.word, 'tag': self.tag, 'stemmed': self.stemmed})

    def __hash__(self):
        return hash(repr(self))

    def __cmp__(self, other):
        return cmp(hash(self), hash(other))

    @property
    def myDeps(self):
        """
        :rtype : CWordDependencies
        """
        if self._myDeps is None:
            aDeps = self._tree.sent_dep_wNodes
            self._myDeps = CWordDependencies(self, aDeps)
        return self._myDeps


class GrammDep(tuple):
    def __new__(cls, t):
        """
        :type t: ( DepType, SyntWordNode, SyntWordNode )
        """
        assert len(t) == 3
        assert isinstance(t[0], str)
        r = t[0]
        if not isinstance(t[0], DepType):
            r = DepType.fromString(t[0])
        assert isinstance(r, DepType)
        assert isinstance(t[1], SyntWordNode)
        assert isinstance(t[2], SyntWordNode)
        return super(GrammDep, cls).__new__(cls, (r, t[1], t[2]))

    @property
    def type(self):
        """
        :rtype : DepType
        """
        return self[0]

    @property
    def gov(self):
        """
        :rtype : SyntWordNode
        """
        return self[1]

    @property
    def slave(self):
        """
        :rtype : SyntWordNode
        """
        return self[2]

    @property
    def equivalent_dep_names(self):
        """

        :type self: (DepType, SyntWordNode, SyntWordNode )
        """
        return self[0].equivalent_names

    @property
    def equivalent_dep_types(self):
        """

        :type self: (DepType, SyntWordNode, SyntWordNode )
        """
        return self[0].equivalent_types


class AbstractSyntacticTree(object):
    root = property(lambda self: self._root)
    """:type : PhraseNode"""

    aWordNodes = property(lambda self: self._aWordNodes)
    """:type : slist[SyntWordNode]"""

    sent_dep_wNodes = property(lambda self: self._sent_dep_wNodes)
    """:type : slist[GrammDep]"""

    def _constructWordNode(self, *args, **kwargs):
        """
        :rtype : SyntWordNode
        """
        # Overwridable method, do not change in classmethod
        return SyntWordNode(*args, **kwargs)

    def _constructPhraseNode(self, *args, **kwargs):
        """
        :rtype : PhraseNode
        """
        # Overwridable method, do not change in classmethod
        return PhraseNode(*args, **kwargs)

    def __init__(self):
        """
        :type jTree: jpype._jclass.edu.stanford.nlp.trees.LabeledScoredTreeNode
        :type sfp: StanfordParser
        """
        self._aNodes = slist()
        self._sent_dep_wNodes = slist()

    def addChild(self, node, child, index=-1):
        """
        :type node: PhraseNode
        """
        assert isinstance(node, SyntNode)
        assert isinstance(child, SyntNode)
        idx = len(self._aNodes)
        child.set_tree_index = idx
        self._aNodes.append(child)
        if index >= 0:
            node.children.insert(index, child)
        else:
            node.children.append(child)
        return idx

    # ----------------------------------------------------------------------
    def __deepcopy__(self, memo):
        # ToDo: document and cover with UnitTest
        if id(self) in memo:
            return memo[id(self)]
        cp = copy.copy(self)
        memo[id(self)] = cp
        for a in ('_aNodes', '_aWordNodes', '_root', '_sent_dep_wNodes'):
            setattr(cp, a, copy.deepcopy(getattr(self, a), memo.copy()))
        ##for
        return cp

    def __str__(self):
        s = ''
        q = [self.root]
        while len(q):
            node = q.pop()
            if isinstance(node, SyntWordNode):
                s += '\t' * node.level + str(node) + '\n'
            else:
                s += '\t' * node.level + str(node) + '\n'
                q.extend(reversed(node.children))
        return s

    def toJson(self):
        j = Json()
        self.root.toJson()
        return j


class AbstractParser(object, metaclass=ABCMeta):
    """Base class for Parser classes"""

    @abstractmethod
    def parse(self, text):
        pass

    def __call__(self, *args, **kwrds):
        return self.parse(*args, **kwrds)


class AbstractParsedSentence(metaclass=ABCMeta):
    @abstractmethod
    def __str__(self):
        """
        When is called str(object) - this method is called
        :rtype : str
        """
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @staticmethod
    def _toWord(swn):
        '''
        :param swn:
        :type swn: SyntWordNode
        :return: str
        :rtype: str
        '''
        assert isinstance(swn, SyntWordNode)
        return swn.word

    @abstractmethod
    def getTaggedText(self):
        """
        :rtype : slist[ SyntWordNode ]
        """
        pass

    @abstractmethod
    def getBestTree(self):
        """
        :return: Returns the list of best parsed syntactical trees of the paper
        :rtype: AbstractSyntacticTree
        """
        pass

    @abstractmethod
    def getDependencies(self):
        """
        Returns the list of dependencies for every sentence of the text
        :rtype : list[ GrammDep ]
        """
        pass
