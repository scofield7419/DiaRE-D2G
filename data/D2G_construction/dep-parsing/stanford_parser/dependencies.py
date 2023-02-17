#!/usr/bin/env python
# coding:utf-8

stanford_dependency_hierarchy = {"dep":
                                     {"aux":       {"auxpass": {},
                                                    "cop":     {}},
                                      "arg":       {"agent": {},
                                                    "comp":  {"acomp": {},
                                                              "attr":  {},
                                                              "ccomp": {},
                                                              "xcomp": {},
                                                              "compl": {},
                                                              "obj":   {"dobj": {},
                                                                        "iobj": {},
                                                                        "pobj": {}},
                                                              "mark":  {},
                                                              "rel":   {}},
                                                    "subj":  {"nsubj": {"nsubjpass": {}},
                                                              "csubj": {}}},
                                      "cc":        {},
                                      "conj":      {},
                                      "expl":      {},
                                      "mod":       {"abbrev":     {},
                                                    "amod":       {},
                                                    "appos":      {},
                                                    "advcl":      {},
                                                    "purpcl":     {},
                                                    "det":        {},
                                                    "predet":     {},
                                                    "preconj":    {},
                                                    "infmod":     {},
                                                    "partmod":    {},
                                                    "advmod":     {"neg": {}},
                                                    "rcmod":      {},
                                                    "quantmod":   {},
                                                    "tmod":       {},
                                                    "measure":    {},
                                                    "nn":         {},
                                                    "num":        {},
                                                    "number":     {},
                                                    "prep":       {},
                                                    "poss":       {},
                                                    "possessive": {},
                                                    "prt":        {}},
                                      "parataxis": {},
                                      "punct":     {},
                                      "ref":       {},
                                      "sdep":      {"xsubj": {}}
                                      }
                                 }


class StanfordDependencyHierarchy:
    """
    Class that encodes the types of dependencies.
    """
    
    def __init__(self, hierarchy=None):
        """
        :type hierarchy: dict[str, dict|str]
        """
        if not hierarchy:
            hierarchy = stanford_dependency_hierarchy
        
        self.hierarchy = hierarchy
        self.flatMap = {}
        self.parentToChildren = {}
        activeSet = [self.hierarchy]
        
        while len(activeSet) != 0:
            newActiveSet = []
            for item in activeSet:
                for key, mapValue in list(item.items()):
                    self.flatMap[key] = mapValue
                    self.parentToChildren[key] = sorted(list(mapValue.keys()))
                    newActiveSet.append(mapValue)
            
            activeSet = newActiveSet
        
        self.ancestorToDescendents = {}
        
        for key, mapValue in list(self.flatMap.items()):
            descendents = []
            
            activeSet = [mapValue]
            while len(activeSet) != 0:
                newActiveSet = []
                for item in activeSet:
                    for childKey, val in list(item.items()):
                        newActiveSet.extend(list(val.values()))
                        descendents.append(childKey)
                activeSet = newActiveSet
            
            self.ancestorToDescendents[key] = sorted(descendents)
    
    def isa(self, relation, ancestor):
        return relation in self.ancestorToDescendents[ancestor]
