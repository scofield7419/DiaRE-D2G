#!/usr/bin/env python
# coding:utf-8

import unittest


class TestCase(unittest.TestCase):
    def testChildren(self):
        from . import dependencies

        hierarchy = dependencies.StanfordDependencyHierarchy()
        self.assertEqual(hierarchy.isa("agent", "arg"), True)

        self.assertEqual(hierarchy.isa("ref", "dep"), True)
        self.assertEqual(hierarchy.isa("dep", "dep"), False)

        self.assertEqual(hierarchy.isa("predet", "mod"), True)


if __name__ == '__main__':
    unittest.main()
