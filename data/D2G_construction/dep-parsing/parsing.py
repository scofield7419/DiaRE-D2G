from nltk.parse.stanford import StanfordDependencyParser
from pprint import pprint
import os
import spacy
import argparse
import numpy as np
import json

java_path = r"xxx\Java\jdk-17.0.5\bin\java.exe"
os.environ['JAVAHOME'] = java_path

# https://stanfordnlp.github.io/CoreNLP/download.html

m_parser = '.\stanford-parser-4.0.0\stanford-parser.jar'
m_model = '.\stanford-parser-4.0.0\stanford-parser-4.0.0-models.jar'

dependency_parser = StanfordDependencyParser(path_to_jar=m_parser, path_to_models_jar=m_model)

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', type=str, default="data")
parser.add_argument('--out_path', type=str, default="prepro_data")

args = parser.parse_args()
in_path = args.in_path
out_path = args.out_path

train_file_name = os.path.join(in_path, 'train.json')
dev_file_name = os.path.join(in_path, 'dev.json')
test_file_name = os.path.join(in_path, 'test.json')


def Parsing(data_file_name, suffix=''):
    ori_data = json.load(open(data_file_name))
    depdata = []
    for i in range(len(ori_data)):
        dep_utts = []
        for sent in ori_data[i][0]:
            result = dependency_parser.raw_parse(sent[11:])
            dep_tree = [list(parse.triples()) for parse in result]
            dep_utts.append('|'.join(list(dep_tree)))
        depdata.append(dep_utts)

    # saving
    print("Saving parsed trees")
    json.dump(depdata, open(os.path.join(out_path, suffix + '.json'), "w"))


print("=========================start to parse the training instances=========================")
Parsing(train_file_name, suffix='_train')
print("=========================start to parse the dev instances=========================")
Parsing(dev_file_name, suffix='_dev')
print("=========================start to parse the test instances=========================")
Parsing(test_file_name, suffix='_test')
