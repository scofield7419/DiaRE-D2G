import numpy as np
import os
import json
import argparse
import spacy
import networkx as nx
import copy
from models.bert import Bert
from pytorch_transformers import *

bert = Bert(BertModel, 'bert-base-uncased')

nlp = spacy.load("en_core_web_sm")

PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'

MAX_SENT_LEN = 200
MAX_NODE_NUM = 200
MAX_ENTITY_NUM = 100
MAX_SENT_NUM = 50
MAX_NODE_PER_SENT = 40

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', type=str, default="data")
parser.add_argument('--out_path', type=str, default="prepro_data")

args = parser.parse_args()
in_path = args.in_path
out_path = args.out_path
case_sensitive = False

char_limit = 16
train_file_name = os.path.join(in_path, 'train.json')
dev_file_name = os.path.join(in_path, 'dev.json')
test_file_name = os.path.join(in_path, 'test.json')

rel2id = json.load(open(os.path.join(in_path, 'rel2id.json'), "r"))
id2rel = {v: u for u, v in rel2id.items()}
json.dump(id2rel, open(os.path.join(in_path, 'id2rel.json'), "w"))
fact_in_train = set([])
fact_in_dev_train = set([])

unk_number = 0


# word_index_file_name = os.path.join(out_path,'vocab.pkl')

def GetVertexSet(sents, entities):
    e_data = []
    for (entity, e_type) in entities:
        ents = entity.split(' ')
        e_len = len(ents)
        data = []
        for s_id, sent in enumerate(sents):
            sent_lower = [word.lower() for word in sent]
            if ents[0].lower() in sent_lower:
                pos_1 = sent_lower.index(ents[0].lower())
                flag = True
                for e_i, pos in enumerate(range(pos_1, pos_1 + e_len)):
                    if ents[e_i].lower() in sent_lower[pos] and len(ents[e_i]) == len(sent[pos]):
                        pass
                    else:
                        flag = False
                        break
                if flag:
                    data.append({
                        "name": entity,
                        "mention": entity,
                        "pos": [pos_1, pos_1 + e_len],
                        "sent_id": s_id,
                        "type": e_type
                    })
        if len(data) == 0:
            for s_id, sent in enumerate(sents):
                pos_1 = -1
                pos_2 = -1
                for ti, token in enumerate(sent):
                    if entity.lower().startswith((token.lower())) and pos_1 == -1:
                        pos_1 = ti
                    if entity.lower().endswith((token.lower())) and pos_1 != -1:
                        pos_2 = ti
                if pos_1 != -1 and pos_2 != -1 and pos_2 - pos_1 > 0:
                    for ind in range(pos_1 + 1, pos_2 + 1):
                        try:
                            if sent[ind].lower() in entity.lower():
                                data.append({
                                    "name": entity,
                                    "mention": entity,
                                    "pos": [pos_1, pos_2 + 1],
                                    "sent_id": s_id,
                                    "type": e_type
                                })
                                break
                        except:
                            print('')
                elif pos_1 != -1 and pos_2 != -1:
                    data.append({
                        "name": entity,
                        "mention": entity,
                        "pos": [pos_1, pos_2 + 1],
                        "sent_id": s_id,
                        "type": e_type
                    })

        bianxing = ['s', 'es', 'ing', 'ed', 'ers', '.']
        bianxing_n = 0
        while len(data) == 0 and bianxing_n < len(bianxing):
            entity_1 = entity + bianxing[bianxing_n]
            ents_1 = entity_1.split(' ')
            for s_id, sent in enumerate(sents):
                sent_lower = [word.lower() for word in sent]
                if ents_1[0].lower() in sent_lower:
                    pos_1 = sent_lower.index(ents_1[0].lower())
                    flag = True
                    for e_i, pos in enumerate(range(pos_1, pos_1 + e_len)):
                        if ents_1[e_i].lower() in sent_lower[pos] and len(ents_1[e_i]) == len(sent[pos]):
                            pass
                        else:
                            flag = False
                            break
                    if flag:
                        data.append({
                            "name": entity,
                            "mention": entity_1,
                            "pos": [pos_1, pos_1 + e_len],
                            "sent_id": s_id,
                            "type": e_type
                        })
            if len(data) == 0:
                for s_id, sent in enumerate(sents):
                    pos_1 = -1
                    pos_2 = -1
                    for ti, token in enumerate(sent):
                        if entity_1.lower().startswith((token.lower())) and pos_1 == -1:
                            pos_1 = ti
                        if entity_1.lower().endswith((token.lower())) and pos_1 != -1:
                            pos_2 = ti
                    if pos_1 != -1 and pos_2 != -1 and pos_2 - pos_1 > 0:
                        for ind in range(pos_1 + 1, pos_2 + 1):
                            try:
                                if sent[ind].lower() in entity_1.lower():
                                    data.append({
                                        "name": entity,
                                        "mention": entity_1,
                                        "pos": [pos_1, pos_2 + 1],
                                        "sent_id": s_id,
                                        "type": e_type
                                    })
                                    break
                            except:
                                print('')
                    elif pos_1 != -1 and pos_2 != -1:
                        data.append({
                            "name": entity,
                            "mention": entity_1,
                            "pos": [pos_1, pos_2 + 1],
                            "sent_id": s_id,
                            "type": e_type
                        })
            bianxing_n += 1
        if len(data) == 0 and 'director' in entity and len(sents) == 9:
            data.append({
                "name": entity,
                "mention": ' '.join(sents[7][28:29]),
                "pos": [28, 29],
                "sent_id": 7,
                "type": e_type
            })
        if len(data) == 0 and 'Dr.' in entity and len(sents) == 12:
            data.append({
                "name": entity,
                "mention": ' '.join(sents[3][13:14]),
                "pos": [13, 14],
                "sent_id": 3,
                "type": e_type
            })
        if len(data) == 0 and 'big spender' in entity and len(sents) == 7:
            data.append({
                "name": entity,
                "mention": ' '.join(sents[0][5:6]),
                "pos": [5, 6],
                "sent_id": 0,
                "type": e_type
            })
            data.append({
                "name": entity,
                "mention": ' '.join(sents[2][4:5]),
                "pos": [4, 5],
                "sent_id": 2,
                "type": e_type
            })
        if len(data) == 0 and 'Don' in entity and len(sents) == 16:
            data.append({
                "name": entity,
                "mention": ' '.join(sents[8][3:5]),
                "pos": [3, 5],
                "sent_id": 8,
                "type": e_type
            })
        if len(data) == 0 and 'man' in entity and len(sents) == 28:
            data.append({
                "name": entity,
                "mention": ' '.join(sents[3][4:5]),
                "pos": [4, 5],
                "sent_id": 3,
                "type": e_type
            })
            data.append({
                "name": entity,
                "mention": ' '.join(sents[5][15:16]),
                "pos": [15, 16],
                "sent_id": 5,
                "type": e_type
            })
            data.append({
                "name": entity,
                "mention": ' '.join(sents[10][9:10]),
                "pos": [9, 10],
                "sent_id": 10,
                "type": e_type
            })
        try:
            assert len(data) != 0
        except:
            print('')

        e_data.append(data)
    new_e_data = copy.deepcopy(e_data)
    for i, ent in enumerate(e_data):
        name = ent[0]['name']
        add_sent = []
        if 'Speaker' in name:
            for ment in ent:
                sent_id = ment['sent_id']
                sent = sents[sent_id]
                if 'I' in sent:
                    item = {
                        'name': name,
                        'mention': 'I',
                        'pos': [sent.index('I'), sent.index('I') + 1],
                        'sent_id': sent_id,
                        'type': ment['type']
                    }
                    new_e_data[i].append(item)

                if sent_id - 1 >= 0:
                    lower_sent = [sen.lower() for sen in sents[sent_id - 1]]
                    prons = ['you']
                    for pron in prons:
                        if pron in lower_sent and pron + str(sent_id - 1) not in add_sent:
                            item = {
                                'name': name,
                                'mention': pron,
                                'pos': [lower_sent.index(pron), lower_sent.index(pron) + 1],
                                'sent_id': sent_id - 1,
                                'type': ment['type']
                            }
                            new_e_data[i].append(item)
                            add_sent.append(pron + str(sent_id - 1))
                if sent_id + 1 < len(sents):
                    lower_sent = [sen.lower() for sen in sents[sent_id + 1]]

                    prons = ['you']
                    for pron in prons:
                        if pron in lower_sent and pron + str(sent_id + 1) not in add_sent:
                            item = {
                                'name': name,
                                'mention': pron,
                                'pos': [lower_sent.index(pron), lower_sent.index(pron) + 1],
                                'sent_id': sent_id + 1,
                                'type': ment['type']
                            }
                            new_e_data[i].append(item)
                            add_sent.append(pron + str(sent_id + 1))

    return new_e_data


def GetNodePosition(data, node_position, node_position_sent, node_sent_num, entity_position, Ls):
    """
    :param data: input
    :param node_position: mention node position in a document (flatten)
    :param node_position_sent: node position in each sentence of a document
    :param node_sent_num: number of nodes in each sentence
    :param entity_position:
    :param Ls: the start position of each sentence in document
    :return:
    """
    nodes = [[] for _ in range(len(data[0]))]
    nodes_sent = [[] for _ in range(len(data[0]))]

    for ns_no, ns in enumerate(data[1]):
        for n in ns:
            sent_id = int(n['sent_id'])
            doc_pos_s = n['pos'][0] + Ls[sent_id]
            doc_pos_e = n['pos'][1] + Ls[sent_id]
            assert (doc_pos_e <= Ls[-1])
            nodes[sent_id].append([sent_id] + [ns_no] + [doc_pos_s, doc_pos_e])
            nodes_sent[sent_id].append([sent_id] + n['pos'])
    id = 0

    for ns in nodes:
        for n in ns:
            n.insert(0, id)
            id += 1

    assert (id <= MAX_NODE_NUM)

    entity_num = len(data[1])
    # sent_num = len(data['sents'])

    # generate entities(nodes) mask for document
    for ns in nodes:
        for n in ns:
            node_position[n[0]][n[3]:n[4]] = 1

    # generate entities(nodes) mask for sentences in a document
    for sent_no, ns in enumerate(nodes_sent):
        # print("len of ns is {}".format(len(ns)))
        assert (len(ns) < MAX_NODE_PER_SENT)
        node_sent_num[sent_no] = len(ns)
        for n_no, n in enumerate(ns):  # node no in a sentence
            assert (sent_no == n[0])
            node_position_sent[sent_no][n_no][n[1]:n[2]] = 1

    # entity matrixs
    for e_no, es in enumerate(data[1]):
        for e in es:
            sent_id = int(e['sent_id'])
            doc_pos_s = e['pos'][0] + Ls[sent_id]
            doc_pos_e = e['pos'][1] + Ls[sent_id]
            entity_position[e_no][doc_pos_s:doc_pos_e] = 1

    total_mentions = id  # + entity_num + sent_num

    total_num_nodes = total_mentions + entity_num
    assert (total_num_nodes <= MAX_NODE_NUM)

    return total_mentions  # only mentions


def ExtractMDPNode(data, sdp_pos, sdp_num, Ls):
    """
    Extract MDP node for each document
    :param data:
    :param sdp_pos: sdp here indicates shortest dependency path:
    :return:
    """
    sents = data[0]
    nodes = [[] for _ in range(len(data[0]))]
    sdp_lists = []
    # create mention's list for each sentence
    for ns_no, ns in enumerate(data[1]):
        for n in ns:
            sent_id = int(n['sent_id'])
            nodes[sent_id].append(n['pos'])

    for sent_no in range(len(sents)):
        spacy_sent = nlp(' '.join(sents[sent_no]))
        edges = []
        if len(spacy_sent) != len(sents[sent_no]):
            # print("{}th doc {}th sent. not valid spacy parsing as the length is not the same to original sentence. ".format(doc_no, sent_no))
            sdp_lists.append([])
            continue
        #       assert (len(spacy_sent) == len(sents[sent_no])) # make sure the length of sentence parsed by spacy is the same as original sentence.
        for token in spacy_sent:
            for child in token.children:
                edges.append(('{0}'.format(token.i), '{0}'.format(child.i)))

        graph = nx.Graph(edges)  # Get the length and path

        mention_num = len(nodes[sent_no])
        sdp_list = []
        # get the shortest dependency path of all mentions in a sentence
        entity_indices = []
        for m_i in range(mention_num):  # m_i is the mention number
            indices_i = [nodes[sent_no][m_i][0] + offset for offset in
                         range(nodes[sent_no][m_i][1] - nodes[sent_no][m_i][0])]
            entity_indices = entity_indices + indices_i
            for m_j in range(mention_num):  #
                if m_i == m_j:
                    continue
                indices_j = [nodes[sent_no][m_j][0] + offset for offset in
                             range(nodes[sent_no][m_j][1] - nodes[sent_no][m_j][0])]
                for index_i in indices_i:
                    for index_j in indices_j:
                        try:
                            sdp_path = nx.shortest_path(graph, source='{0}'.format(index_i),
                                                        target='{0}'.format(index_j))
                        except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
                            # print("no path")
                            # print(e)
                            continue
                        sdp_list.append(sdp_path)
        # get the sdp indices in a sentence
        sdp_nodes_flat = [sdp for sub_sdp in sdp_list for sdp in sub_sdp]
        entity_set = set(entity_indices)
        sdp_nodes_set = set(sdp_nodes_flat)
        # minus the entity node
        sdp_list = list(set([int(n) for n in sdp_nodes_set]) - entity_set)
        sdp_list.sort()
        sdp_lists.append(sdp_list)

    # calculate the sdp position in a document
    if len(sents) != len(sdp_lists):
        print("len mismatch")
    for i in range(len(sents)):
        if len(sdp_lists[i]) == 0:
            continue
        for j, sdp in enumerate(sdp_lists[i]):
            if j > len(sdp_lists[i]) - 1:
                print("list index out of range")
            sdp_lists[i][j] = sdp + Ls[i]

    flat_sdp = [sdp for sub_sdp in sdp_lists for sdp in sub_sdp]

    # set the sdp poistion as 1. for example, if the sdp_pos size is 100 X 512, then we will set the value in each row as 1 according to flat_sdp[i]
    for i in range(len(flat_sdp)):
        if i > MAX_ENTITY_NUM - 1:
            continue
        sdp_pos[i][flat_sdp[i]] = 1

    sdp_num[0] = len(flat_sdp)


def GetEntityId(entity, vertex):
    for e_i, ver in enumerate(vertex):
        if entity in ver[0]['name'] and len(entity) == len(ver[0]['name']):
            return e_i


def SelectSent(sents, entities, triggers):
    select_sent = []
    for entity in entities:
        for mention in entity:
            sent_id = mention['sent_id']
            if sent_id not in select_sent:
                select_sent.append(sent_id)
    trigger_index = [[] for _ in triggers]
    for t_i, trigger in enumerate(triggers):
        if len(trigger) != 0:
            sent_ids = []
            ents = trigger.split(' ')
            e_len = len(ents)
            for s_id, sent in enumerate(sents):
                sent_lower = [word.lower() for word in sent]
                if ents[0].lower() in sent_lower:
                    pos_1 = sent_lower.index(ents[0].lower())
                    flag = True
                    for e_i, pos in enumerate(range(pos_1, pos_1 + e_len)):
                        if ents[e_i].lower() in sent_lower[pos] and len(ents[e_i]) == len(sent[pos]):
                            pass
                        else:
                            flag = False
                            break
                    if flag:
                        trigger_index[t_i].append([pos_1, pos_1 + e_len, s_id])
                        sent_ids.append(s_id)
            if len(sent_ids) == 0:
                for s_id, sent in enumerate(sents):
                    pos_1 = -1
                    pos_2 = -1
                    for ti, token in enumerate(sent):
                        if trigger.lower().startswith((token.lower())) and pos_1 == -1:
                            pos_1 = ti
                        if trigger.lower().endswith((token.lower())) and pos_1 != -1:
                            pos_2 = ti
                    if pos_1 != -1 and pos_2 != -1 and pos_2 - pos_1 > 0:
                        for ind in range(pos_1 + 1, pos_2 + 1):
                            try:
                                if sent[ind].lower() in trigger.lower():
                                    trigger_index[t_i].append([pos_1, pos_2 + 1, s_id])
                                    sent_ids.append(s_id)
                                    break
                            except:
                                print('')
                    elif pos_1 != -1 and pos_2 != -1:
                        trigger_index[t_i].append([pos_1, pos_2 + 1, s_id])
                        sent_ids.append(s_id)
            bianxing = ['s', 'es', 'ing', 'ed', 'ers', '.']
            bianxing_n = 0
            while len(sent_ids) == 0 and bianxing_n < len(bianxing):
                trigger_1 = trigger + bianxing[bianxing_n]
                ents_1 = trigger_1.split(' ')
                for s_id, sent in enumerate(sents):
                    sent_lower = [word.lower() for word in sent]
                    if ents_1[0].lower() in sent_lower:
                        pos_1 = sent_lower.index(ents_1[0].lower())
                        flag = True
                        for e_i, pos in enumerate(range(pos_1, pos_1 + e_len)):
                            if ents_1[e_i].lower() in sent_lower[pos] and len(ents_1[e_i]) == len(sent[pos]):
                                pass
                            else:
                                flag = False
                                break
                        if flag:
                            trigger_index[t_i].append([pos_1, pos_1 + e_len, s_id])
                            sent_ids.append(s_id)
                if len(sent_ids) == 0:
                    for s_id, sent in enumerate(sents):
                        pos_1 = -1
                        pos_2 = -1
                        for ti, token in enumerate(sent):
                            if trigger_1.lower().startswith((token.lower())) and pos_1 == -1:
                                pos_1 = ti
                            if trigger_1.lower().endswith((token.lower())) and pos_1 != -1:
                                pos_2 = ti
                        if pos_1 != -1 and pos_2 != -1 and pos_2 - pos_1 > 0:
                            for ind in range(pos_1 + 1, pos_2 + 1):
                                try:
                                    if sent[ind].lower() in trigger_1.lower():
                                        trigger_index[t_i].append([pos_1, pos_2 + 1, s_id])
                                        sent_ids.append(s_id)
                                        break
                                except:
                                    print('')
                        elif pos_1 != -1 and pos_2 != -1:
                            trigger_index[t_i].append([pos_1, pos_2 + 1, s_id])
                            sent_ids.append(s_id)
                bianxing_n += 1
            sent_ids = list(set(sent_ids))
            try:
                assert len(sent_ids) != 0
            except:
                print('')
                if len(sents) == 28 and "love" in trigger:
                    trigger_index[t_i].append([10, 11, 24])
            select_sent += sent_ids
    select_sent = list(set(select_sent))
    return select_sent, trigger_index


def Init(data_file_name, pair_tot, rel2id, max_length=512, is_training=True, suffix=''):
    ori_data = json.load(open(data_file_name))
    sen_tot = len(ori_data)

    Ma = 0
    Ma_e = 0
    data = []
    intrain = notintrain = notindevtrain = indevtrain = 0

    node_position = np.zeros((pair_tot, MAX_NODE_NUM, max_length), dtype=np.int16)
    node_position_sent = np.zeros((pair_tot, MAX_SENT_NUM, MAX_NODE_PER_SENT, MAX_SENT_LEN), dtype=np.int16)
    node_sent_num = np.zeros((pair_tot, MAX_SENT_NUM), dtype=np.int16)
    entity_position = np.zeros((pair_tot, MAX_ENTITY_NUM, max_length), dtype=np.int16)
    node_num = np.zeros((pair_tot, 1), dtype=np.int16)

    sdp_position = np.zeros((pair_tot, MAX_ENTITY_NUM, max_length), dtype=np.int16)
    sdp_num = np.zeros((pair_tot, 1), dtype=np.int16)

    newdata = []
    pair_num = 0
    for i in range(len(ori_data)):
        new_d = {}
        Ls = [0]
        L = 0
        doc = []
        for x in ori_data[i][0]:
            spacy_sent = nlp(x)
            tokens = []
            L += len(spacy_sent)
            Ls.append(L)
            for token in spacy_sent:
                tokens.append(token.text)
            doc.append(tokens)
        for instance in ori_data[i][1]:
            if pair_num % 200 == 0:
                print("generating the {}th instance from the file {}".format(pair_num, data_file_name))
            entities = []
            h_entity = instance['x']
            h_type = instance['x_type']
            t_entity = instance['y']
            t_type = instance['y_type']
            triggers = instance['t']
            entities.append((h_entity, h_type))
            entities.append((t_entity, t_type))
            entities = GetVertexSet(doc, entities)
            select_sentences, trigger_index = SelectSent(doc, entities, triggers)
            node_num[pair_num] = GetNodePosition([doc, entities], node_position[pair_num], node_position_sent[pair_num],
                                                 node_sent_num[pair_num], entity_position[pair_num], Ls)

            ExtractMDPNode([doc, entities], sdp_position[pair_num], sdp_num[pair_num], Ls)

            vertexSet = entities
            # point position added with sent start position
            for j in range(len(vertexSet)):
                for k in range(len(vertexSet[j])):
                    vertexSet[j][k]['sent_id'] = int(vertexSet[j][k]['sent_id'])

                    sent_id = vertexSet[j][k]['sent_id']
                    dl = Ls[sent_id]
                    pos1 = vertexSet[j][k]['pos'][0]
                    pos2 = vertexSet[j][k]['pos'][1]
                    vertexSet[j][k]['pos'] = (pos1 + dl, pos2 + dl)

            new_d['vertexSet'] = vertexSet

            item = {}
            item['vertexSet'] = vertexSet
            item['sents'] = doc
            item['select_sents'] = select_sentences
            labels = [instance]
            label = labels[0]
            rels = label['r']
            h_id = GetEntityId(label['x'], vertexSet)
            t_id = GetEntityId(label['y'], vertexSet)
            rid = label['rid']
            label['h'] = h_id
            label['t'] = t_id
            label['triggers'] = triggers
            label['triggers_index'] = trigger_index
            item['labels'] = [label]
            item['Ls'] = Ls
            h_words = []
            t_words = []
            for mention in vertexSet[label['h']]:
                for tok in mention['mention'].split(' '):
                    if tok not in h_words:
                        h_words += [tok]
            for mention in vertexSet[label['t']]:
                for tok in mention['mention'].split(' '):
                    if tok not in t_words:
                        t_words += [tok]
            h_t_pair_words = h_words + t_words

            item['h_t_pair_words'] = h_t_pair_words
            data.append(item)
            pair_num += 1

    print('data_len:', len(ori_data))
    print('pair len:', pair_num)

    # saving
    print("Saving files")
    if is_training:
        name_prefix = "train"
    else:
        name_prefix = "dev"

    json.dump(data, open(os.path.join(out_path, name_prefix + suffix + '.json'), "w"))

    char2id = json.load(open(os.path.join(in_path, "char2id.json")))
    word2id = json.load(open(os.path.join(in_path, "word2id.json")))

    ner2id = json.load(open(os.path.join(in_path, "ner2id.json")))

    sen_word = np.zeros((pair_tot, max_length), dtype=np.int64)
    sen_wordstr = np.zeros((pair_tot, max_length), dtype=np.object)
    sen_pos = np.zeros((pair_tot, max_length), dtype=np.int16)
    sen_ner = np.zeros((pair_tot, max_length), dtype=np.int16)
    sen_char = np.zeros((pair_tot, max_length, char_limit), dtype=np.int16)
    sen_seg = np.zeros((pair_tot, max_length), dtype=np.int16)
    # 新增
    bert_token = np.zeros((pair_tot, 512), dtype=np.int64)
    bert_mask = np.zeros((pair_tot, 512), dtype=np.int64)
    bert_starts = np.zeros((pair_tot, 512), dtype=np.int64)

    unkown_words = set()
    max_len_doc = 0
    for i in range(len(data)):

        item = data[i]
        words = []
        sen_seg[i][0] = 1
        for sent in item['sents']:
            words += sent
            sen_seg[i][len(words) - 1] = 1
        bert_token[i], bert_mask[i], bert_starts[i] = bert.subword_tokenize_to_ids(words)
        max_len_doc = max(max_len_doc, len(words))
        for j, word in enumerate(words):
            word = word.lower()
            sen_wordstr[i][j] = word
            # print(sen_wordstr[i][j])
            if j < max_length:
                if word in word2id:
                    sen_word[i][j] = word2id[word]
                else:
                    sen_word[i][j] = word2id['UNK']
                    unkown_words.add(word)
                if sen_word[i][j] < 0:
                    print("the id should not be negative")
            for c_idx, k in enumerate(list(word)):
                if c_idx >= char_limit:
                    break
                sen_char[i, j, c_idx] = char2id.get(k, char2id['UNK'])

        for j in range(j + 1, max_length):
            sen_word[i][j] = word2id['BLANK']

        vertexSet = item['vertexSet']

        for idx, vertex in enumerate(vertexSet, 1):
            for v in vertex:
                sen_pos[i][v['pos'][0]:v['pos'][1]] = idx
                ner_type_B = ner2id[v['type']]
                ner_type_I = ner_type_B + 1
                sen_ner[i][v['pos'][0]] = ner_type_B
                sen_ner[i][v['pos'][0] + 1:v['pos'][1]] = ner_type_I

    print("Finishing processing")
    print("max_len_doc:" + str(max_len_doc))
    np.save(os.path.join(out_path, name_prefix + suffix + '_word.npy'), sen_word[:, :512])
    np.save(os.path.join(out_path, name_prefix + suffix + '_pos.npy'), sen_pos[:, :512])
    np.save(os.path.join(out_path, name_prefix + suffix + '_ner.npy'), sen_ner[:, :512])
    np.save(os.path.join(out_path, name_prefix + suffix + '_char.npy'), sen_char[:, :512, :])
    np.save(os.path.join(out_path, name_prefix + suffix + '_wordstr.npy'), sen_wordstr[:, :512])
    np.save(os.path.join(out_path, name_prefix + suffix + '_seg.npy'), sen_seg[:, :512])
    np.save(os.path.join(out_path, name_prefix + suffix + '_node_position.npy'), node_position[:, :, :512])
    np.save(os.path.join(out_path, name_prefix + suffix + '_node_position_sent.npy'), node_position_sent)
    np.save(os.path.join(out_path, name_prefix + suffix + '_node_num.npy'), node_num)
    np.save(os.path.join(out_path, name_prefix + suffix + '_entity_position.npy'), entity_position[:, :, :512])
    np.save(os.path.join(out_path, name_prefix + suffix + '_sdp_position.npy'), sdp_position[:, :, :512])
    np.save(os.path.join(out_path, name_prefix + suffix + '_sdp_num.npy'), sdp_num)
    np.save(os.path.join(out_path, name_prefix + suffix + '_node_sent_num.npy'), node_sent_num)
    np.save(os.path.join(out_path, name_prefix + suffix + '_bert_word.npy'), bert_token)
    np.save(os.path.join(out_path, name_prefix + suffix + '_bert_mask.npy'), bert_mask)
    np.save(os.path.join(out_path, name_prefix + suffix + '_bert_starts.npy'), bert_starts)

    print("unk number for {} is: {}".format(suffix, len(unkown_words)))
    print("Finish saving")


print("=========================start to generate the training instances=========================")
Init(train_file_name, 5963, rel2id, max_length=700, is_training=False, suffix='_train')
print("=========================start to generate the dev instances=========================")
Init(dev_file_name, 1928, rel2id, max_length=700, is_training=False, suffix='_dev')
print("=========================start to generate the test instances=========================")
Init(test_file_name, 1858, rel2id, max_length=700, is_training=False, suffix='_test')
