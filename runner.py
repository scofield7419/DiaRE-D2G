# coding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import json
import gc
from collections import defaultdict
import random
from models.evaluation import Accuracy

from operator import add

MAX_NODE_NUM = 512

IGNORE_INDEX = -100
is_transformer = False

DEBUG_DOC_NO = 60

from utils import torch_utils


def isNaN(num):
    return num != num


class Runner(object):
    def __init__(self, args):
        self.opt = args
        self.acc_NA = Accuracy()
        self.acc_not_NA = Accuracy()
        self.acc_total = Accuracy()
        self.data_path = args.data_path
        self.use_spemb = args.use_spemb
        self.use_wratt = args.use_wratt
        self.use_gcn = args.use_gcn
        self.use_bag = False
        self.use_gpu = True
        self.is_training = True
        self.max_length = 512
        self.pos_num = 2 * self.max_length
        self.entity_num = self.max_length

        self.relation_num = 37
        self.ner_vocab_len = 13

        self.max_sent_len = 200
        self.max_entity_num = 100
        self.max_sent_num = 50
        self.max_node_num = 200
        self.max_node_per_sent = 40

        self.rnn_hidden = args.hidden_dim  # hidden emb dim
        # self.coref_size = args.coref_dim # coref emb dim
        # self.entity_type_size = args.pos_dim # entity emb dim
        self.max_epoch = args.num_epoch
        self.opt_method = 'Adam'
        self.optimizer = None

        self.checkpoint_dir = './checkpoint'
        self.fig_result_dir = './fig_result'
        self.test_epoch = 1
        self.pretrain_model = args.pretrain_model

        self.word_size = 100
        self.epoch_range = None
        self.dropout_rate = args.dropout_rate  # for sequence
        self.keep_prob = 0.8  # for lstm

        self.period = 50
        self.batch_size = args.batch_size
        self.h_t_limit = 1800
        self.max_patience = 15
        self.patience = 0

        self.test_batch_size = self.batch_size
        self.test_relation_limit = 1800
        self.char_limit = 16
        self.sent_limit = 50
        self.max_entity_length = 20
        self.dis2idx = np.zeros((512), dtype='int64')
        self.dis2idx[1] = 1
        self.dis2idx[2:] = 2
        self.dis2idx[4:] = 3
        self.dis2idx[8:] = 4
        self.dis2idx[16:] = 5
        self.dis2idx[32:] = 6
        self.dis2idx[64:] = 7
        self.dis2idx[128:] = 8
        self.dis2idx[256:] = 9
        self.dis_size = 20

        self.train_prefix = args.train_prefix
        self.test_prefix = args.test_prefix

        self.lr = args.lr
        self.decay_epoch = args.decay_epoch

        self.lr_decay = args.lr_decay
        if not os.path.exists("log"):
            os.mkdir("log")

        self.softmax = nn.Softmax(dim=-1)

        self.dropout_emb = args.dropout_emb
        self.dropout_rnn = args.dropout_rnn
        self.dropout_gcn = args.dropout_gcn

        self.max_grad_norm = args.max_grad_norm
        self.optim = args.optim
        self.evaluate_epoch = args.evaluate_epoch

    def set_data_path(self, data_path):
        self.data_path = data_path

    def set_max_length(self, max_length):
        self.max_length = max_length
        self.pos_num = 2 * self.max_length

    def set_num_classes(self, num_classes):
        self.num_classes = num_classes

    def set_window_size(self, window_size):
        self.window_size = window_size

    def set_word_size(self, word_size):
        self.word_size = word_size

    def set_max_epoch(self, max_epoch):
        self.max_epoch = max_epoch

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_opt_method(self, opt_method):
        self.opt_method = opt_method

    def set_drop_prob(self, drop_prob):
        self.drop_prob = drop_prob

    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

    def set_test_epoch(self, test_epoch):
        self.test_epoch = test_epoch

    def set_pretrain_model(self, pretrain_model):
        self.pretrain_model = pretrain_model

    def set_is_training(self, is_training):
        self.is_training = is_training

    def set_use_bag(self, use_bag):
        self.use_bag = use_bag

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu

    def set_epoch_range(self, epoch_range):
        self.epoch_range = epoch_range

    def load_train_data(self):
        print("Reading training data...")
        prefix = self.train_prefix

        print('train', prefix)

        self.data_train_word = np.load(os.path.join(self.data_path, prefix + '_word.npy'))

        # elmo_ids = batch_to_ids(batch_words).cuda()
        self.data_train_pos = np.load(os.path.join(self.data_path, prefix + '_pos.npy'))
        self.data_train_ner = np.load(os.path.join(self.data_path, prefix + '_ner.npy'))  # word_embedding
        self.data_train_char = np.load(os.path.join(self.data_path, prefix + '_char.npy'))
        self.data_train_seg = np.load(os.path.join(self.data_path, prefix + '_seg.npy'))
        # 
        self.data_train_bert_word = np.load(os.path.join(self.data_path, prefix + '_bert_word.npy'))
        self.data_train_bert_mask = np.load(os.path.join(self.data_path, prefix + '_bert_mask.npy'))
        self.data_train_bert_starts = np.load(os.path.join(self.data_path, prefix + '_bert_starts.npy'))

        self.data_train_node_position = np.load(os.path.join(self.data_path, prefix + '_node_position.npy'))

        self.data_train_node_position_sent = np.load(os.path.join(self.data_path, prefix + '_node_position_sent.npy'))

        self.data_train_node_sent_num = np.load(os.path.join(self.data_path, prefix + '_node_sent_num.npy'))

        self.data_train_node_num = np.load(os.path.join(self.data_path, prefix + '_node_num.npy'))
        self.data_train_entity_position = np.load(os.path.join(self.data_path, prefix + '_entity_position.npy'))
        self.train_file = json.load(open(os.path.join(self.data_path, prefix + '.json')))

        self.data_train_sdp_position = np.load(os.path.join(self.data_path, prefix + '_sdp_position.npy'))
        self.data_train_sdp_num = np.load(os.path.join(self.data_path, prefix + '_sdp_num.npy'))

        self.train_len = ins_num = self.data_train_word.shape[0]

        assert (self.train_len == len(self.train_file))
        print("Finish reading, total reading {} train documetns".format(self.train_len))

        self.train_order = list(range(ins_num))
        self.train_batches = ins_num // self.batch_size
        if ins_num % self.batch_size != 0:
            self.train_batches += 1

    def load_test_data(self):
        print("Reading testing data...")

        self.rel2id = json.load(open(os.path.join('data', 'rel2id.json')))
        self.id2rel = {v: k for k, v in self.rel2id.items()}

        prefix = self.test_prefix

        print(prefix)
        self.is_test = ('dev_test' == prefix)

        self.data_test_word = np.load(os.path.join(self.data_path, prefix + '_word.npy'))
        self.data_test_pos = np.load(os.path.join(self.data_path, prefix + '_pos.npy'))
        self.data_test_ner = np.load(os.path.join(self.data_path, prefix + '_ner.npy'))
        self.data_test_char = np.load(os.path.join(self.data_path, prefix + '_char.npy'))
        # 
        self.data_test_bert_word = np.load(os.path.join(self.data_path, prefix + '_bert_word.npy'))
        self.data_test_bert_mask = np.load(os.path.join(self.data_path, prefix + '_bert_mask.npy'))
        self.data_test_bert_starts = np.load(os.path.join(self.data_path, prefix + '_bert_starts.npy'))

        self.data_test_node_position = np.load(os.path.join(self.data_path, prefix + '_node_position.npy'))

        self.data_test_node_position_sent = np.load(os.path.join(self.data_path, prefix + '_node_position_sent.npy'))
        # self.data_test_adj = np.load(os.path.join(self.data_path, prefix+'_adj.npy'))

        self.data_test_node_sent_num = np.load(os.path.join(self.data_path, prefix + '_node_sent_num.npy'))

        self.data_test_node_num = np.load(os.path.join(self.data_path, prefix + '_node_num.npy'))
        self.data_test_entity_position = np.load(os.path.join(self.data_path, prefix + '_entity_position.npy'))
        self.test_file = json.load(open(os.path.join(self.data_path, prefix + '.json')))
        self.data_test_seg = np.load(os.path.join(self.data_path, prefix + '_seg.npy'))
        self.test_len = self.data_test_word.shape[0]

        self.data_test_sdp_position = np.load(os.path.join(self.data_path, prefix + '_sdp_position.npy'))
        self.data_test_sdp_num = np.load(os.path.join(self.data_path, prefix + '_sdp_num.npy'))

        assert (self.test_len == len(self.test_file))

        print("Finish reading, total reading {} test documetns".format(self.test_len))

        self.test_batches = self.data_test_word.shape[0] // self.test_batch_size
        if self.data_test_word.shape[0] % self.test_batch_size != 0:
            self.test_batches += 1

        self.test_order = list(range(self.test_len))
        self.test_order.sort(key=lambda x: np.sum(self.data_test_word[x] > 0), reverse=True)

    def get_train_batch(self):

        random.shuffle(self.train_order)
        context_idxs = torch.LongTensor(self.batch_size, self.max_length).cuda()
        context_pos = torch.LongTensor(self.batch_size, self.max_length).cuda()
        h_mapping = torch.Tensor(self.batch_size, self.h_t_limit, self.max_length).cuda()
        t_mapping = torch.Tensor(self.batch_size, self.h_t_limit, self.max_length).cuda()
        relation_multi_label = torch.Tensor(self.batch_size, self.h_t_limit, self.relation_num - 1).cuda()
        relation_mask = torch.Tensor(self.batch_size, self.h_t_limit).cuda()

        context_masks = torch.LongTensor(self.batch_size, self.max_length).cuda()
        context_starts = torch.LongTensor(self.batch_size, self.max_length).cuda()
        # 
        # context_s_starts = torch.LongTensor(self.batch_size, self.max_length).cuda()

        pos_idx = torch.LongTensor(self.batch_size, self.max_length).cuda()

        context_ner = torch.LongTensor(self.batch_size, self.max_length).cuda()
        context_char_idxs = torch.LongTensor(self.batch_size, self.max_length, self.char_limit).cuda()
        # 
        attention_label_mask = torch.Tensor(self.batch_size, self.max_length, self.relation_num - 1).cuda()

        speaker_label = torch.LongTensor(self.batch_size, self.max_length).cuda()
        speaker_sent_label = torch.Tensor(self.batch_size, self.max_sent_num, self.max_sent_num, 2).cuda()
        speaker_sent_label_mask = torch.Tensor(self.batch_size, self.max_sent_num, self.max_sent_num).cuda()

        # relation_label = torch.LongTensor(self.batch_size, self.h_t_limit).cuda()

        ht_pair_pos = torch.LongTensor(self.batch_size, self.h_t_limit).cuda()

        context_seg = torch.LongTensor(self.batch_size, self.max_length).cuda()

        node_position_sent = torch.zeros(self.batch_size, self.max_sent_num, self.max_node_per_sent,
                                         self.max_sent_len).float()

        node_position = torch.zeros(self.batch_size, self.max_node_num, self.max_length).float().cuda()

        sdp_position = torch.zeros(self.batch_size, self.max_entity_num, self.max_length).float().cuda()
        sdp_num = torch.zeros(self.batch_size, 1).long().cuda()

        node_sent_num = torch.zeros(self.batch_size, self.max_sent_num).float().cuda()

        entity_position = torch.zeros(self.batch_size, self.max_entity_num, self.max_length).float().cuda()
        node_num = torch.zeros(self.batch_size, 1).long().cuda()

        for b in range(self.train_batches):

            entity_num = []
            sentence_num = []
            sentence_len = []
            node_num_per_sent = []

            start_id = b * self.batch_size
            cur_bsz = min(self.batch_size, self.train_len - start_id)
            cur_batch = list(self.train_order[start_id: start_id + cur_bsz])
            cur_batch.sort(key=lambda x: np.sum(self.data_train_word[x] > 0), reverse=True)

            for mapping in [h_mapping, t_mapping, attention_label_mask, speaker_label, speaker_sent_label]:
                mapping.zero_()

            for mapping in [relation_multi_label, relation_mask, pos_idx, speaker_sent_label_mask]:
                mapping.zero_()

            ht_pair_pos.zero_()

            max_h_t_cnt = 1
            context_words = [[] for _ in cur_batch]
            indexes = []
            h_t_pair_words = []

            sdp_nums = []
            for i, index in enumerate(cur_batch):

                context_idxs[i].copy_(torch.from_numpy(self.data_train_bert_word[index, :]))
                context_pos[i].copy_(torch.from_numpy(self.data_train_pos[index, :]))  # ???
                context_char_idxs[i].copy_(torch.from_numpy(self.data_train_char[index, :]))
                context_ner[i].copy_(torch.from_numpy(self.data_train_ner[index, :]))
                context_seg[i].copy_(torch.from_numpy(self.data_train_seg[index, :]))

                context_masks[i].copy_(torch.from_numpy(self.data_train_bert_mask[index, :]))
                context_starts[i].copy_(torch.from_numpy(self.data_train_bert_starts[index, :]))

                indexes.append(index)
                ins = self.train_file[index]
                context_words[i] = ins['sents']
                h_t_pair_words.append(ins['h_t_pair_words'])
                labels = ins['labels']
                Ls = ins['Ls']
                idx2label = defaultdict(list)

                for label in labels:
                    for rid in label['rid']:
                        idx2label[(label['h'], label['t'])].append(rid - 1)
                    for trigger in label['triggers_index']:
                        for t_index in trigger:
                            sent_id = t_index[2]
                            pos_1 = t_index[0] + Ls[sent_id]
                            pos_2 = t_index[1] + Ls[sent_id]
                speaker_sents = defaultdict(list)
                speaker_sent_label[i, :, :, 0] = 1
                speaker_sent_label[i, :, :, 1] = 0
                speaker_sent_label_mask[i, :(len(Ls) - 1), :(len(Ls) - 1)] = 1
                for si in range(len(Ls) - 1):
                    start = Ls[si]
                    end = Ls[si + 1]
                    speaker_id = int(ins['sents'][si][1][:1])
                    if start > speaker_label.shape[1]:
                        break
                    speaker_label[i, start:end] = speaker_id
                    speaker_sents[speaker_id].append(si)
                for speaker_id, sents in speaker_sents.items():
                    for si in sents:
                        for si2 in sents:
                            speaker_sent_label[i, si, si2, 1] = 1
                            speaker_sent_label[i, si, si2, 0] = 0

                node_position[i].copy_(torch.from_numpy(self.data_train_node_position[index]))

                node_position_sent[i].copy_(torch.from_numpy(self.data_train_node_position_sent[index]))

                node_sent_num[i].copy_(torch.from_numpy(self.data_train_node_sent_num[index]))

                node_num[i].copy_(torch.from_numpy(self.data_train_node_num[index]))
                entity_position[i].copy_(torch.from_numpy(self.data_train_entity_position[index]))

                entity_num.append(len(ins['vertexSet']))
                sentence_num.append(len(ins['sents']))
                sentence_len.append(max([len(sent) for sent in ins['sents']]))  # max sent len of a document
                node_num_per_sent.append(max(node_sent_num[i].tolist()))

                sdp_position[i].copy_(torch.from_numpy(self.data_train_sdp_position[index]))
                sdp_num[i].copy_(torch.from_numpy(self.data_train_sdp_num[index]))

                sdp_no_trucation = sdp_num[i].item()
                if sdp_no_trucation > self.max_entity_num:
                    sdp_no_trucation = self.max_entity_num
                sdp_nums.append(sdp_no_trucation)
                evi_sentences = ins['select_sents']
                # for evi in evi_sentences:
                #     start_sen = Ls[evi]
                #     end_sen = Ls[evi+1]
                # attention_label[i, start_sen:end_sen, :] = 1
                attention_label_mask[i, :Ls[-1], :] = 1
                for j in range(self.max_length):
                    if self.data_train_word[index, j] == 0:
                        break
                    pos_idx[i, j] = j + 1

                train_tripe = list(idx2label.keys())
                for j, (h_idx, t_idx) in enumerate(train_tripe):
                    hlist = ins['vertexSet'][h_idx]
                    tlist = ins['vertexSet'][t_idx]

                    for h in hlist:
                        h_mapping[i, j, h['pos'][0]:h['pos'][1]] = 1.0 / len(hlist) / (h['pos'][1] - h['pos'][0])

                    for t in tlist:
                        t_mapping[i, j, t['pos'][0]:t['pos'][1]] = 1.0 / len(tlist) / (t['pos'][1] - t['pos'][0])

                    label = idx2label[(h_idx, t_idx)]
                    for r in label:
                        if r < 36:
                            relation_multi_label[i, j, r] = 1

                    relation_mask[i, j] = 1
                    delta_dis = hlist[0]['pos'][0] - tlist[0]['pos'][0]

                    if abs(delta_dis) >= self.max_length:  # for gda
                        continue

                    if delta_dis < 0:
                        ht_pair_pos[i, j] = -int(self.dis2idx[-delta_dis])
                    else:
                        ht_pair_pos[i, j] = int(self.dis2idx[delta_dis])

                    # rt = np.random.randint(len(label))
                    # relation_label[i, j] = label[rt]

                max_h_t_cnt = max(max_h_t_cnt, len(train_tripe))

            input_lengths = (context_idxs[:cur_bsz] > 0).long().sum(dim=1)
            max_c_len = int(input_lengths.max())  # max length of a document
            max_token = int((context_starts[:cur_bsz, :max_c_len] > 0).long().sum(dim=1).max())
            entity_mention_num = list(map(add, entity_num, node_num[:cur_bsz].squeeze(1).tolist()))
            max_sdp_num = max(sdp_nums)
            all_node_num = list(map(add, sdp_nums, entity_mention_num))

            max_entity_num = max(entity_num)
            max_sentence_num = max(sentence_num)
            b_max_mention_num = int(node_num[:cur_bsz].max())  # - max_entity_num - max_sentence_num
            all_node_num = torch.LongTensor(all_node_num)

            yield {'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
                   'context_pos': context_pos[:cur_bsz, :max_c_len].contiguous(),
                   'h_mapping': h_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
                   't_mapping': t_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
                   # 'relation_label': relation_label[:cur_bsz, :max_h_t_cnt].contiguous(),
                   'input_lengths': input_lengths,
                   'pos_idx': pos_idx[:cur_bsz, :max_c_len].contiguous(),
                   'relation_multi_label': relation_multi_label[:cur_bsz, :max_h_t_cnt],
                   'relation_mask': relation_mask[:cur_bsz, :max_h_t_cnt],
                   'context_ner': context_ner[:cur_bsz, :max_c_len].contiguous(),
                   'context_char_idxs': context_char_idxs[:cur_bsz, :max_c_len].contiguous(),
                   'ht_pair_pos': ht_pair_pos[:cur_bsz, :max_h_t_cnt],
                   'context_seg': context_seg[:cur_bsz, :max_c_len].contiguous(),
                   'node_position': node_position[:cur_bsz, :b_max_mention_num, :max_c_len].contiguous(),
                   'node_sent_num': node_sent_num[:cur_bsz, :max_sentence_num].contiguous(),
                   'entity_position': entity_position[:cur_bsz, :max_entity_num, :max_c_len].contiguous(),
                   'all_node_num': all_node_num,
                   'entity_num': entity_num,
                   'sent_num': sentence_num,
                   'sdp_position': sdp_position[:cur_bsz, :max_sdp_num, :max_c_len].contiguous(),
                   'sdp_num': sdp_nums,
                   'context_masks': context_masks[:cur_bsz, :max_c_len].contiguous(),
                   'context_starts': context_starts[:cur_bsz, :max_c_len].contiguous(),
                   'attention_label_mask': attention_label_mask[:cur_bsz, :max_token, :],
                   'context_words': context_words,
                   'indexes': indexes,
                   'h_t_pair_words': h_t_pair_words,
                   'speaker_label': speaker_label[:cur_bsz, :max_c_len].contiguous(),
                   'speaker_sent_label': speaker_sent_label[:cur_bsz, :max_sentence_num, :max_sentence_num, :],
                   'speaker_sent_label_mask': speaker_sent_label_mask[:cur_bsz, :max_sentence_num, :max_sentence_num],
                   # 'context_s_starts':context_s_starts[:cur_bsz, :max_c_len].contiguous()
                   }

    def get_test_batch(self):
        context_idxs = torch.LongTensor(self.test_batch_size, self.max_length).cuda()
        context_pos = torch.LongTensor(self.test_batch_size, self.max_length).cuda()
        h_mapping = torch.Tensor(self.test_batch_size, self.test_relation_limit, self.max_length).cuda()
        t_mapping = torch.Tensor(self.test_batch_size, self.test_relation_limit, self.max_length).cuda()
        context_ner = torch.LongTensor(self.test_batch_size, self.max_length).cuda()
        context_char_idxs = torch.LongTensor(self.test_batch_size, self.max_length, self.char_limit).cuda()

        attention_label_mask = torch.Tensor(self.test_batch_size, self.max_length, self.relation_num - 1).cuda()
        context_masks = torch.LongTensor(self.test_batch_size, self.max_length).cuda()
        context_starts = torch.LongTensor(self.test_batch_size, self.max_length).cuda()
        speaker_label = torch.LongTensor(self.test_batch_size, self.max_length).cuda()
        speaker_sent_label = torch.Tensor(self.test_batch_size, self.max_sent_num, self.max_sent_num, 2).cuda()
        speaker_sent_label_mask = torch.Tensor(self.test_batch_size, self.max_sent_num, self.max_sent_num).cuda()

        relation_mask = torch.Tensor(self.test_batch_size, self.h_t_limit).cuda()
        ht_pair_pos = torch.LongTensor(self.test_batch_size, self.h_t_limit).cuda()
        context_seg = torch.LongTensor(self.batch_size, self.max_length).cuda()
        relation_multi_label = torch.LongTensor(self.batch_size, self.h_t_limit, self.relation_num - 1).cuda()

        node_position_sent = torch.zeros(self.batch_size, self.max_sent_num, self.max_node_per_sent,
                                         self.max_sent_len).float()

        node_position = torch.zeros(self.batch_size, self.max_node_num, self.max_length).float().cuda()
        entity_position = torch.zeros(self.batch_size, self.max_entity_num, self.max_length).float().cuda()
        node_num = torch.zeros(self.batch_size, 1).long().cuda()

        node_sent_num = torch.zeros(self.batch_size, self.max_sent_num).float().cuda()

        sdp_position = torch.zeros(self.batch_size, self.max_entity_num, self.max_length).float().cuda()
        sdp_num = torch.zeros(self.batch_size, 1).long().cuda()

        for b in range(self.test_batches):

            entity_num = []
            sentence_num = []
            sentence_len = []
            node_num_per_sent = []

            start_id = b * self.test_batch_size
            cur_bsz = min(self.test_batch_size, self.test_len - start_id)
            cur_batch = list(self.test_order[start_id: start_id + cur_bsz])

            for mapping in [h_mapping, t_mapping, relation_mask, relation_multi_label,
                            attention_label_mask, speaker_label,
                            speaker_sent_label, speaker_sent_label_mask]:
                mapping.zero_()

            ht_pair_pos.zero_()

            max_h_t_cnt = 1

            cur_batch.sort(key=lambda x: np.sum(self.data_test_word[x] > 0), reverse=True)

            labels = []

            L_vertex = []
            sdp_nums = []

            vertexSets = []
            context_words = [[] for _ in cur_batch]
            indexes = []
            h_t_pair_words = []
            triggers = []
            evi_sentences_all = []
            for i, index in enumerate(cur_batch):
                context_idxs[i].copy_(torch.from_numpy(self.data_test_bert_word[index, :]))
                context_pos[i].copy_(torch.from_numpy(self.data_test_pos[index, :]))
                context_char_idxs[i].copy_(torch.from_numpy(self.data_test_char[index, :]))
                context_ner[i].copy_(torch.from_numpy(self.data_test_ner[index, :]))
                context_seg[i].copy_(torch.from_numpy(self.data_test_seg[index, :]))
                context_masks[i].copy_(torch.from_numpy(self.data_test_bert_mask[index, :]))
                context_starts[i].copy_(torch.from_numpy(self.data_test_bert_starts[index, :]))

                idx2label = defaultdict(list)
                ins = self.test_file[index]

                context_words[i] = ins['sents']
                Ls = ins['Ls']
                h_t_pair_words.append(ins['h_t_pair_words'])
                triggers.append(ins['labels'][0]['triggers'])
                if len(ins['labels']) == 0:
                    print('')
                for label in ins['labels']:
                    for rid in label['rid']:
                        idx2label[(label['h'], label['t'])].append(rid - 1)
                    for trigger in label['triggers_index']:
                        for t_index in trigger:
                            sent_id = t_index[2]
                            pos_1 = t_index[0] + Ls[sent_id]
                            pos_2 = t_index[1] + Ls[sent_id]
                speaker_sents = defaultdict(list)
                speaker_sent_label[i, :, :, 0] = 1
                speaker_sent_label[i, :, :, 1] = 0
                speaker_sent_label_mask[i, :(len(Ls) - 1), :(len(Ls) - 1)] = 1
                for si in range(len(Ls) - 1):
                    start = Ls[si]
                    end = Ls[si + 1]
                    speaker_id = int(ins['sents'][si][1][:1])
                    if start > speaker_label.shape[1]:
                        break
                    speaker_label[i, start:end] = speaker_id
                    speaker_sents[speaker_id].append(si)
                for speaker_id, sents in speaker_sents.items():
                    for si in sents:
                        for si2 in sents:
                            speaker_sent_label[i, si, si2, 1] = 1
                            speaker_sent_label[i, si, si2, 0] = 0
                node_position[i].copy_(torch.from_numpy(self.data_test_node_position[index]))
                node_position_sent[i].copy_(torch.from_numpy(self.data_test_node_position_sent[index]))

                node_sent_num[i].copy_(torch.from_numpy(self.data_test_node_sent_num[index]))

                node_num[i].copy_(torch.from_numpy(self.data_test_node_num[index]))
                entity_position[i].copy_(torch.from_numpy(self.data_test_entity_position[index]))
                entity_num.append(len(ins['vertexSet']))
                sentence_num.append(len(ins['sents']))
                sentence_len.append(max([len(sent) for sent in ins['sents']]))  # max sent len of a document
                node_num_per_sent.append(max(node_sent_num[i].tolist()))

                sdp_position[i].copy_(torch.from_numpy(self.data_test_sdp_position[index]))
                sdp_num[i].copy_(torch.from_numpy(self.data_test_sdp_num[index]))

                sdp_no_trucation = sdp_num[i].item()
                if sdp_no_trucation > self.max_entity_num:
                    sdp_no_trucation = self.max_entity_num
                sdp_nums.append(sdp_no_trucation)

                evi_sentences = ins['select_sents']
                evi_sentences_all.append(evi_sentences)
                # for evi in evi_sentences:
                #     start_sen = Ls[evi]
                #     end_sen = Ls[evi+1]
                #     attention_label[i, start_sen:end_sen, :] = 1
                L = len(ins['vertexSet'])
                # titles.append(ins['title'])

                vertexSets.append(ins['vertexSet'])

                train_tripe = list(idx2label.keys())

                for j, (h_idx, t_idx) in enumerate(train_tripe):

                    hlist = ins['vertexSet'][h_idx]
                    tlist = ins['vertexSet'][t_idx]

                    for h in hlist:
                        h_mapping[i, j, h['pos'][0]:h['pos'][1]] = 1.0 / len(hlist) / (h['pos'][1] - h['pos'][0])

                    for t in tlist:
                        t_mapping[i, j, t['pos'][0]:t['pos'][1]] = 1.0 / len(tlist) / (t['pos'][1] - t['pos'][0])

                    label = idx2label[(h_idx, t_idx)]
                    for r in label:
                        if r < 36:
                            relation_multi_label[i, j, r] = 1

                    relation_mask[i, j] = 1

                    delta_dis = hlist[0]['pos'][0] - tlist[0]['pos'][0]

                    if abs(delta_dis) >= self.max_length:  # for gda
                        continue

                    if delta_dis < 0:
                        ht_pair_pos[i, j] = -int(self.dis2idx[-delta_dis])
                    else:
                        ht_pair_pos[i, j] = int(self.dis2idx[delta_dis])

                    # rt = np.random.randint(len(label))
                    # relation_label[i, j] = label[rt]

                max_h_t_cnt = max(max_h_t_cnt, len(train_tripe))
                label_set = {}
                for label in ins['labels']:
                    for rid in label['rid']:
                        label_set[(label['h'], label['t'], rid - 1)] = False

                labels.append(label_set)

                L_vertex.append(L)
                indexes.append(index)

            input_lengths = (context_idxs[:cur_bsz] > 0).long().sum(dim=1)
            max_c_len = int(input_lengths.max())
            max_token = int((context_starts[:cur_bsz, :max_c_len] > 0).long().sum(dim=1).max())
            entity_mention_num = list(map(add, entity_num, node_num[:cur_bsz].squeeze(1).tolist()))
            max_sdp_num = max(sdp_nums)
            all_node_num = list(map(add, sdp_nums, entity_mention_num))

            max_entity_num = max(entity_num)
            max_sentence_num = max(sentence_num)
            b_max_mention_num = int(node_num[:cur_bsz].max())  # - max_entity_num - max_sentence_num
            all_node_num = torch.LongTensor(all_node_num)

            yield {'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
                   'context_pos': context_pos[:cur_bsz, :max_c_len].contiguous(),
                   'h_mapping': h_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
                   't_mapping': t_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
                   'context_seg': context_seg[:cur_bsz, :max_c_len].contiguous(),
                   'labels': labels,
                   'relation_multi_label': relation_multi_label[:cur_bsz, :max_h_t_cnt],
                   'L_vertex': L_vertex,
                   'input_lengths': input_lengths,
                   'context_ner': context_ner[:cur_bsz, :max_c_len].contiguous(),
                   'context_char_idxs': context_char_idxs[:cur_bsz, :max_c_len].contiguous(),
                   'relation_mask': relation_mask[:cur_bsz, :max_h_t_cnt],
                   # 'titles': titles,
                   'ht_pair_pos': ht_pair_pos[:cur_bsz, :max_h_t_cnt],
                   'node_position': node_position[:cur_bsz, :b_max_mention_num, :max_c_len].contiguous(),
                   'node_sent_num': node_sent_num[:cur_bsz, :max_sentence_num].contiguous(),
                   'entity_position': entity_position[:cur_bsz, :max_entity_num, :max_c_len].contiguous(),
                   'all_node_num': all_node_num,
                   'entity_num': entity_num,
                   'sent_num': sentence_num,
                   'sdp_position': sdp_position[:cur_bsz, :max_sdp_num, :max_c_len].contiguous(),
                   'sdp_num': sdp_nums,
                   'vertexsets': vertexSets,
                   'context_masks': context_masks[:cur_bsz, :max_c_len].contiguous(),
                   'context_starts': context_starts[:cur_bsz, :max_c_len].contiguous(),
                   'context_words': context_words,
                   'indexes': indexes,
                   'triggers': triggers,
                   'evis': evi_sentences_all,
                   'attention_label_mask': attention_label_mask[:cur_bsz, :max_token, :],
                   'speaker_label': speaker_label[:cur_bsz, :max_c_len].contiguous(),
                   'speaker_sent_label': speaker_sent_label[:cur_bsz, :max_sentence_num, :max_sentence_num, :],
                   'speaker_sent_label_mask': speaker_sent_label_mask[:cur_bsz, :max_sentence_num, :max_sentence_num],
                   # 'context_s_starts': context_s_starts[:cur_bsz, :max_c_len].contiguous()
                   }

    def accuracy(self, out, labels):
        out = np.array(out).reshape(-1)
        out = 1 / (1 + np.exp(-out))
        labels = np.array(labels).reshape(-1)
        return np.sum((out > 0.5) == (labels > 0.5)) / 36

    def train(self, model_pattern, model_name):

        ori_model = model_pattern(config=self)
        if self.pretrain_model != None:
            ori_model.load_state_dict(torch.load(self.pretrain_model))
        ori_model.cuda()

        parameters = [p for p in ori_model.parameters() if p.requires_grad]

        optimizer = torch_utils.get_optimizer(self.optim, parameters, self.lr)

        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, self.lr_decay)

        model = nn.DataParallel(ori_model)

        BCE = nn.BCEWithLogitsLoss(reduction='none')

        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

        best_auc = 0.0
        best_f1 = 0.0
        best_epoch = 0

        model.train()

        global_step = 0
        total_loss = 0
        start_time = time.time()

        def logging(s, print_=True, log_=True):
            if print_:
                print(s)
            if log_:
                with open(os.path.join(os.path.join("log", model_name)), 'a+') as f_log:
                    f_log.write(s + '\n')

        dev_score_list = []
        f1 = 0
        dev_score_list.append(f1)

        for epoch in range(self.max_epoch):
            gc.collect()
            self.acc_NA.clear()
            self.acc_not_NA.clear()
            self.acc_total.clear()
            print("epoch:{}, Learning rate:{}".format(epoch, optimizer.param_groups[0]['lr']))

            epoch_start_time = time.time()
            relations = []
            predictions = []

            # model.eval()
            # f1, bestT2 = self.test(model, model_name)
            for no, data in enumerate(self.get_train_batch()):
                indexes = data['indexes']
                context_idxs = data['context_idxs']
                context_pos = data['context_pos']
                h_mapping = data['h_mapping']
                t_mapping = data['t_mapping']
                # relation_label = data['relation_label']
                input_lengths = data['input_lengths']
                relation_multi_label = data['relation_multi_label']
                relation_mask = data['relation_mask']
                context_ner = data['context_ner']
                context_char_idxs = data['context_char_idxs']
                ht_pair_pos = data['ht_pair_pos']
                context_seg = data['context_seg']
                context_masks = data['context_masks']
                context_starts = data['context_starts']
                # context_s_starts = data['context_s_starts']
                context_words = data['context_words']
                speaker_label = data['speaker_label']

                attention_label_mask = data['attention_label_mask']

                dis_h_2_t = ht_pair_pos + 10
                dis_t_2_h = -ht_pair_pos + 10

                # torch.cuda.empty_cache()

                context_idxs = context_idxs.cuda()
                context_pos = context_pos.cuda()
                context_ner = context_ner.cuda()
                # context_char_idxs = context_char_idxs.cuda()
                # input_lengths = input_lengths.cuda()
                h_mapping = h_mapping.cuda()
                t_mapping = t_mapping.cuda()
                relation_mask = relation_mask.cuda()
                dis_h_2_t = dis_h_2_t.cuda()
                dis_t_2_h = dis_t_2_h.cuda()

                node_position = data['node_position'].cuda()
                entity_position = data['entity_position'].cuda()
                node_sent_num = data['node_sent_num'].cuda()
                all_node_num = data['all_node_num'].cuda()
                entity_num = torch.Tensor(data['entity_num']).cuda()
                # sent_num = torch.Tensor(data['sent_num']).cuda()

                sdp_pos = data['sdp_position'].cuda()
                sdp_num = torch.Tensor(data['sdp_num']).cuda()
                # print(model.state_dict())
                predict_re, attention_weight = model(context_idxs,
                                                     h_mapping, t_mapping, relation_mask,
                                                     node_position, entity_position, node_sent_num,
                                                     entity_num, sdp_pos, sdp_num,
                                                     context_masks, context_starts, attention_label_mask,
                                                     speaker_label)

                relation_multi_label = relation_multi_label.cuda()

                loss = torch.sum(BCE(predict_re, relation_multi_label) * relation_mask.unsqueeze(2)) / torch.sum(
                    relation_mask)

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                optimizer.step()

                predict_re = predict_re[torch.eq(relation_mask, 1)].data.cpu().numpy().tolist()

                # predict_re = [np.argmax(pre) for pre in predict_re]

                relations += relation_multi_label[torch.eq(relation_mask, 1)].data.cpu().numpy().tolist()
                predictions += predict_re
                global_step += 1
                total_loss += loss.item()
                if global_step % self.period == 0:
                    cur_loss = total_loss / self.period
                    elapsed = time.time() - start_time
                    logging('| epoch {:2d} | step {:4d} |  ms/b {:5.2f} | train loss {:5.3f}'.format(epoch, global_step,
                                                                                                     elapsed * 1000 / self.period,
                                                                                                     cur_loss))
                    total_loss = 0
                    start_time = time.time()
            acc = self.accuracy(predictions, relations) / len(predictions)
            logging('| epoch {:2d} | acc {:3.4f}'.format(epoch, acc))
            if epoch >= self.evaluate_epoch:

                logging('-' * 89)
                eval_start_time = time.time()
                model.eval()

                f1, bestT2 = self.test(model, model_name)

                logging("test result || f1: {:3.4f}, T2: {:3.4f}".format(f1, bestT2))

                model.train()
                logging('| epoch {:3d} | time: {:5.2f}s'.format(epoch, time.time() - eval_start_time))
                logging('-' * 89)

                if f1 > best_f1:
                    best_f1 = f1
                    best_epoch = epoch
                    path = os.path.join(self.checkpoint_dir, model_name)
                    torch.save(ori_model.state_dict(), path)
                    logging("best f1 is: {:3.4f}, epoch is: {}, save path is: {}".format(best_f1, best_epoch, path))
                    self.patience = 0
                elif self.patience < self.max_patience:
                    self.patience += 1
                elif self.patience >= self.max_patience:
                    logging("Early stop here. best f1 is: {:3.4f}, epoch is: {}".format(best_f1, best_epoch))
                    break

            if epoch > self.decay_epoch:  # and epoch < self.evaluate_epoch:# and epoch < self.evaluate_epoch:
                if self.optim == 'sgd' and f1 < dev_score_list[-1]:
                    self.lr *= self.lr_decay
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = self.lr

                if self.optim == 'adam' and optimizer.param_groups[0][
                    'lr'] > 1e-4:  # epoch < 30:# and f1 < dev_score_list[-1]:
                    scheduler.step()

            dev_score_list.append(f1)
            print("train time for epoch {}: {}".format(epoch, time.time() - epoch_start_time))

        print("Finish training")
        print("Best epoch = {} | F1 {}, auc = {}".format(best_epoch, best_f1, best_auc))
        print("Storing best result...")
        print("Finish storing")

    def f1_eval(self, logits, features, output_error, all_indexes=None):
        def getpred(result, T1=0.5, T2=0.4):
            ret = []
            for i in range(len(result)):
                r = []
                maxl, maxj = -1, -1
                for j in range(len(result[i])):
                    if result[i][j] > T1:
                        r += [j]
                    if result[i][j] > maxl:
                        maxl = result[i][j]
                        maxj = j
                if len(r) == 0:
                    if maxl <= T2:
                        r = [36]
                    else:
                        r += [maxj]
                ret += [r]
            return ret

        def geteval(devp, data):
            correct_sys, all_sys = 0, 0
            correct_gt = 0

            if output_error:
                error_list = []
                error_index = []
            for i in range(len(data)):
                for id in data[i]:
                    if id != 36:
                        correct_gt += 1
                        if id in devp[i]:
                            correct_sys += 1
                        else:
                            if output_error and all_indexes[i] not in error_index:
                                error_index.append(all_indexes[i])
                                error_list.append(self.test_file[all_indexes[i]])
                                error_list[-1]['error_pred'] = [self.id2rel[devp_id + 1] for devp_id in devp[i]]
                                error_list[-1]['error_pred_id'] = [devp_id + 1 for devp_id in devp[i]]
                    elif output_error:
                        if 36 not in devp[i] and all_indexes[i] not in error_index:
                            error_index.append(all_indexes[i])
                            error_list.append(self.test_file[all_indexes[i]])
                            error_list[-1]['error_pred'] = [self.id2rel[devp_id + 1] for devp_id in devp[i]]
                            error_list[-1]['error_pred_id'] = [devp_id + 1 for devp_id in devp[i]]

                for id in devp[i]:
                    if id != 36:
                        all_sys += 1

            precision = 1 if all_sys == 0 else correct_sys / all_sys
            recall = 0 if correct_gt == 0 else correct_sys / correct_gt
            f_1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

            if output_error:
                return f_1, error_list
            return f_1

        logits = np.asarray(logits)
        logits = list(1 / (1 + np.exp(-logits)))

        labels = []
        for f in features:
            label = []
            assert (len(f) == 36)
            for i in range(36):
                if f[i] == 1:
                    label += [i]
            if len(label) == 0:
                label = [36]
            labels += [label]
        assert (len(labels) == len(logits))

        bestT2 = bestf_1 = 0
        best_output_error = []
        for T2 in range(51):

            devp = getpred(logits, T2=T2 / 100.)
            if output_error:
                f_1, output_error_list = geteval(devp, labels)
            else:
                f_1 = geteval(devp, labels)
            if f_1 > bestf_1:
                bestf_1 = f_1
                bestT2 = T2 / 100.
                if output_error:
                    best_output_error = output_error_list

        if output_error:
            for item in best_output_error:
                item['word_length'] = item['Ls'][-1]
                item['sents'] = [' '.join(sent) for sent in item['sents']]
                del item['vertexSet']
                del item['h_t_pair_words']
            random.shuffle(best_output_error)
            with open('analysis.txt', 'w+') as file:
                for o_i in range(50):
                    item = best_output_error[o_i]
                    x = item['labels'][0]['x']
                    y = item['labels'][0]['y']
                    r = ';'.join([str(rid) + rel for rid, rel in zip(item['labels'][0]['rid'], item['labels'][0]['r'])])
                    pred = ';'.join([str(rid) + rel for rid, rel in zip(item['error_pred_id'], item['error_pred'])])
                    file.write('\t'.join([x, y, r, pred]) + '\n')

            json.dump(best_output_error, open('error.json', 'w+', encoding='utf-8'), ensure_ascii=False)

        return bestf_1, bestT2

    def test(self, model, model_name, output=False, input_theta=-1, output_error=False):
        data_idx = 0
        eval_start_time = time.time()

        def logging(s, print_=True, log_=True):
            if print_:
                print(s)
            if log_:
                with open(os.path.join(os.path.join("log", model_name)), 'a+') as f_log:
                    f_log.write(s + '\n')

        relations = []
        predictions = []

        all_indexes = []
        for i, data in enumerate(self.get_test_batch()):
            with torch.no_grad():
                context_idxs = data['context_idxs']
                context_pos = data['context_pos']
                h_mapping = data['h_mapping']
                t_mapping = data['t_mapping']
                labels = data['labels']
                relation_multi_label = data['relation_multi_label']
                L_vertex = data['L_vertex']
                context_ner = data['context_ner']
                relation_mask = data['relation_mask']
                ht_pair_pos = data['ht_pair_pos']
                context_masks = data['context_masks']
                context_starts = data['context_starts']
                context_words = data['context_words']
                attention_label_mask = data['attention_label_mask']
                speaker_label = data['speaker_label']

                # titles = data['titles']
                indexes = data['indexes']
                triggers = data['triggers']
                evidences = data['evis']

                context_seg = data['context_seg']

                dis_h_2_t = ht_pair_pos + 10
                dis_t_2_h = -ht_pair_pos + 10

                node_position = data['node_position'].cuda()
                entity_position = data['entity_position'].cuda()
                # node_position_sent = data['node_position_sent']#.cuda()
                node_sent_num = data['node_sent_num'].cuda()
                all_node_num = data['all_node_num'].cuda()
                entity_num = torch.Tensor(data['entity_num']).cuda()
                # sent_num = torch.Tensor(data['sent_num']).cuda()
                sdp_pos = data['sdp_position'].cuda()
                sdp_num = torch.Tensor(data['sdp_num']).cuda()
                # if i==1:
                #     print(i)
                predict_re, attention_weight = model(context_idxs,
                                                     h_mapping, t_mapping, relation_mask,
                                                     node_position, entity_position, node_sent_num,
                                                     entity_num, sdp_pos, sdp_num,
                                                     context_masks, context_starts, attention_label_mask,
                                                     speaker_label)

            # predict_re = torch.sigmoid(predict_re)

            predict_re = predict_re[torch.eq(relation_mask, 1)].data.cpu().numpy().tolist()

            # predict_re = [np.argmax(pre) for pre in predict_re]

            predictions += predict_re
            relations += relation_multi_label[torch.eq(relation_mask, 1)].data.cpu().numpy().tolist()
            if len(indexes) != len(predict_re):
                print('')

            all_indexes += indexes

            data_idx += 1

            if data_idx % self.period == 0:
                print('| step {:3d} | time: {:5.2f}'.format(data_idx // self.period, (time.time() - eval_start_time)))
                eval_start_time = time.time()

        bestf_1, bestT2 = self.f1_eval(predictions, relations, output_error, all_indexes)
        return bestf_1, bestT2

    def testall(self, model_pattern, model_name, input_theta):  # , ignore_input_theta):
        model = model_pattern(config=self)

        model.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, model_name)))
        model.cuda()
        model.eval()
        bestf_1, bestT2 = self.test(model, model_name, False, input_theta, False)
        print('f1:', bestf_1)
