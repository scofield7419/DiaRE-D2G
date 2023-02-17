import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from models.encoder import Encoder
from models.attention import SelfAttention, MultiHeadedAttention
from models.reasoner import DynamicReasoner
from models.reasoner import StructInduction
from models.arc import GraphConvLayer, MultiGraphConvLayer
from pytorch_transformers import *


path = "./modeling_bert"

class DiaRED2G(nn.Module):
    def __init__(self, config):
        super(DiaRED2G, self).__init__()
        self.config = config
        modelConfig = BertConfig.from_pretrained(path + "/" + "bert-base-uncased-config.json")
        self.bert1 = BertModel.from_pretrained(
            path + "/" + 'bert-base-uncased-pytorch_model.bin', config=modelConfig)

        hidden_size = config.rnn_hidden

        bert_hidden_size = 768
        speaker_hidden_size = 16
        if self.config.use_spemb:
            self.speaker_emb = nn.Embedding(10, speaker_hidden_size)
            self.rel_emb = nn.Embedding(config.relation_num-1, bert_hidden_size + speaker_hidden_size)
            self.linear_bert_re = nn.Linear(bert_hidden_size + speaker_hidden_size, hidden_size)
            self.linear_context = nn.Linear(bert_hidden_size + speaker_hidden_size, hidden_size)
            self.multi_att = MultiHeadedAttention(16, bert_hidden_size + speaker_hidden_size)
        else:
            self.linear_bert_re = nn.Linear(bert_hidden_size, hidden_size)
            self.linear_context = nn.Linear(bert_hidden_size, hidden_size)
            self.rel_emb = nn.Embedding(config.relation_num-1, bert_hidden_size)
            self.multi_att = MultiHeadedAttention(16, bert_hidden_size)

        self.self_att = SelfAttention(hidden_size)

        self.bili = torch.nn.Bilinear(hidden_size,  hidden_size, hidden_size)

        self.linear_output = nn.Linear(2 * hidden_size, config.relation_num-1)

        self.relu = nn.ReLU()
        self.dropout_rate = nn.Dropout(config.dropout_rate)
        self.hidden_size = hidden_size

        self.dropout_arc = nn.Dropout(config.dropout_arc)
        if config.use_arc:
            self.arc_head = 16
            self.arc_layer = MultiGraphConvLayer(hidden_size, 1, self.arc_head, self.dropout_arc)



    def forward(self, context_idxs,
                h_mapping, t_mapping,
                relation_mask, mention_node_position, entity_position,
                mention_node_sent_num, entity_num_list, sdp_pos, sdp_num_list,
                context_masks, context_starts, attention_label_mask,
                speaker_label):
       

        context_output1 = self.bert1(context_idxs, attention_mask=context_masks)[0]
        context_output = [layer[starts.nonzero().squeeze(1)]
                          for layer, starts in zip(context_output1, context_starts)]
        del context_output1
        context_output = pad_sequence(context_output, batch_first=True, padding_value=-1)
        max_doc_len = context_output.shape[1]
        rel_embedding = self.rel_emb(torch.tensor([i for i in range(36)]).cuda())
        rel_embedding = rel_embedding.unsqueeze(0).expand(context_output.shape[0],-1,-1)

        speaker_label = speaker_label[:, :max_doc_len]
        if self.config.use_spemb:
            speaker_emb = self.speaker_emb(speaker_label)
            context_output = torch.cat([context_output, speaker_emb], dim=-1)
        if self.config.use_wratt:

            h_t_query, attn = self.multi_att(context_output, rel_embedding, rel_embedding, mask=attention_label_mask)
            lsr_input = self.linear_bert_re(h_t_query)
            context_output = self.linear_context(context_output)
        else:
            context_output = self.linear_context(context_output)
            lsr_input = context_output
            attn = torch.zeros(context_output.shape[0],16,context_output.shape[1],36)

        if self.config.use_arc:
            '''extract Mention node representations'''
            mention_num_list = torch.sum(mention_node_sent_num, dim=1).long().tolist()
            max_mention_num = max(mention_num_list)
            mentions_rep = torch.bmm(mention_node_position[:, :max_mention_num, :max_doc_len], lsr_input) # mentions rep
            '''extract MDP(meta dependency paths) node representations'''
            sdp_num_list = sdp_num_list.long().tolist()
            max_sdp_num = max(sdp_num_list)
            sdp_rep = torch.bmm(sdp_pos[:,:max_sdp_num, :max_doc_len], lsr_input)
            '''extract Entity node representations'''
            entity_rep = torch.bmm(entity_position[:,:,:max_doc_len], lsr_input)
            '''concatenate all nodes of an instance'''
            arc_inputs = []
            all_node_num_batch = []
            for batch_no, (m_n, e_n, s_n) in enumerate(zip(mention_num_list, entity_num_list.long().tolist(), sdp_num_list)):
                m_rep = mentions_rep[batch_no][:m_n]
                e_rep = entity_rep[batch_no][:e_n]
                s_rep = sdp_rep[batch_no][:s_n]
                arc_inputs.append(torch.cat((m_rep, e_rep, s_rep),dim=0))
                node_num = m_n + e_n + s_n
                all_node_num_batch.append(node_num)

            arc_inputs = pad_sequence(arc_inputs).permute(1, 0, 2)
            output = arc_inputs

            adj_matrix = torch.zeros(output.shape[0], output.shape[1], output.shape[1], self.arc_head).cuda()
            for ni, node_num in enumerate(all_node_num_batch):
                adj_matrix[ni, :node_num, :node_num, :] = 1
            output = self.arc_layer(adj_matrix, output)
            mention_node_position = mention_node_position.permute(0, 2, 1)
            output = torch.bmm(mention_node_position[:, :max_doc_len, :max_mention_num], output[:, :max_mention_num])
            context_output = torch.add(context_output, output)
        start_re_output = torch.matmul(h_mapping[:, :, :max_doc_len], context_output) # aggregation
        end_re_output = torch.matmul(t_mapping[:, :, :max_doc_len], context_output) # aggregation

        re_rep = self.dropout_rate(self.relu(self.bili(start_re_output, end_re_output)))

        re_rep = self.self_att(re_rep, re_rep, relation_mask)
        return self.linear_output(re_rep), attn

