from utils.get_config import get_config

import pdb
from random import random
import torch
from torch import nn
import torch.nn.functional as F

configs = get_config()  # 提取超参、地址等信息
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GRU用法
# GRU(input_size, hidden_size, num_layers)
# input(seq_len, batch_size, hidden_size)
# hidden(num_layers*num_direction, batch_size, hidden_size)

class Encoder(nn.Module):
    def __init__(self, dim_input, dim_hidden):
        '''

        :param dim_input: ques_lan.num_words = 63624
        :param dim_hidden: configs['dim_hidden'] = 256
        '''
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=dim_input, embedding_dim=dim_hidden)
        self.gru = nn.GRU(input_size=dim_hidden, hidden_size=dim_hidden)

    def forward(self, word_idx, hidden):
        # 每次输入一个单词的idx
        # (), (1, 1, 256)
        embed = self.embedding(word_idx).view(1, 1, -1)     # (1, 1, 256)
        output, hidden = self.gru(embed, hidden)
        # (1, 1, 256), (1, 1, 256)
        return output, hidden


class Decoder(nn.Module):
    def __init__(self, dim_hidden, dim_output, max_length, dropout_prob):
        '''

        :param dim_hidden: configs['dim_hidden'] = 256
        :param dim_output: ans_lan.num_words = 65347
        :param max_length: 生成句子的最大长度
        '''

        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=dim_output, embedding_dim=dim_hidden)
        self.gru = nn.GRU(input_size=dim_hidden, hidden_size=dim_hidden)
        self.out = nn.Linear(dim_hidden, dim_output)
        self.attn = nn.Linear(dim_hidden*2, max_length)     # ?
        self.attn_combine = nn.Linear(dim_hidden*2, dim_hidden)     # ?
        self.dropout = nn.Dropout(dropout_prob)     # ?
        # GRU(input_size, hidden_size, num_layers)
        # input(seq_len, batch_size, hidden_size)
        # hidden(num_layers*num_direction, batch_size, hidden_size)

    def forward(self, word_idx, hidden, encoder_outputs):
        embed = self.embedding(word_idx).view(1, 1, -1)    # (1, 1) --> (1, 1, 256)
        embed = self.dropout(embed)
        attn_weights = F.softmax(
            self.attn(
                torch.cat(
                    (embed, hidden), dim=2  # (1, 1, 256), (1, 1, 256)
                )  # (1, 1, 512)
            ), dim=2  # (1, 1, 200)
        )  # (1, 1, 200)
        attn_applied = torch.bmm(attn_weights, encoder_outputs)
        # (1, 1, 256) = (1, 1, 200) * (1, 200, 256)

        att_input = torch.cat((embed, attn_applied), dim=2)
        # (1, 1, 512)
        att_input = self.attn_combine(att_input)
        # (1, 1, 256)
        att_input = F.relu(att_input)


        output, hidden = self.gru(att_input, hidden)
        # (1, 1, 256), (1, 1, 256)

        output = F.log_softmax(self.out(output), dim=2)  # (1, 1, 65347)
        return output, hidden


def train_step(ques_tensor, ans_tensor, encoder, decoder):
    optimizer_enc = torch.optim.SGD(encoder.parameters(), lr=configs['lr'])
    optimizer_dec = torch.optim.SGD(decoder.parameters(), lr=configs['lr'])
    optimizer_enc.zero_grad(), optimizer_dec.zero_grad()

    criterion = nn.NLLLoss()
    loss = 0.


    encoder_outputs = torch.zeros(1, configs['max_length'], configs['dim_hidden']).to(device)  # ?
    # (1, 200, 256)
    hidden = torch.zeros(1, 1, configs['dim_hidden']).to(device)   # batch_size=1
    for i in range(len(ques_tensor)):
        output, hidden = encoder(ques_tensor[i], hidden)
        # shape = (1, 1, 256)
        encoder_outputs[0][i] = output[0][0]

    hidden = hidden     # 代表encoder --> decoder
    teacher_forcing = True if random() < configs['teacher_forcing_prob'] else False     # 对半分
    if teacher_forcing:
        # 老师领着做（不管输出什么，下一步的输入是GT）
        for i in range(len(ans_tensor) - 1):    # 因为串行输入，最后一个（'EOS'）不需要输入
            output, hidden = decoder(ans_tensor[i], hidden, encoder_outputs)    # 使用GT，不使用decoder.input
            loss += criterion(output[0], ans_tensor[i+1].view(1))

    else:
        input = torch.tensor([configs['sos_idx']]).to(device)
        for i in range(len(ans_tensor) - 1):
            output, hidden = decoder(input, hidden, encoder_outputs)
            loss += criterion(output[0], ans_tensor[i+1].view(1))
            top_v, top_i = output.topk(k=1)  # values, indices.
            input = top_i
            if input.item() == configs['eos_idx']:
                break   # 则结束

    loss.backward()
    optimizer_enc.step()
    optimizer_dec.step()

    return loss.item() / len(ans_tensor)

def predict_step(ques_tensor, encoder, decoder):
    encoder_outputs = torch.zeros(1, configs['max_length'], configs['dim_hidden']).to(device)  # ?
    # (1, 200, 256)
    hidden = torch.zeros(1, 1, configs['dim_hidden']).to(device)   # batch_size=1
    for i in range(len(ques_tensor)):
        output, hidden = encoder(ques_tensor[i], hidden)
        # shape = (1, 1, 256)
        encoder_outputs[0][i] = output[0][0]

    hidden = hidden     # 代表encoder --> decoder
    input = torch.tensor([configs['sos_idx']]).to(device)
    ans = [configs['sos_idx']]
    for i in range(configs['max_length']):
        output, hidden = decoder(input, hidden, encoder_outputs)
        top_v, top_i = output.topk(k=1)  # values, indices.
        input = top_i
        ans.append(input.item())
        if input.item() == configs['eos_idx']:
            break   # 则结束
    ans_tensor = torch.tensor(ans)
    return ans_tensor