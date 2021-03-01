from utils import get_config
from model import *
import torch
import pdb
from time import time
import jieba

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
configs = get_config()  # 提取超参、地址等信息


class Lan:
    # language
    # 记录单词和索引的几个dict
    def __init__(self, name):
        self.name = name
        self.word2index = {"SOS": configs['sos_idx'], "EOS": configs['eos_idx']}    # 0, 1
        self.index2word = {configs['sos_idx']: "SOS", configs['eos_idx']: "EOS"}    # 0, 1
        self.word2count = {"SOS": 0, "EOS": 0}
        self.num_words = 2  # Count SOS and EOS

    def add_word(self, word):
        if word not in self.word2index:
            self.index2word[self.num_words] = word
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def read_sentence(self, sentence):
        for word in sentence.split('/'):
            self.add_word(word)


def create_dataset(path):
    with open(path, encoding='utf-8') as file_in:
        Q_A = []
        for line in file_in:
            line = line.strip()
            assert len(line.split('\t')) == 2
            ques, ans = line.split('\t')[0], line.split('\t')[1]
            ques, ans = f'SOS/{ques}/EOS', f'SOS/{ans}/EOS'
            Q_A.append([ques, ans])
        # 组织成问答格式，shape=(num_pairs, 2)，并在每句的前后加上SOS、EOS
        ques_lan = Lan('ques')
        ans_lan = Lan('ans')
        for pair in Q_A:
            ques_lan.read_sentence(pair[0])
            ans_lan.read_sentence(pair[1])
    return ques_lan, ans_lan, Q_A


def create_tensor(ques_lan, ans_lan, Q_A):
    def get_tensor_from_sentence(sentence, lan):
        sentence = sentence.split('/')
        idx = [lan.word2index[word] for word in sentence]
        return torch.tensor(idx).to(device)

    ques_tensors, ans_tensors = [], []
    for pair in Q_A:
        ques, ans = pair[0], pair[1]
        ques_tensor, ans_tensor = get_tensor_from_sentence(ques, ques_lan), get_tensor_from_sentence(ans, ans_lan)
        ques_tensors.append(ques_tensor), ans_tensors.append(ans_tensor)
    return ques_tensors, ans_tensors


ques_lan, ans_lan, Q_A = create_dataset(path=configs['dir_qa'])

def train():
    print('---training')
    num_epoch = configs['num_epoch']

    print('提取tensor')
    ques_lan, ans_lan, Q_A = create_dataset(path=configs['dir_qa'])
    ques_tensors, ans_tensors = create_tensor(ques_lan, ans_lan, Q_A)

    assert len(ques_tensors) == len(ans_tensors)    # list of tensor.

    encoder = Encoder(
        ques_lan.num_words, configs['dim_hidden']
    ).to(device)
    decoder = Decoder(
        configs['dim_hidden'], ans_lan.num_words,
        max_length=configs['max_length'],
        dropout_prob=configs['dropout_prob']
    ).to(device)

    print('开始训练')
    for epoch in range(num_epoch):
        print(f'epoch {epoch}/{num_epoch}')
        start = time()
        epoch_loss = 0
        num_lines = min(configs['using_train_data'], len(ques_tensors))
        for i in range(num_lines):
            if i % 1000 == 0:
                if i:
                    print(f'当前进度 {int(i/1000)}k/{num_lines}, 处理1k数据所花时间 {int(time() - time_k)}s')
                    print(f'1k数据平均loss {(epoch_loss - loss_k) / 1000}')
                    torch.save({
                        'encoder': encoder.state_dict(),
                        'decoder': decoder.state_dict()
                    }, configs['dir_model'])
                    print('成功更新模型参数')
                time_k = time()
                loss_k = epoch_loss
            ques_tensor, ans_tensor = ques_tensors[i], ans_tensors[i]
            loss = train_step(ques_tensor, ans_tensor, encoder, decoder)
            epoch_loss += loss
        print(f'本epoch所花时间 {int(time() - start)}s，本epoch每句平均loss {epoch_loss / num_lines}')
    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict()
    }, configs['dir_model'])
    print('成功更新模型参数')


def predict(ques):
    print(f'---predict \'{ques}\'')
    encoder = Encoder(
        ques_lan.num_words, configs['dim_hidden']
    ).to(device)
    decoder = Decoder(
        configs['dim_hidden'], ans_lan.num_words,
        max_length=configs['max_length'],
        dropout_prob=configs['dropout_prob']
    ).to(device)
    model_ckpt = torch.load(configs['dir_model'], map_location='cpu')   # 加载模型时加载到cpu上，否则无法运行
    encoder.load_state_dict(model_ckpt['encoder'])
    decoder.load_state_dict(model_ckpt['decoder'])  # 加载模型参数

    ques, _ = '/'.join(jieba.lcut(ques)), 'SOS/EOS'
    ques, _ = f'SOS/{ques}/EOS', 'SOS/EOS'
    Q_A = [[ques, _]]
    ques_tensor, _ = create_tensor(ques_lan, ans_lan, Q_A)
    ques_tensor = ques_tensor[0]
    ans_tensor = predict_step(ques_tensor, encoder, decoder)
    ans = []
    for i in ans_tensor:
        word = ans_lan.index2word[i.item()]
        ans.append(word)
    str = ''
    for i in ans[1: -1]:
        str += i
    return str


def main():
    print(f'---运行环境 {device}')

    # train()   # 如果已经有模型就别训练了
    while True:
        print('请输入您要发送的信息')
        ques = input()
        ans = predict(ques)
        print(f'预测结果 {ans}')

if __name__ == '__main__':
    main()