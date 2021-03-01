import pdb
import jieba
from time import time
from utils import get_config


def main():
    config_dc = get_config.get_config()     # 提取超参、地址等信息
    dir_raw = config_dc['dir_raw']  # 原始语料存储位置
    dir_QA = config_dc['dir_qa']    # 处理后的问答语料存储位置
    ques, ans = [], []
    with open(dir_raw, encoding='utf-8') as file_in:
        print('---提取问答语料')
        for line in file_in:
            line = line.strip()
            line = line.split()
            if not line:
                continue
            if line[0] != 'M':
                continue
            new_line = ''
            for word in line[1: ]:
                new_line += word
            if len(ques) > len(ans):
                ans.append(new_line)
            else:
                ques.append(new_line)


    assert len(ques) == len(ans)    # 问句和答句的行数需要是一致的
    print('---对问答进行分词')
    start = time()
    for i in range(len(ques)):
        if i and i % 100000 == 0:
            print(f'处理进度 {i}/{len(ques)}行')
        ques_i, ans_i = ques[i], ans[i]
        ques_i, ans_i = '/'.join(jieba.lcut(ques_i)), '/'.join(jieba.lcut(ans_i))
        ques[i], ans[i] = ques_i, ans_i
    print(f'用时 {time() - start}s')


    with open(dir_QA, 'w', encoding='utf-8') as file_out:
        print('---保存问答数据')
        for ques_i, ans_i in zip(ques, ans):
            if ques_i == '' or ans_i == '':
                continue    # 剔除无效对话
            if len(ques_i.split('/')) > 15 or len(ans_i.split('/')) > 15:
                continue    # 剔除超长句子

            file_out.write(ques_i)
            file_out.write('\t')    # 问句答句以'\t'分隔
            file_out.write(ans_i)
            file_out.write('\n')


if __name__ == '__main__':
    main()