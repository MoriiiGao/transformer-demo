"""
输入word_seq->id2vec->add_position
"""
import torch
import torch.nn as nn
from dataset import de_vocab, de_preprocess, train_dataset
import math

class EmbeddingWithPosition(nn.Module):
    def __init__(self,
                 vocab_size,
                 emb_size,
                 dropout=0.1,
                 seq_max_len=500):
        """
        带有位置信息的词嵌入
        位置信息不参与学习
        :param vocab_size:
        :param emb_size:
        :param dropout:
        :param seq_max_len:
        """
        super().__init__()

        # 定义词向量
        self.emb = nn.Embedding(vocab_size, emb_size)
        # 为tokens中的每个位置准备一个位置向量，宽为emb_size
        position_idx = torch.arange(0,
                                    seq_max_len,
                                    dtype=torch.float).unsqueeze(-1)
        # 扩充位置向量维度
        position_emb_fill = position_idx * torch.exp(
            -torch.arange(0, emb_size, 2) * math.log(10000.0) / emb_size)
        # 位置编码
        pos_encoding = torch.zeros(seq_max_len, emb_size)
        pos_encoding[:, 0::2] = torch.sin(position_emb_fill)
        pos_encoding[:, 1::2] = torch.cos(position_emb_fill)
        self.register_buffer('pos_encoding', pos_encoding)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """

        :param x: [batch_size, seq_len]
        :return:
        """

        # [batch_size, seq_len] -> [batch_size, seq_len, emb_size]
        x = self.emb(x)
        # [batch_size, seq_len, emb_size]
        x = x + self.pos_encoding.unsqueeze(0)[:, :x.size()[1], :]
        return self.dropout(x)

if __name__=='__main__':
    emb = EmbeddingWithPosition(len(de_vocab), 128)

    de_tokens, de_ids = de_preprocess(train_dataset[0][0]) # 取de句子转词ID序列
    de_ids_tensor = torch.tensor(de_ids, dtype=torch.long)

    emb_result = emb(de_ids_tensor.unsqueeze(0)) # 转batch再输入模型
    print('de_ids_tensor:', de_ids_tensor.size(), 'emb_result:', emb_result.size())
