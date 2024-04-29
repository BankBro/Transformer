import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib as plt
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_rate=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # position: (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))  # x = exp(ln(x))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # pe增加一个维度是为了广播 (max_len, d_model) -> (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)  # ???
    
    def forward(self, x):
        # x: (batch_size, src_len / tgt_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
    
    def forward(self, Q, K, V, attn_mask):
        # 输入:
        # Q: (batch_size, n_heads, q_len, d_q)
        # K: (batch_size, n_heads, k_len, d_k)
        # V: (batch_size, n_heads, v_len, d_v)
        # attn_mask: (batch_size, n_heads, q_len, k_len), attn_mask的形状和维度必须和scores相同

        # 求attention scores
        # scores的每一行都是同一个q对所有的k的分数
        # scores: (batch_size, n_heads, q_len, k_len)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)

        # 注意力掩码，将非零部分填充成绝对值很大的负数，形状不变
        scores.masked_fill_(attn_mask, -1e9)
        # dim=-1表示对一行的每列的元素做saftmax，attn的形状和scores相同
        attn = nn.Softmax(dim=-1)(scores)
        # context: (batch_size, n_heads, q_len, d_v)
        context = torch.matmul(attn, V)

        return context, attn

# 多头注意力机制，很复杂也很重要的一个部分
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()

        # 输入进来的QKV是相等的，使用映射Linear做映射得到参数矩阵Wq, Wk, Wv
        self.W_Q = nn.Linear(d_model, d_q * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        # 多头注意力步骤：首先映射分头，然后计算atten_scores, 最后计算atten_value

        # batch的一个样本有一个Q K V
        # Q: (batch_size, q_len, d_model)
        # K: (batch_size, k_len, d_model)
        # V: (batch_size, v_len, d_model)
        assert Q.size()[0] == K.size()[0] == V.size()[0] , 'Q K V batch_size is not eq!'
        
        # 交叉注意力层用的残差是解码器的Q
        residual, batch_size = Q, Q.size()[0]
        
        # 先映射，后分头
        # (B, S, D) -> (B, S, D * H) -> (B, S, H, D) -> (B, H, S, D)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_q).transpose(1, 2)  # q_s: (batch_size, n_heads, q_len, d_q)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # k_s: (batch_size, n_heads, k_len, d_k)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)  # v_s: (batch_size, n_heads, v_len, d_v)

        # 输入attn_mask: (batch_size, q_len, k_len)
        # 输出attn_mask: (batch_size, n_heads, q_len, k_len)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        # 计算scaled dot product attention
        # context: (batch_size, n_heads, q_len, d_v)
        # attn: (batch_size, n_heads, q_len, k_len)
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        
        # context: (batch_size, q_len, n_heads * d_v)
        # 注意这里的矩阵变形方法，转置->内存连续->新形状
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        # output: (batch_size, q_len, d_model)
        output = self.linear(context)

        return self.layer_norm(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, inputs):
        # inputs: (batch_size, src_len / tgt_len, d_model)
        residual = inputs

        output = self.conv1(inputs.transpose(1, 2))
        output = self.relu(output)
        output = self.conv2(output).transpose(1, 2)

        return self.layer_norm(output + residual)  # 保持输入与输出的形状


# encoder layer包含两个部分：多头注意力机制和前馈神经网络
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
    
    def forward(self, enc_inputs, enc_self_attn_mask):
        # 自注意力层
        # 输入:
        # 三个enc_inputs作为原始QKV: (batch_size, src_len, d_model)
        # enc_self_attn_mask: (batch_size, src_len, src_len)
        # 输出：
        # enc_outputs: (batch_size, src_len, d_model)
        # attn: (batch_size, n_heads, src_len, src_len)
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        
        # 前馈网络
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs保持形状
        return enc_outputs, attn


# 为什么要有这个掩码？？？
# k的有pad的位置都被标记为true
# mask的每一行是一个q对每个k求的score
def get_attn_pad_mask(q, k):
    # q: (batch_size, q_len)
    # k: (batch_size, k_len)
    batch_size_q = q.size()[0]
    batch_size_k = k.size()[0]
    assert batch_size_q == batch_size_k, 'batch size of q is not eq to k\'s!'
    batch_size = batch_size_q

    q_len = q.size()[1]
    k_len = k.size()[1]

    # pad_attn_mask: (batch_size, 1, k_len)
    pad_attn_mask = k.data.eq(0).unsqueeze(1)

    # 返回: (batch_size, q_len, k_len)
    return pad_attn_mask.expand(batch_size, q_len, k_len)

# encoder包含三部分：embedding, positional encoding, attention layer和FC
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)  # 词嵌入
        self.pos_emb = PositionalEncoding(d_model)  # 位置编码
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])  # 包括多头注意力和前馈

    def forward(self, enc_inputs):
        # enc_inputs: (batch_size, src_len)

        # embeding, 将enc_inputs的整数索引转为词向量
        # enc_outputs输出形状：(batch_size, src_len, d_model)
        enc_outputs = self.src_emb(enc_inputs)

        # 位置编码，将pos信息加入到batch的每一个样本，enc_outputs形状不变
        enc_outputs = self.pos_emb(enc_outputs)

        # get_attn_pad_mask是为了得到句子中pad的位置信息
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)

        # 自注意力和前馈网络
        enc_self_attns = []  # 收集每轮的attention scores
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        
        return enc_outputs, enc_self_attns

# 包括解码器的自注意力、交叉注意力、前馈神经网络层
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        # decocer自注意力层
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)

        # decoder和encoder交叉注意力层
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)

        # 前馈神经网络
        dec_outputs = self.pos_ffn(dec_outputs)

        return dec_outputs, dec_self_attn, dec_enc_attn

# 解码器计算注意力分数时，可见顺序的掩码（不包括主对角线的上三角形矩阵）
def get_attn_subsequent_mask(seq):
    # seq: (batch_size, tgt_len)
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    
    # 不包括主对角线的上三角形矩阵，后面的流程位置上被标记为1的元素会转为0
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)

    # (batch_size, tgt_len, tgt_len)
    return torch.from_numpy(subsequence_mask).byte()

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])
    
    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        # 词嵌入
        dec_outputs = self.tgt_emb(dec_inputs)
        # 位置编码
        dec_outputs = self.pos_emb(dec_outputs)

        # 解码器自注意力层K的pad标记
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        # 解码器自注意力层的上三角矩阵
        dec_self_attn_subsequent_maks = get_attn_subsequent_mask(dec_inputs)
        # 两个矩阵相加，大于0的为True
        dec_self_attn_mask = dec_self_attn_pad_mask + dec_self_attn_subsequent_maks
        dec_self_attn_mask = torch.gt(dec_self_attn_mask, 0)

        # 交互注意力层的mask，q来自解码器，k和v来自编码器
        # dec_enc_attn_mask: (batch_size, tgt_len, src_len)
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = \
                layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)

            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        
        return dec_outputs, dec_self_attns, dec_enc_attns

# 3 layers: encoder, decoder, output layer
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)
    
    def forward(self, enc_inputs, dec_inputs):
        # transformer有两个输入
        # 一个是enc_inputs (batch_size, src_len)，作为编码器的输入
        # 一个是dec_inputs (batch_size, tgt_len)，作为解码器的输入

        # encoder 的主要输出是enc_outputs，
        # 另一个输出enc_self_attns是attenstion score, Q K相乘经softmax后的矩阵
        # enc_outputs: (batch_size, src_len, d_modle)
        # enc_self_attns: [(batch_size, n_heads, src_len, src_len)]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)

        # decoder的主要输出是dec_outputs，用于后续linear映射，(batch_size, tgt_len, d_modle)
        # dec_self_attns是decoder输入的自注意力分数，(batch_size, n_heads, tgt_len, tgt_len)
        # dec_enc_attns是decoder中每个单词对encoder的交叉注意力分数，(batch_size, n_heads, tgt_len, src_len)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)

        # dec_outputs做映射到词表大小
        # dec_logits: (batch_size, tgt_len, tgt_vocab_size)，最后一个维度表示每个位置预测词表中每个词的概率
        dec_logits = self.projection(dec_outputs)
        
        # dec_logits: (batch_size * tgt_len, tgt_vocab_size)
        return dec_logits.view(-1, dec_logits.size()[-1]), enc_self_attns, dec_self_attns, dec_enc_attns

def make_batch(sentences):
    input_batch = np.array([src_vocab[voc] for sentence in sentences[:, 0]
        for voc in sentence.split()]).reshape(-1, src_len)

    output_batch = np.array([tgt_vocab[voc] for sentence in sentences[:, 1]
        for voc in sentence.split()]).reshape(-1, tgt_len)

    target_batch = np.array([tgt_vocab[voc] for sentence in sentences[:, 2]
        for voc in sentence.split()]).reshape(-1, tgt_len)

    # (batch_size, src_len / tgt_len)
    return torch.LongTensor(input_batch), torch.LongTensor(output_batch), torch.LongTensor(target_batch)


if __name__ == '__main__':
    # src:德语 -> tgt:英语
    # 句子输入：encoder输入，decoder输入，最终输出的label
    # sentences = np.array([['ich mochte ein bier P', 'S i want a beer', 'i want a beer E'],
    #                       ['ich liebe dich P P',    'S i love you P',  'i love you E P']])
    sentences = np.array([['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']])
    
    # Transformer Parameters
    # Padding Should be Zero
    ## 构建德语词表（德转英）
    src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4, 'liebe': 5, 'dich': 6}
    src_vocab_size = len(src_vocab)

    tgt_vocab = {'P': 0, 'S': 1, 'E': 2, 'i': 3, 'want': 4, 'a': 5, 'beer': 6, 'love':7, 'you': 8}
    tgt_vocab_size = len(tgt_vocab)

    src_len = 5 # length of source，encoder输入的句子最长长度
    tgt_len = 5 # length of target

    ## 模型参数
    d_model = 512         # embedding size
    d_ff = 2048           # feed forward dimension
    d_k = d_q = d_v = 64  # dimension of K(=Q) V
    n_layers = 6          # number of encoder or decoder layer
    n_heads = 8           # number of heads in multi-head attention


    model = Transformer()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    enc_inputs, dec_inputs, target_batch = make_batch(sentences)

    for epoch in range(50):
        optimizer.zero_grad()
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = \
            model(enc_inputs, dec_inputs)
        
        loss = criterion(outputs, target_batch.contiguous().view(-1))
        print(f'Epoch: {epoch + 1} | Loss: {loss:.4f}')
        loss.backward()
        optimizer.step()

    tgt_key_list = list(tgt_vocab.keys())
    pred, _, _, _ = model(enc_inputs, dec_inputs)
    pred = [tgt_key_list[i.item()] for i in pred.max(1)[1]]
    print(pred)