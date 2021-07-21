import datetime
import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from aggregator import LocalAggregator, GlobalAggregator
from torch.nn import Module, Parameter
import torch.nn.functional as F


class CombineGraph(Module):
    def __init__(self, opt, num_node, adj_all, num):
        super(CombineGraph, self).__init__()
        self.opt = opt

        self.batch_size = opt.batch_size
        self.num_node = num_node
        self.dim = opt.hiddenSize
        self.dropout_local = opt.dropout_local
        self.dropout_global = opt.dropout_global
        self.hop = opt.n_iter
        self.sample_num = opt.n_sample
        self.adj_all = trans_to_cuda(torch.Tensor(adj_all)).long()
        self.num = trans_to_cuda(torch.Tensor(num)).float()
        

        # Aggregator
        self.local_agg = LocalAggregator(self.dim, self.opt.alpha, dropout=opt.long_edge_dropout, hop=opt.hop)
        self.global_agg = []
        for i in range(self.hop):
            if opt.activate == 'relu':
                agg = GlobalAggregator(self.dim, opt.dropout_gcn, act=torch.relu)
            else:
                agg = GlobalAggregator(self.dim, opt.dropout_gcn, act=torch.tanh)
            self.add_module('agg_gcn_{}'.format(i), agg)
            self.global_agg.append(agg)

        # Item representation & Position representation
        self.embedding = nn.Embedding(num_node, self.dim)
        
        self.pos_emb = nn.Parameter(torch.Tensor(opt.pos_num, opt.pos_emb_len, self.dim))
        self.mine_w_1 = nn.Parameter(torch.Tensor(1, opt.pos_emb_len))
        self.mine_q_1 = nn.Parameter(torch.Tensor(1, opt.pos_emb_len))
        
        self.Q = nn.Parameter(torch.Tensor(self.dim+1, self.dim))
        self.P = nn.Parameter(torch.Tensor(self.dim, opt.pos_num))

        # Parameters
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.glu1 = nn.Linear(self.dim, self.dim)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=False)
        self.linear_transform = nn.Linear(self.dim, self.dim, bias=False)

        self.leakyrelu = nn.LeakyReLU(opt.alpha)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        self.reset_parameters()
        
        
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def sample(self, target, n_sample):
        # neighbor = self.adj_all[target.view(-1)]
        # index = np.arange(neighbor.shape[1])
        # np.random.shuffle(index)
        # index = index[:n_sample]
        # return self.adj_all[target.view(-1)][:, index], self.num[target.view(-1)][:, index]
        return self.adj_all[target.view(-1)], self.num[target.view(-1)]

    def compute_scores(self, hidden, mask, inputs, epoch):
        mask = mask.float().unsqueeze(-1)

        batch_size = hidden.shape[0]
        len = hidden.shape[1]

        hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
        
        '''(1)
        key = torch.matmul(self.mine_w_1, self.pos_emb)
        #query = hs.unsqueeze(-2).unsqueeze(-2)
        
        query = (torch.sum(self.embedding(inputs) * mask, -2) / torch.sum(mask, 1)).unsqueeze(-2).unsqueeze(-2)
        e = torch.matmul(query,key.transpose(-2,-1))
        gama = torch.softmax(self.leakyrelu(e), 1)
        pos_emb = (gama * self.pos_emb).sum(1)
        pos_emb = pos_emb[:,:len,:]
        self.gama = gama
        '''
        '''(2)
        pos_emb = self.pos_emb[:, :len, :].unsqueeze(0).repeat(batch_size, 1, 1, 1)
        #h = hidden.unsqueeze(1).repeat(1, self.opt.pos_num, 1, 1)
        h = hidden.unsqueeze(1).repeat(1, self.opt.pos_num, 1, 1)
        key = torch.cosine_similarity(pos_emb, h, dim=-1).unsqueeze(-1) 
        query = self.mine_q_1[:, :len]
        e = torch.matmul(query, key)
        gama = torch.softmax(self.leakyrelu(e) * min(0.5 * pow(20 / 0.5, epoch / self.opt.E), 20), 1)        
        pai = gama * pos_emb
        pos_emb = pai.sum(1)
        l2 = pai.pow(2).sum(-1).sum(-1).pow(0.5).sum(1) / pos_emb.pow(2).sum(-1).sum(-1).pow(0.5)
        pos_emb = l2.view(batch_size, 1, 1) * pos_emb
        self.gama = gama
        '''
        '''(3)'''
        pos_emb = self.pos_emb[:, :len, :].unsqueeze(0).repeat(batch_size, 1, 1, 1)
        h = torch.matmul(self.leakyrelu(torch.matmul(torch.cat((hs, torch.log2(mask.squeeze(-1).sum(-1).unsqueeze(-1))), -1), self.Q)), self.P)
        gama = torch.softmax(h * min(0.5 * pow(2 / 0.5, epoch / self.opt.E), 2), 1).view(batch_size, self.opt.pos_num, 1, 1)
        pai = gama * pos_emb
        pos_emb = pai.sum(1)
        l2 = (pai).pow(2).sum(-1).sum(-1).pow(0.5).sum(-1) / (pos_emb).pow(2).sum(-1).sum(-1).pow(0.5)
        pos_emb = l2.view(batch_size, 1, 1) * pos_emb
        self.gama = gama
        
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * hidden, 1)

        b = self.embedding.weight[1:]  # n_nodes x latent_size
        scores = torch.matmul(select, b.transpose(1, 0))
        return scores

    def forward(self, inputs, adj, mask_item, item):
        batch_size = inputs.shape[0]
        seqs_len = inputs.shape[1]
        h = self.embedding(inputs)

        # local
        h_local = self.local_agg(h, adj, mask_item)

        # global
        
        # combine
        h_local = F.dropout(h_local, self.dropout_local, training=self.training)
        
        output = h_local 

        return output


def SSL(sess_emb_hgnn, sess_emb_lgcn):
    def row_shuffle(embedding):
        corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
        return corrupted_embedding

    def row_column_shuffle(embedding):
        corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
        corrupted_embedding = corrupted_embedding[:, torch.randperm(corrupted_embedding.size()[1])]
        return corrupted_embedding

    def score(x1, x2):
        return torch.sum(torch.mul(x1, x2), 1)

    pos = score(sess_emb_hgnn, sess_emb_lgcn)
    neg1 = score(sess_emb_lgcn, row_column_shuffle(sess_emb_hgnn))
    one = torch.cuda.FloatTensor(neg1.shape[0], neg1.shape[1]).fill_(1)
    # one = zeros = torch.ones(neg1.shape[0])
    con_loss = torch.sum(-torch.log(1e-8 + torch.sigmoid(pos)) - torch.log(1e-8 + (one - torch.sigmoid(neg1))))
    return con_loss

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, data, epoch):
    alias_inputs, adj, items, mask, targets, inputs = data
    alias_inputs = trans_to_cuda(alias_inputs).long()
    items = trans_to_cuda(items).long()
    adj = trans_to_cuda(adj).float()
    mask = trans_to_cuda(mask).long()
    inputs = trans_to_cuda(inputs).long()

    hidden = model(items, adj, mask, inputs)
    get = lambda index: hidden[index][alias_inputs[index]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    return targets, model.compute_scores(seq_hidden, mask, inputs, epoch)


def train_test(model, train_data, test_data, epoch):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=model.batch_size,
                                               shuffle=True, pin_memory=True)
    for data in tqdm(train_loader):
        model.optimizer.zero_grad()
        targets, scores = forward(model, data, epoch)
        targets = trans_to_cuda(targets).long()
        loss = model.loss_function(scores, targets - 1) 
        loss.backward()
        model.optimizer.step()
        total_loss += loss
    print('\tLoss:\t%.3f' % total_loss)
    model.scheduler.step()

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=model.batch_size,
                                              shuffle=False, pin_memory=True)
    result = []
    hit, mrr, hit_alias, mrr_alias = [], [], [], []
    for data in test_loader:
        targets, scores = forward(model, data, epoch)
        sub_scores = scores.topk(20)[1]
        sub_scores_alias = scores.topk(10)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        sub_scores_alias = trans_to_cpu(sub_scores_alias).detach().numpy()
        targets = targets.numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            #@20
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
                
        for score, target, mask in zip(sub_scores_alias, targets, test_data.mask):
            #@10
            hit_alias.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr_alias.append(0)
            else:
                mrr_alias.append(1 / (np.where(score == target - 1)[0][0] + 1))
            

    result.append(np.mean(hit) * 100)
    result.append(np.mean(mrr) * 100)
    
    result.append(np.mean(hit_alias) * 100)
    result.append(np.mean(mrr_alias) * 100)
    return result
