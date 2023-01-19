# Designer:Pan YuDong
# Coder:God's hand
# Time:2022/3/29 19:17
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
class CELoss_Marginal_Smooth(nn.Module):

    def __init__(self, class_num, alpha=0.4, stimulus_type='12'):
        super(CELoss_Marginal_Smooth, self).__init__()
        self.class_num = class_num
        self.alpha = alpha
        self.stimulus_matrix_4 = [[0, 1],
                                  [2, 3]]

        self.stimulus_matrix_12 = [[0, 1, 2, 3],
                                   [4, 5, 6, 7],
                                   [8, 9, 10, 11]]

        self.stimulus_matrix_40 = [[0, 1, 2, 3, 4, 5, 6, 7],
                                   [8, 9, 10, 11, 12, 13, 14, 15],
                                   [16, 17, 18, 19, 20, 21, 22, 23],
                                   [24, 25, 26, 27, 28, 29, 30, 31],
                                   [32, 33, 34, 35, 36, 37, 38, 39]]

        if stimulus_type == '4':
            self.stimulus_matrix = self.stimulus_matrix_4

        elif stimulus_type == '12':
            self.stimulus_matrix = self.stimulus_matrix_12

        elif stimulus_type == '40':
            self.stimulus_matrix = self.stimulus_matrix_40

        self.rows = len(self.stimulus_matrix[:])
        self.cols = len(self.stimulus_matrix[0])


        self.attention_lst = [[1.0 / (int(0 <= (i // self.cols - 1) <= self.rows - 1) +
                                      int(0 <= (i // self.cols + 1) <= self.rows - 1) +
                                      int(0 <= (i % self.cols - 1) <= self.cols - 1) +
                                      int(0 <= (i % self.cols + 1) <= self.cols - 1) +
                          int(0 <= (i // self.cols - 1) <= self.rows - 1 and 0 <= i % self.cols - 1 <= self.cols - 1) +
                          int(0 <= (i // self.cols - 1) <= self.rows - 1 and 0 <= i % self.cols + 1 <= self.cols - 1) +
                          int(0 <= (i // self.cols + 1) <= self.rows - 1 and 0 <= i % self.cols - 1 <= self.cols - 1) +
                          int(0 <= (i // self.cols + 1) <= self.rows - 1 and 0 <= i % self.cols + 1 <= self.cols - 1))
                               for j in range(class_num)] for i in range(class_num)]


        # print("att_lst:", self.attention_lst)

    def forward(self, outputs, targets):
        '''
        :param outputs: predictive results,shape:(batch_size, class_num)
        :param targets: ground truth,shape:(batch_size, )
        :return:
        '''
        targets_data = targets.cpu().data
        smoothed_labels = torch.empty(size=(outputs.shape[0], outputs.shape[1]))
        for i in range(smoothed_labels.shape[0]):
            label = targets_data[i]
            for j in range(smoothed_labels.shape[1]):
                if j == label:
                    smoothed_labels[i][j] = 1.0
                else:
                    smoothed_labels[i][j] = self.attention_lst[label][j]
        smoothed_labels = smoothed_labels.to(outputs.device)

        log_prob = F.log_softmax(outputs, dim=1)
        att_loss = - torch.sum(log_prob * smoothed_labels) / outputs.size(-2)
        ce_loss = nn.CrossEntropyLoss()(outputs, targets)
        loss_add = self.alpha * att_loss + (1 - self.alpha) * ce_loss
        return loss_add



class Data2DataCrossEntropyLoss(torch.nn.Module):
    def __init__(self, num_classes, temperature, m_p, device):
        super(Data2DataCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.m_p = m_p
        self.device = device
        self.calculate_similarity_matrix = self._calculate_similarity_matrix()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def _calculate_similarity_matrix(self):
        return self._cosine_simililarity_matrix

    def _cosine_simililarity_matrix(self, x, y):
        v = self.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def make_index_matrix(self, labels):
        labels = labels.detach().cpu().numpy()
        num_samples = labels.shape[0]
        mask_multi, target = np.ones([self.num_classes, num_samples]), 0.0

        for c in range(self.num_classes):
            c_indices = np.where(labels == c)
            mask_multi[c, c_indices] = target
        return torch.tensor(mask_multi).type(torch.long).to(self.device)

    def remove_diag(self, M):
        h, w = M.shape
        assert h == w, "h and w should be same"
        mask = np.ones((h, w)) - np.eye(h)
        mask = torch.from_numpy(mask)
        mask = (mask).type(torch.bool)
        return M[mask].view(h, -1).to(self.device)

    def forward(self, embed, proxy, label, **_):
        '''
        :param embed: feature embedding
        :param proxy: proxy for ground truth
        :param label: ground truth for dataset
        :param _:
        :return:
        '''
        # calculate similarities between sample embeddings
        sim_matrix = self.calculate_similarity_matrix(embed, embed) + self.m_p - 1
        # remove diagonal terms
        sim_matrix = self.remove_diag(sim_matrix/self.temperature)
        # for numerical stability
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = F.relu(sim_matrix) - sim_max.detach()

        # calculate similarities between sample embeddings and the corresponding proxies
        smp2proxy = self.cosine_similarity(embed, proxy)
        # make false negative removal
        removal_fn = self.remove_diag(self.make_index_matrix(label)[label])
        # apply the negative removal to the similarity matrix
        improved_sim_matrix = removal_fn*torch.exp(sim_matrix)

        # compute positive attraction term
        pos_attr = F.relu((self.m_p - smp2proxy) / self.temperature)
        # compute negative repulsion term
        neg_repul = torch.log(torch.exp(-pos_attr) + improved_sim_matrix.sum(dim=1))
        # compute data to data cross-entropy criterion
        criterion = pos_attr + neg_repul
        return criterion.mean()

class LeCamEMA(object):
    # Simple wrapper that applies EMA to losses.
    # https://github.com/google/lecam-gan/blob/master/third_party/utils.py
    def __init__(self, init=7777, decay=0.9, start_iter=0):
        self.G_loss = init
        self.D_loss_real = init
        self.D_loss_fake = init
        self.D_real = init
        self.D_fake = init
        self.decay = decay
        self.start_itr = start_iter

    def update(self, cur, mode, itr):
        if itr < self.start_itr:
            decay = 0.0
        else:
            decay = self.decay
        if mode == "G_loss":
            self.G_loss = self.G_loss*decay + cur*(1 - decay)
        elif mode == "D_loss_real":
            self.D_loss_real = self.D_loss_real*decay + cur*(1 - decay)
        elif mode == "D_loss_fake":
            self.D_loss_fake = self.D_loss_fake*decay + cur*(1 - decay)
        elif mode == "D_real":
            self.D_real = self.D_real*decay + cur*(1 - decay)
        elif mode == "D_fake":
            self.D_fake = self.D_fake*decay + cur*(1 - decay)

def lecam_reg(d_logit_real, d_logit_fake, ema):
    reg = torch.mean(F.relu(d_logit_real - ema.D_fake).pow(2)) + \
          torch.mean(F.relu(ema.D_real - d_logit_fake).pow(2))
    return reg

def d_vanilla(d_logit_real, d_logit_fake):
    d_loss = torch.mean(F.softplus(-d_logit_real)) + torch.mean(F.softplus(d_logit_fake))
    return d_loss


def g_vanilla(d_logit_fake):
    return torch.mean(F.softplus(-d_logit_fake))


def d_logistic(d_logit_real, d_logit_fake):
    d_loss = F.softplus(-d_logit_real) + F.softplus(d_logit_fake)
    return d_loss.mean()


def g_logistic(d_logit_fake):
    # basically same as g_vanilla.
    return F.softplus(-d_logit_fake).mean()


def d_ls(d_logit_real, d_logit_fake):
    d_loss = 0.5 * (d_logit_real - torch.ones_like(d_logit_real))**2 + 0.5 * d_logit_fake ** 2
    return d_loss.mean()


def g_ls(d_logit_fake):
    gen_loss = 0.5 * (d_logit_fake - torch.ones_like(d_logit_fake))**2
    return gen_loss.mean()


def d_hinge(d_logit_real, d_logit_fake):
    return torch.mean(F.relu(1. - d_logit_real)) + torch.mean(F.relu(1. + d_logit_fake))


def g_hinge(d_logit_fake):
    return -torch.mean(d_logit_fake)


def d_wasserstein(d_logit_real, d_logit_fake):
    return torch.mean(d_logit_fake - d_logit_real)


def g_wasserstein(d_logit_fake):
    return -torch.mean(d_logit_fake)






