# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2022/1/26 18:32
import torch
import numpy as np
import scipy.sparse as sp
from sklearn import preprocessing

def MaxAbs(signal_data):
    signal_data = preprocessing.MaxAbsScaler().fit_transform(signal_data.reshape(-1, 1))
    signal_data = signal_data.squeeze(-1)
    # maxAbs = np.max(np.abs(signal_data), axis=0)
    # signal_data = signal_data / maxAbs
    return signal_data

def MaxMin(signal_data):
    min = np.min(signal_data)
    max = np.max(signal_data)
    signal_data = (signal_data - min) / (max - min)
    return signal_data

def Z_Score(signal_data):
    expectation = np.mean(signal_data, axis=0)
    variance = np.std(signal_data, axis=0)
    signal_data = (signal_data - expectation) / variance
    return signal_data

def DeNorm_MaxAbs(src_data, tar_data):
    maxAbs = np.max(np.abs(tar_data), axis=0)
    src_data = src_data * maxAbs
    return src_data

def DeNorm_MaxMin(src_data, tar_data):
    min = np.argmin(tar_data, axis=0)
    max = np.argmax(tar_data, axis=0)
    src_data = src_data * (max - min) + min
    return src_data

def DeNorm_Z_score(src_data, tar_data):
    expectation = np.mean(tar_data, axis=0)
    variance = np.std(tar_data, axis=0)
    src_data = src_data * variance + expectation
    return src_data


def normalized_laplacian(adj):
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return (sp.eye(adj.shape[0]) - d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)).tocoo()


def laplacian(adj):
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1)).flatten()
    d_mat = sp.diags(row_sum)
    return (d_mat - adj).tocoo()


def gcn(adj):
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return (sp.eye(adj.shape[0]) + d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)).tocoo()


def aug_normalized_adjacency(adj):
    adj = adj + torch.eye(adj.shape[0]).cuda()
    row_sum = torch.sum(adj, dim=1)
    d_inv_sqrt = torch.flatten(torch.pow(row_sum, -0.5))
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)


def bingge_norm_adjacency(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return (d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt) + sp.eye(adj.shape[0])).tocoo()


def normalized_adjacency(adj):
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return (d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)).tocoo()


def random_walk_laplacian(adj):
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv = np.power(row_sum, -1.0).flatten()
    d_mat = sp.diags(d_inv)
    return (sp.eye(adj.shape[0]) - d_mat.dot(adj)).tocoo()


def aug_random_walk(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv = np.power(row_sum, -1.0).flatten()
    d_mat = sp.diags(d_inv)
    return (d_mat.dot(adj)).tocoo()


def random_walk(adj):
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv = np.power(row_sum, -1.0).flatten()
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).tocoo()


def no_norm(adj):
    adj = sp.coo_matrix(adj)
    return adj


def i_norm(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    return adj


def fetch_normalization(type):
    switcher = {
        'NormLap': normalized_laplacian,  # A' = I - D^-1/2 * A * D^-1/2
        'Lap': laplacian,  # A' = D - A
        'RWalkLap': random_walk_laplacian,  # A' = I - D^-1 * A
        'FirstOrderGCN': gcn,  # A' = I + D^-1/2 * A * D^-1/2
        'AugNormAdj': aug_normalized_adjacency,
        # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2          !!!!!!!!!!!!!!!!!!!!
        'BingGeNormAdj': bingge_norm_adjacency,  # A' = I + (D + I)^-1/2 * (A + I) * (D + I)^-1/2
        'NormAdj': normalized_adjacency,  # D^-1/2 * A * D^-1/2
        'RWalk': random_walk,  # A' = D^-1*A
        'AugRWalk': aug_random_walk,  # A' = (D + I)^-1*(A + I)
        'NoNorm': no_norm,  # A' = A
        'INorm': i_norm,  # A' = A + I
    }
    func = switcher.get(type, lambda: "Invalid normalization technique.")
    return func


def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


# a simple test example
# x = np.array([1, 2, 3, 4, -1, -2, -3, -4])
# # norm_x = MaxAbs(x)
# # norm_x = MaxMin(x)
# norm_x = Z_Score(x)
# # denorm_x = DeNorm_MaxAbs(norm_x, x)
# # denorm_x = DeNorm_MaxMin(norm_x, x)
# denorm_x = DeNorm_Z_score(norm_x, x)
# print("x:", x)
# print("norm_x:", norm_x)
# print("denorm_x:", denorm_x)
