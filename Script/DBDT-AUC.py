#  -*-  coding: UTF-8  -*-
# This is a Python script for DBDT-AUC package. Owing to fraud is a small probability event, fraud
# detection can translate into how to better deal with imbalanced data substantially. Compared with
# accuracy and the corresponding surrogated loss, AUC is a more reasonable measure for imbalanced data,
# and meanwhile the AUC-based surrogate loss function also has better performance when training
# deep learning models with imbalanced data, which can then be effectively solved with theoretical
# convergence using a primal-dual algorithm Proximal Primal-Dual Stochastic Gradient.

from libauc.losses import AUCMLoss
from libauc.optimizers import PESG
import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold,StratifiedShuffleSplit
torch.set_default_tensor_type(torch.DoubleTensor)

pathxtrain = r'../Data'
pathxtest = r'../Data'
pathytrain = r'../Data'
pathytest = r'../Data'

x_train = pd.read_csv(pathxtrain, header=0, encoding='utf-8')
y_train = pd.read_csv(pathytrain, header=0, encoding='utf-8')
x_test = pd.read_csv(pathxtest, header=0, encoding='utf-8')
y_test = pd.read_csv(pathytest, header=0, encoding='utf-8')

# imbalance for dataset
imratio = 394 / (394+227451)

x_train = (x_train - x_train.min()) / (x_train.max() - x_train.min())
x_test = (x_test - x_test.min()) / (x_test.max() - x_test.min())
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train).astype(np.float32)
y_test = np.array(y_test).astype(np.float32)
y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)
y_train = y_train * 2 - 1
y_test = y_test * 2 - 1

# 定义SDT+boost
class SDTs(nn.Module):
    def __init__(
            self,
            n_x,
            tree_levels=4,
            lmbd=0.1,
            T=100,
            imratio=None,
            margin=1,
            backend='ce',
            use_cuda=True):
        super(SDTs, self).__init__()
        self.tree_levels = tree_levels  # 树深度
        self.leafs_numb = 2 ** (self.tree_levels - 1)
        self.inner_numb = self.leafs_numb - 1
        self.lmbd = lmbd
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.T = T
        self.margin = margin
        self.p = imratio
        self.a = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.b = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.alpha = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.backend = 'ce'  # TODO:

        self.V_next_value = torch.zeros([self.inner_numb, self.T])
        #         self.parameters = self.__dict__

        self.param_dict = {}
        for i in range(T):
            self.param_dict['W' + str(i)] = nn.Parameter(
                torch.nn.init.xavier_uniform_(Variable(torch.randn(n_x, self.inner_numb)).cuda(), gain=1))
            self.param_dict['b' + str(i)] = nn.Parameter(
                torch.nn.init.constant_(Variable(torch.randn(1, self.inner_numb)).cuda(), 0))
            self.param_dict['phi' + str(i)] = nn.Parameter(
                torch.nn.init.xavier_uniform_(Variable(torch.randn(self.leafs_numb, 1), ).cuda(), gain=1))

        #         import pdb; pdb.set_trace()
        self.params = nn.ParameterDict(
            self.param_dict
        )

    #         for i in range(T):
    #             self.parameters['W' + str(i)] = nn.Parameter(
    #                 torch.nn.init.xavier_uniform_(Variable(torch.randn(n_x, self.inner_numb)).cuda(), gain=1))
    #             self.parameters['b' + str(i)] = nn.Parameter(
    #                 torch.nn.init.constant_(Variable(torch.randn(1, self.inner_numb)).cuda(), 0))
    #             self.parameters['phi' + str(i)] = nn.Parameter(
    #                 torch.nn.init.xavier_uniform_(Variable(torch.randn(self.leafs_numb, 1), ).cuda(), gain=1))
    #             self.parameters['W' + str(i)].requires_grad = True
    #             self.parameters['b' + str(i)].requires_grad = True
    #             self.parameters['phi' + str(i)].requires_grad = True

    def node_probability(self, index_node, A):
        p = torch.ones(A.shape[0]).cuda()
        while index_node - 1 >= 0:
            father_index = int((index_node - 1) / 2)
            if (index_node - 1) % 2 == 0:
                p = p * (1.0 - A[:, father_index])
            else:
                p = p * (A[:, father_index])
            index_node = father_index
        return p

    def forward_propagation_Boosting(self, X, W, b):
        Z = torch.add(torch.matmul(X, W), b)
        A = torch.sigmoid(1 * Z)
        return A

    def compute_leafs_prob_matrix(self, A):
        ta = list()
        i = 0
        while i < self.leafs_numb:
            ta.append(self.node_probability(self.leafs_numb - 1 + i, A))
            i = i + 1
        leafs_prob_matrix = torch.stack(ta, dim=0)
        return leafs_prob_matrix

    def compute_inner_prob_matrix(self, A):
        ta = list()
        i = 0
        while i < self.inner_numb:
            ta.append(self.node_probability(i, A))
            i = i + 1
        inner_prob_matrix = torch.stack(ta, dim=0)
        return inner_prob_matrix

    # 计算 regularization为惩罚项，
    def compute_regularization(self, A, inner_prob_matrix, V_prec):
        ta = list()
        ema = list()
        i = 0
        while i < self.inner_numb:
            depth = int(np.log(i + 1) / np.log(2))
            decay = 1. - np.exp(-depth)
            a_i = torch.div(torch.matmul(inner_prob_matrix[i, :], A[:, i]), torch.sum(inner_prob_matrix[i, :]))
            w_i = decay * V_prec[i] + (1. - decay) * a_i
            r_i = -self.lmbd * (2 ** (-depth)) * (
                    0.5 * torch.log(w_i) + 0.5 * torch.log(1.0 - w_i))
            ta.append(r_i)
            ema.append(w_i)
            i = i + 1
        regularization = torch.sum(torch.stack(ta, dim=0))
        V_next = torch.stack(ema, dim=0)
        return regularization, V_next

    def compute_Boosting_output(self, X, Y):
        #         weight = torch.ones_like(Y)
        #         weight [Y > 0] = 1
        #         weight.view(-1)
        output_sum = torch.full_like(Y, Y.mean())
        output = []
        #         print('training phi', self.param_dict['phi' + str(0)])
        for t in range(self.T):
            A = self.forward_propagation_Boosting(X, self.param_dict['W' + str(t)], self.param_dict['b' + str(t)])
            leafs_prob_matrix = self.compute_leafs_prob_matrix(A)
            inner_prob_matrix = self.compute_inner_prob_matrix(A)
            output.append(torch.matmul(leafs_prob_matrix.permute(1, 0), self.param_dict['phi' + str(t)]))
            output_sum = output_sum + 0.1 * torch.squeeze(output[t])
        return output_sum


# paramaters
SEED = 123
BATCH_SIZE = 128
lr = 0.1
gamma = 500
weight_decay = 1e-4
margin = 1.0

if __name__ == "__main__":
    batch_size = 12800
    epochs = 400
    t = 40
    #     imratio = 0.95
    margin = 1.0
    lr = 0.1
    gamma = 500
    weight_decay = 1e-4
    beta1 = 0.9  # try different values: e.g., [0.999, 0.99, 0.9]
    beta2 = 0.999  # try different values: e.g., [0.999, 0.99, 0.9]
    train_data = torch.tensor(x_train).cuda()
    train_labels = torch.tensor(y_train).cuda()
    eval_data = torch.tensor(x_test).cuda()
    eval_labels = torch.tensor(y_test).cuda()

    train_dataset = Data.TensorDataset(train_data, train_labels)
    eval_dataset = Data.TensorDataset(eval_data, eval_labels)

    train_loader = Data.DataLoader(
        train_dataset,
        batch_size=batch_size
    )
    test_loader = Data.DataLoader(
        eval_dataset,
        batch_size=batch_size
    )
    (m, n_x) = train_data.shape
    costs = []
    minibatches = int(m / batch_size)
    sdts = SDTs(n_x)
    sdts = sdts.cuda()
    alpha = 0.05
    Loss = AUCMLoss(imratio=imratio)
    optimizer = PESG(sdts,
                     a=Loss.a,
                     b=Loss.b,
                     alpha=Loss.alpha,
                     imratio=imratio,
                     lr=lr,
                     gamma=gamma,
                     margin=margin,
                     weight_decay=weight_decay)

    test_auc_max = 0
    print('-' * 30)
    best_testing_auc = 0.0
    for epoch in range(epochs):
        # Training
        sdts.train()
        epoch_cost = 0
        if epoch == int(0.5 * epochs) or epoch == int(0.75 * epochs):
            optimizer.update_regularizer(decay_factor=10)
        train_pred = []
        train_true = []
        for batch_idx, (data, target) in enumerate(train_loader):
            torch.cuda.empty_cache()
            data = data
            target = target.view(-1)
            optimizer.zero_grad()
            y_pred = sdts.compute_Boosting_output(data, target)
            loss = Loss(y_pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_pred.append(y_pred.cpu().detach().numpy())
            train_true.append(target.cpu().detach().numpy())

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_auc = roc_auc_score(train_true, train_pred)
        # Evaluating
        sdts.eval()
        torch.set_grad_enabled(False)
        torch.cuda.empty_cache()
        test_pred = sdts.compute_Boosting_output(eval_data, eval_labels)
        val_auc = roc_auc_score(eval_labels.cpu().detach().numpy(), test_pred.cpu().detach().numpy())
        predictions = (torch.tanh(test_pred) > 0).type(torch.FloatTensor) - (torch.tanh(test_pred) < 0).type(
            torch.FloatTensor)
        predictions = predictions.type(torch.DoubleTensor).cuda()

        if test_auc_max < (1 - val_auc):
            test_auc_max = 1 - val_auc

        # print results
        print("epoch: {}, train_auc:{:4f}, test_auc:{:4f}, test_auc_max:{:4f}".format(epoch, (1 - train_auc),
                                                                                      (1 - val_auc),
                                                                                      (1 - test_auc_max), optimizer.lr))
        sdts.train()
        torch.set_grad_enabled(True)

        #         # training_accuracy, training_output, _ = sdts.compute_accuracy_Boosting(train_data, train_labels)
        #         test_accuracy, test_output, pre = sdts.compute_accuracy_Boosting(eval_data, eval_labels)
        #         # training_fpr, training_tpr, training_thresholds = roc_curve(train_labels.cpu().detach().numpy(),
        #         #                                                            training_output.cpu().detach().numpy())
        #         #         training_auc = auc(training_fpr, training_tpr)
        #         test_fpr, test_tpr, test_thresholds = roc_curve(eval_labels.cpu().detach().numpy(), pre.cpu().detach().numpy())
        #         test_auc = auc(test_fpr, test_tpr)
        #         #         ac_train = training_accuracy.detach().numpy()
        #         ac_test = test_accuracy.detach().numpy()
        #         #         auc_train = training_auc
        #         auc_test = test_auc
        #         if auc_test > best_testing_auc:
        #             best_testing_auc = auc_test
        #             best_testing_acc = ac_test
        #             a = classification_report(pre.cpu().detach().numpy(), eval_labels.cpu().detach().numpy(), digits=8)
        #         loss = epoch_cost.cpu().detach().numpy()
        #         result = result.append(
        #             pd.DataFrame({'epoch': [epoch], 'loss': [loss / t], 'ac_test': [ac_test], 'auc_test': [auc_test]}),
        #             ignore_index=True)
        #         print(result)
        torch.set_grad_enabled(True)