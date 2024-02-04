import numpy as np
import pandas as pd
from numpy import nan

csvff = "Embeddingpkl_csv/199_K=4.pkl.csv"


def get_dic():
    df = pd.read_csv(csvff)
    usecols = [0, 1, 2, 3]
    df0 = pd.read_csv(csvff, usecols=[0])
    df1 = pd.read_csv(csvff, usecols=[1])
    key = []
    value = []
    list_protein = list(df["protein"])
    list_protein_embedding = list(df["embedding"])
    for i in list_protein:
        key.append(i)
    key[13] = 'NULL'
    ten = torch.load('Embeddingpkl_csv/Large_KI_112_epoch3.pt')
    tensor_ptfile_read = ten['masked_lm']
    tensor_ptfile_read_index = 0
    for j in list_protein_embedding:
        LLL = j.replace('\n', '').replace('\n', '').replace('[', '').replace(']', '').split(' ')
        x = 0
        for a in LLL:
            if a == 'nan':
                LLL = [0] * 512
                break
            if a == '':
                LLL.pop(x)
                continue
            x = x + 1
        LLL = [float(i) for i in LLL if i != '']
        LLL.append(tensor_ptfile_read[tensor_ptfile_read_index])
        tensor_ptfile_read_index += 1
        value.append(LLL)
    dic = dict(zip(key, value))
    L1 = []
    with open('ki/bindingDB_ki_5000.txt') as f1:
        flag_done = 1
        while flag_done:
            readline = f1.readline().strip().split('\t')
            readline_one = readline[0]

            if readline_one == '':
                flag_done = 0
                break
            readline_two = readline[1]
            L1.append(readline_two)

    x = 0
    y = 0
    for i in L1:
        y = y + 1
        try:
            dic[i.upper()]
        except:
            x = x + 1

    n_a = 0
    for i in dic.values():
        if len(i) != 513:
            aaa = len(i)
            for j in range(513 - len(i) - 1):
                i.append(float(0))
            i.append(i[aaa - 1])
            i[aaa - 1] = float(0)

            n_a = n_a + 1
    return dic


import torch
import torch.nn as nn
import numpy as np
import pickle
import argparse
import torch.optim as optim
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from metric import *
from model import GanDTI

# cmpnn
from Deal_parsing import add_train_args
from chemprop.models import build_model

parser = argparse.ArgumentParser()
parser.add_argument("--lr", help="learning rate", type=float, default=0.001)
parser.add_argument("--ld", help="learning rate decay", type=float, default=0.5)
parser.add_argument("--epoch", help="epoch", type=int, default=3000)
parser.add_argument("--features", help="feature dimension", type=int, default=40)
parser.add_argument("--GNN_depth", help="gnn layer number", type=int, default=3)
parser.add_argument("--MLP_depth", type=int, default=2)
parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay (L2 loss on parameters).')
parser.add_argument("--mode", type=str, help="regression or classification", default='regression')
parser.add_argument("--dataset", type=str, default='ki')

add_train_args(parser)

args = parser.parse_args()

data_list = []

# CPCPort
protein_CPCPort_list = []

protein_num1 = 0
with open('ki/bindingDB_ki_5000.txt') as f1:
    flag_done = 1
    while flag_done:
        readline = f1.readline().strip().split('\t')
        readline_one = readline[0]

        if readline_one == '':
            flag_done = 0
            break
        readline_two = readline[1]
        data_list.append(readline_one)

        if readline_two not in protein_CPCPort_list:
            protein_num1 = protein_num1 + 1
        protein_CPCPort_list.append(readline_two)
smiless_batch = data_list

protein_CPCPort = protein_CPCPort_list
protein_CPCPort_list_read = []
LL1 = []
with open('Embeddingpkl_csv/protein_original.txt') as f1:
    flag_done = 1
    while flag_done:
        readline = f1.readline().strip().split('\t')
        readline_one = readline[0]

        if readline_one == '':
            flag_done = 0
            break

        LL1.append(readline_one)

dic = get_dic()
for i in LL1:
    protein_CPCPort_list_read.append(dic[i.upper()])

# load GPU or CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('program uses GPU. start now')
else:
    device = torch.device('cpu')
    print('program uses CPU')


def loadNpy(fileName, dtype):
    tensor = [dtype(data).to(device) for data in np.load(fileName + '.npy', allow_pickle=True)]
    return tensor


def loadPickle(fileName):
    with open(fileName, 'rb') as f:
        return pickle.load(f)


def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return mult / float(y_obs_sq * y_pred_sq)


def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))


def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

    return 1 - (upp / float(down))


def get_rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))


def load_data(dataset, mode):
    # load preprocessed data
    data_file = './' + dataset + '/'
    compounds = loadNpy(data_file + 'compounds', torch.LongTensor)
    adjacencies = loadNpy(data_file + 'adjacencies', torch.FloatTensor)
    proteins = loadNpy(data_file + 'proteins', torch.LongTensor)

    proteins_tensor = []
    proteins1 = protein_CPCPort_list_read
    fingerprintDict = loadPickle(data_file + 'fingerprint.pickle')
    wordDict = loadPickle(data_file + 'wordDict.pickle')
    compound_len = len(fingerprintDict)
    protein_len = len(wordDict)
    if mode == 'classification':
        interactions = loadNpy(data_file + 'interactions', torch.LongTensor)
    elif mode == 'regression':
        interactions = loadNpy(data_file + 'interactions', torch.FloatTensor)
    dataset = list(zip(smiless_batch, adjacencies, proteins1, interactions))

    np.random.seed(seed=1234)
    np.random.shuffle(dataset)
    trainNumber = int(len(dataset) * 0.8)
    trainData = dataset[:trainNumber]
    testData = dataset[trainNumber:]
    return trainData, testData, compound_len, protein_len  # need revision


def load_data2(dataset, mode):
    # load preprocessed data
    data_file = './' + dataset + '/'
    compounds = loadNpy(data_file + 'compounds', torch.LongTensor)
    adjacencies = loadNpy(data_file + 'adjacencies', torch.FloatTensor)
    proteins = loadNpy(data_file + 'proteins', torch.LongTensor)

    proteins_tensor = []
    proteins1 = protein_CPCPort_list_read
    fingerprintDict = loadPickle(data_file + 'fingerprint.pickle')
    wordDict = loadPickle(data_file + 'wordDict.pickle')
    compound_len = len(fingerprintDict)
    protein_len = len(wordDict)
    if mode == 'classification':
        interactions = loadNpy(data_file + 'interactions', torch.LongTensor)
    elif mode == 'regression':
        interactions = loadNpy(data_file + 'interactions', torch.FloatTensor)
    dataset = list(zip(smiless_batch, adjacencies, proteins1, interactions))
    np.random.seed(seed=1234)
    np.random.shuffle(dataset)
    trainNumber = int(len(dataset) * 0.8)
    testNumber = int(len(dataset) * 0.9)
    trainData = dataset[:trainNumber]
    testData = dataset[trainNumber:testNumber]
    valData = dataset[testNumber:]
    return trainData, testData, valData, compound_len, protein_len  # need revision


def train(dataset, mode, optimizer):
    np.random.shuffle(dataset)
    for data in dataset:
        optimizer.zero_grad()
        output = model(data)
        output.backward()
        optimizer.step()
    return output


def test_data_process(test_dataset):
    labels, predictions, scores = [], [], []
    for data in test_dataset:
        (label, predict, score) = model(data, train=False)
        labels.append(label)
        predictions.append(predict)
        scores.append(score)
    return labels, predictions, scores


def test_regression(test_dataset):
    labels, predictions = [], []
    for data in test_dataset:
        (label, predict, score) = model(data, train=False)
        labels.append(label)
        predictions.append(predict)
        # scores.append(score)
    RMSE = rmse(labels, predictions)
    Pearson = pearson(labels, predictions)

    return RMSE, Pearson,


def val_regression(test_dataset):
    labels, predictions = [], []
    for data in test_dataset:
        (label, predict, score) = model(data, train=False)
        labels.append(label)
        predictions.append(predict)
        # scores.append(score)

    MSE = mse(labels, predictions)
    CI = ci(labels, predictions)

    return MSE, CI,


def test_classification(dataset):
    labels, predictions, scores = test_data_process(dataset)
    AUC = roc_auc_score(labels, scores)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    predictions_array = np.array(predictions)
    labels_array = np.array(labels)
    roce1 = getROCE(predictions_array, labels_array, 0.5)
    roce2 = getROCE(predictions_array, labels_array, 1)
    roce3 = getROCE(predictions_array, labels_array, 2)
    roce4 = getROCE(predictions_array, labels_array, 5)
    return AUC, precision, recall, roce1, roce2, roce3, roce4



if args.dataset == 'ki' or args.dataset == 'davis':
    train_data, test_data, val_data, compound_len, protein_len = load_data2(args.dataset, args.mode)

# load the model
torch.manual_seed(0)
model = GanDTI(compound_len, protein_len, args.features, args.GNN_depth, args.MLP_depth, args.mode, args
               ).to(device)
# optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=args.lr,
                       weight_decay=args.weight_decay, amsgrad=True)


def main():
    bestpearson = 0
    bestrmse = 1
    for epoch in range(args.epoch):
        model.train()
        lossTrain = train(train_data, args.mode, optimizer)
        model.eval()
        if args.mode == 'regression':

            if args.dataset == 'ki' or args.dataset == 'davis':
                mse_test, ci_test = val_regression(test_data)
                print('Epoch:{:03d}'.format(epoch + 1),
                      'train loss:{:.5f}'.format(lossTrain),
                      'mse:{:5}'.format(str(mse_test)),
                      'ci:{:5}'.format(str(ci_test)),
                      )
                mse_val, ci_val = val_regression(val_data)
                print(
                    'val:',
                    'mse:{:5}'.format(str(mse_val)),
                    'ci:{:5}'.format(str(ci_val)),
                )



if __name__ == '__main__':
    main()

