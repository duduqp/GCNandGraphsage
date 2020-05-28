# coding: UTF-8
import time
import torch
import numpy as np
import models
from config import opt
import data
from sklearn import metrics, manifold
from tqdm import tqdm
import visdom
from torch.utils.data import DataLoader
from models.Graphsage import *

def cvt_mat(labels):
    s=np.max(labels)+1
    I=np.identity(s)
    label_mat=list()
    for i in range(len(labels)):
        label_mat.append(I[labels[i]])
    labels_mat=np.array(label_mat)
    return labels_mat

def train(**kwargs):
    viz=visdom.Visdom(env='main')

    # 创建窗口并初始化
    viz.line([[0.,0.]], [0], win='train_loss_acc_per_epoch', opts=dict(title='train_loss_acc_per_epoch',legend=['loss*200','acc'],showlegend=True))

    opt._parse(kwargs)
    if opt.model is 'GraphSage':
        run_cora(viz)
        # run_pubmed(viz)
        return
    adj, features, labels, idx_train, idx_val, idx_test = data.load_data(opt)
    labels_vis = labels#拷贝一份
    #labels:[0...1...4...6...4...3...5...] -> label__mat=[ [1,0,0,0,0,0,0] , [0,1,0,0,0,0,0] , [....]]
    label_mat=cvt_mat(labels)
    print(label_mat.shape)
    results=t_SNE(label_mat,3)
    Visualization(results,labels,viz)

    train = data.Dataload(labels, idx_train)
    val = data.Dataload(labels, idx_val)
    test = data.Dataload(labels, idx_test)
    model = getattr(models, opt.model)(features.shape[1], 128, max(labels) + 1).train()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model = model.to(opt.device)
    adj = adj.to(opt.device)
    features = features.to(opt.device)   # 将模型以及在模型中需要使用到的矩阵加载到设备中
    train_dataloader = DataLoader(train, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_dataloader = DataLoader(val, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    test_dataloader = DataLoader(test, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    criterion = F.nll_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    lr = opt.lr
    for epoch in range(opt.max_epoch):
        for trains, labels in tqdm(train_dataloader):
            labels = labels.to(opt.device)
            trains = trains.to(opt.device)
            optimizer.zero_grad()
            outputs = model(features, adj)
            loss = criterion(outputs[trains], labels)
            loss.backward()
            optimizer.step()
        lr = lr * opt.lr_decay
        for param_groups in optimizer.param_groups:
            param_groups['lr'] = lr
        evalute(opt, model, val_dataloader, epoch, features, adj,idx_val,labels_vis,viz)
        model.train()
        model.save()
    output_test=Test(model,features,labels_vis,adj,idx_test)
    output_test = output_test.cpu().detach().numpy()
    result=t_SNE(output_test,3)
    Visualization(result,labels_vis,viz)

def evalute(opt, model, val_dataloader, epoch, features, adj,idx,label,viz):

    model.eval()
    loss_total = 0
    predict_all = list()
    labels_all = list()
    critetion = F.nll_loss


    with torch.no_grad():
        for evals, labels in tqdm(val_dataloader):
            labels = labels.to(opt.device)
            evals = evals.to(opt.device)
            outputs = model(features, adj)
            loss = critetion(outputs[evals], labels)#算整个数据集，但是只用validationset索引取出validation的那一部分算损失
            # print(evals)
            # print(labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs[evals].data, 1)[1].cpu().numpy()
            labels = list(labels)
            predic = list(predic)
            labels_all.extend(labels)
            predict_all.extend(predic)
    acc = metrics.accuracy_score(labels_all, predict_all)
    viz.line([[loss_total/200,acc]],[epoch+1],win='train_loss_acc_per_epoch',update='append',opts=dict(title='train_loss_acc_per_epoch',legend=['loss*200','acc'],showlegend=True))
    print("\nThe acc for Epoch %s is %f" % (str(epoch+1), acc))
    return acc


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def Test(model,features,labels,adj,idx_test):
    model.eval()
    output = model(features, adj)
    idx_test = torch.LongTensor(idx_test)
    predict=output[idx_test]
    labels=labels[idx_test]
    labels=torch.LongTensor(labels)
    loss_test = F.nll_loss(predict,labels)
    acc_test = accuracy(predict, labels)#只取test idx
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    return output                                               # 可视化返回output,是整个数据集的

# t-SNE 降维
def t_SNE(output, dimention):
    # output:待降维的数据
    # dimention：降低到的维度
    tsne = manifold.TSNE(n_components=dimention, init='pca', random_state=0)
    result = tsne.fit_transform(output)
    return result

# Visualization with visdom
def Visualization(result, labels,vis):
    labels=np.array(labels)
    result=np.array(result)
    vis.scatter(
        X = result,
        Y = labels+1,           # 将label的最小值从0变为1，显示时label不可为0
       opts=dict(markersize=5,title='Dimension reduction to %dD' %(result.shape[1])),
    )


if __name__ == '__main__':
    train()
