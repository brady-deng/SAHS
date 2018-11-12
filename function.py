import pandas as pd
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier, export_graphviz
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, VotingClassifier, \
    ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import graphviz

from mpl_toolkits.mplot3d import Axes3D


###########################################
# deng的SAHS数据集处理函数包
# load_data读取MATLAB输出的特征集合xlsx文件
# createahiset时间轴上均匀切割数据
# createdataset产生在时间轴上均匀切割分布的数据集合，可以使用交叉训练，不用担心冗余问题
# parop超参数调优，可以返回各参数对应的分类结果，使用gridsearchcv方法
# train97直接切割数据集的方式训练分类器
# train_911针对于每个被试每一折数据的超参数寻优
# train_912主要是为了与MATLAB对比，不涉及超参数寻优问题，用的参数与MATLAB相同
# resana_911,train911与train912的结果都可以使用该函数
# resana_913,parop超参数寻有结果使用该函数，#res,超参数网格寻优结果,#N，原始数据切割段数,#index，评价指标内容
# OB，data，训练数据，label，训练标签，name，输出文件名称，其他为决策树参数
# createahiset,产生ahi训练测试集合
# AHIval，输入的分别是分类器名称，切分好的数据集，标签，返回的是标签的ahi区间，预测的ahi区间以及片段的评价指标
# AHIcal，输入的分别是分类器名称，切分好的数据集，标签，返回的分别是ah次数、相应长度、开始时间以及结束时间
# AHIres,clf,分类器名称，data输入的训练数据，label输入的标签数据，返回的分别是片段正确率，错误分类的区间，区间正确率以及遗漏的区间
# smoothres,对输出结果做平滑处理，输入的是label以及相应的阈值，返回的与AHIcal相同.
# clfcas,分类器级联定位ahi区间，clfs为输入的级联分类器，data、label为1号分类器的训练数据，data2、label2为2号分类器的训练数据
# 返回的是1号分类器的片段分类性能，2号分类器的片段分类性能，最终ahi区间的分类性能以及各种错误区间。
# drawtree,输入项为clf与输出文件名称，保存为pdf格式，可以观察树的结构
# dataseg,用于级联分类器的单数据集按比例切割
# clfcastrain，级联分类器训练以及性能评估
# clfcastest，级联分类器测试及性能评估
# getdataset，按比例获得训练集、测试集，输入的是包含各被试的data、label以及数据集切割比例P，返回的是训练集以及测试集
# AHItrain，单分类器训练
# AHItest，单分类器测试
# casdata，为级联分类器产生K折交叉训练数据
# clfcaskfold，级联分类器K折交叉训练并且返回训练结果
# clfkfold,单分类器交叉训练结果
###########################################
def durmerge(label,time,timeroi,start,end,thre):
    #label，待处理标签
    #time，标签对应的时间索引
    #timeroi，感兴趣区域的时间索引
    #start，timeroi的序列索引
    #end，timeroi的序列索引
    #thre，merge的阈值
    #返回的是处理之后的标签
    dur = durdetect(start,end)
    for i in range(len(dur)):
        if dur[i] < thre:
            label[time2ind(time,timeroi[start[i]]):time2ind(time,timeroi[end[i]])+1] = 0
    return label
def intmerge(label,time,timeroi,start,end,thre):
    #label，待处理标签
    #time，标签对应的时间索引
    #timeroi，感兴趣区域的时间索引
    #start，timeroi的序列索引
    #end，timeroi的序列索引
    #返回的是处理之后的标签
    inter = intdetect(timeroi,start,end)
    for i in range(len(inter)):
        if inter[i]  < thre:
            label[time2ind(time, timeroi[end[i]])+1:time2ind(time, timeroi[start[i+1]])] = 1
    return label

def intdetect(timeroi,start,end):
    #timeroi，感兴趣区域时间序列
    #start，timeroi的序列索引
    #end，timeroi的序列索引
    #返回的是阳性片段的间隔
    inter = []
    for i in range(len(start) - 1):
        inter.append(timeroi[start[i+1]] - timeroi[end[i]] - 1)
    inter = np.array(inter)
    return inter
def ind2time(time,ind):
    #ind，序列索引
    #time，时间序列
    #返回的是时间索引
    if len(time)!=0:
        timeind = time[ind]
    else:
        timeind = []
    return  timeind
def time2ind(time,timeind):
    #time，时间序列
    #timeind，时间索引
    #返回的是序列索引

    if type(timeind) == np.int64:
        temp = np.where(time == timeind)
        temp = temp[0][0]
        ind = temp
        indroi = ind
    else:
        ind = []
        for item in timeind:
            temp = np.where(time == item)
            temp = temp[0][0]
            ind.append(temp)
        ind = np.array(ind)
        indroi = ind

    return indroi
def roidetect(label,time):
    #label，待检测标签
    #time，标签对应的时间序列
    #返回的是timeroi,temps,tempe
    #timeroi，感兴趣区域的时间序列,np.array
    #temps，感兴趣区域的开始点序列索引,list
    #tempe，感兴趣区域的结束点序列索引,list
    indpre = np.where(label == 1)
    indpre = indpre[0]
    if len(indpre) != 0:
        timeroi = time[indpre]
        tempdiff = np.diff(timeroi)
        tempchange = np.where(tempdiff != 1)
        tempchange = tempchange[0]
        temps = [0]
        tempe = []
        if len(tempchange) != 0:
            for i in range(len(tempchange)):
                tempe.append(tempchange[i])
                temps.append(tempchange[i] + 1)
        tempe.append(len(timeroi)-1)
    else:
        timeroi = []
        temps = []
        tempe = []
    return timeroi,temps,tempe
def durdetect(start,end):
    #start，timeroi的序列索引
    #end，timeroi的序列索引
    #返回的是阳性片段的持续时长
    dur = []
    for i in range(len(start)):
        dur.append(end[i]-start[i]+1)
    dur = np.array(dur)
    return dur
def prefilter(label,time,labelpre,thredur,threint):
    ################################
    #label，原始数据标签
    #time，原始数据时间索引
    #labelpre，分类器预测标签
    #thre，事件检测阈值
    #返回的是labelpre
    #labelpre，处理之后的标签数据
    timeroi,temps,tempe = roidetect(labelpre,time)
    labelpre = intmerge(labelpre, time, timeroi, temps, tempe,threint)
    timeroi, temps, tempe = roidetect(labelpre, time)
    labelpre = durmerge(labelpre,time,timeroi,temps,tempe,thredur)
    return labelpre
def mapwindow(WT1,WT2,ind1,ind2):
    #############################
    #WT1,窗口1的宽度
    #WT2，窗口2的宽度
    #ind1，timeroi，感兴趣区域的时间索引
    #ind2，级联分类器2训练数据的时间索引
    #############################
    tempob = np.diff(ind1)
    tempchange = np.where(tempob!=1)
    temps = [0]
    tempe = []
    # tempe = [len(tempob)]
    if len(tempchange[0]) != 0:
        for i in range(len(tempchange[0])):
            tempe.append(tempchange[0][i])
            temps.append(tempchange[0][i]+1)
    tempe.append(len(tempob))
    tempe = np.array(tempe)
    temps = np.array(temps)
    mape = ind1[tempe]+WT1-WT2
    maps = ind1[temps]
    temps = time2ind(ind2,maps)
    tempe = time2ind(ind2,mape)
    # temps = []
    # tempe = []
    # for i in range(len(mape)):
    #     tempvar = np.where(ind2 == mape[i])
    #     tempe.append(tempvar[0][0])
    #     tempvar = np.where(ind2 == maps[i])
    #     temps.append(tempvar[0][0])
    indtest = []
    for i in range(len(temps)):
        indtest.append([k for k in range(temps[i],tempe[i]+1)])
    temptest = []
    for item in indtest:
        temptest.append(np.array(item))
    indtest = temptest[0]
    for i in range(1,len(temptest)):
        indtest = np.hstack([indtest,temptest[i]])
    return indtest
def clfkfold(clf,data,label,N):
    ######################################
    #单分类器交叉验证
    #输入的是分类器名称，data，label以及被试数目
    #返回的score包含acu、recall、precision
    ######################################
    ind = input('Please input the par of the Decision tree(** ** **):').split() #决策树的叶子节点的参数设置，最小样本切割，最小叶子样本，最大深度
    ind = [int(item) for item in ind]
    classweight = input('Please input the classweight of the decision tree(** **):') #决策树的类权重
    classweight = [int(item) for item in classweight.split()]
    # wei，决策树的类权重
    if classweight[0] != 0:
        wei = {0:classweight[0],1:classweight[1]}
    else:
        wei = 'balanced'
    if clf == "Dec":
        tempclf = DecisionTreeClassifier(class_weight=wei, min_samples_split=int(ind[0]),
                                                          min_samples_leaf=int(ind[1]),
                                                          max_depth=int(ind[2]), random_state=0)
    if clf == "Ran":
        tempclf = RandomForestClassifier(n_estimators=20,n_jobs=4,class_weight=wei, min_samples_split=int(ind[0]),
                                                          min_samples_leaf=int(ind[1]),
                                                          max_depth=int(ind[2]), random_state=0, oob_score= True, max_features=1)
    acuscore = []
    recascore = []
    prescore = []
    # for i in range(N):
    #     acuscore.append(cross_val_score(tempclf,data[i],label[i],n_jobs=4,cv=2,scoring='accuracy')) #分类器交叉训练得到的正确率
    #     recascore.append(cross_val_score(tempclf, data[i], label[i], n_jobs=4, cv=2, scoring='recall')) #分类器交叉训练得到的召回率
    #     prescore.append(cross_val_score(tempclf, data[i], label[i], n_jobs=4, cv=2, scoring='precision'))   #分类器交叉训练得到的精准率
    # score = []
    # for i in range(N):
    #     score.append([acuscore[i].mean(),recascore[i].mean(),prescore[i].mean()])
    kf = KFold(n_splits=2)
    res = []

    count = 0
    for i in range(N):
        count = 0
        eva = np.zeros((3, 5))
        tempacu = []
        temprec = []
        temppr = []
        for train_index,test_index in kf.split(data[i]):
            datatrain,labeltrain = data[i][train_index],label[i][train_index]
            datatest,labeltest = data[i][test_index],label[i][test_index]
            resclf = AHItrain(tempclf,datatrain,labeltrain)
            tempclf.fit(datatrain,labeltrain)
            temppre = tempclf.predict(datatest)
            tempacu.append(accuracy_score(labeltest,temppre))
            temprec.append(recall_score(labeltest,temppre))
            temppr.append(precision_score(labeltest,temppre))
            tempres = AHItest(resclf,datatest,labeltest)
            eva[count,0] = len(tempres[1])
            eva[count,1] = len(tempres[3])
            eva[count,2] = tempres[4]
            eva[count,3:] = tempres[2]
            count +=1
        tempacu = np.array(tempacu)
        temprec = np.array(temprec)
        temppr = np.array(temppr)
        acuscore.append(tempacu)
        recascore.append(temprec)
        prescore.append(temppr)
        eva[-1, 0] = eva[0:2, 0].sum()
        eva[-1, 1] = eva[0:2, 1].sum()
        eva[-1, 2] = eva[0:2, 2].sum()
        eva[-1,3] = (eva[-1,2] - eva[-1,1])/(eva[-1,2]-eva[-1,1]+eva[-1,0])
        eva[-1,4] = (eva[-1,2]-eva[-1,1])/eva[-1,2]
        res.append(eva)

    score = []
    ob = np.zeros([N+1,5])

    for i in range(N):
        score.append([acuscore[i].mean(),recascore[i].mean(),prescore[i].mean()])
        ob[i,:] = res[i][-1,:]
    ob[-1,0] = ob[0:-1,0].sum()
    ob[-1,1] = ob[0:-1,1].sum()
    ob[-1,2] = ob[0:-1,2].sum()
    ob[-1, 3] = (ob[-1, 2] - ob[-1, 1]) / (ob[-1, 2] - ob[-1, 1] + ob[-1, 0])
    ob[-1, 4] = (ob[-1, 2] - ob[-1, 1]) / ob[-1, 2]
    return score,ob
def clfcaskfold(data, label, data2, label2, timeind1,timeind2,N,WT1,WT2, classweight = ['balanced','balanced']):
    #级联分类器交叉训练函数
    #data窗口1对应的数据，list类型
    #label窗口1对应的标签，list类型
    #data2窗口2对应的数据，list类型
    #label2窗口2对应的标签，list类型
    #timeind1，窗口1对应的时间序列
    #timeind2，窗口2对应的时间序列
    #N，被试数目
    #WT1，窗口1的长度
    #WT2，窗口2的长度
    #返回的是res，resave
    P = float(input('Please input the proportion of the dataset:')) #训练街所占比重
    ind = input('Please input the par of the Decision tree(** ** ** ** ** **):').split()
    ind = [int(item) for item in ind]
    res = []
    resave = np.zeros([N+1, 22])
    for i in range(N):
        resi = []
        datatrain1, labeltrain1, datatest1, labeltest1, timetest1,datatrain2, labeltrain2, datatest2, labeltest2,timetest2 = casdata(
            data[i], label[i], data2[i], label2[i], timeind1[i],timeind2[i],WT1, WT2, P)  #切割获得训练集、测试集
        resmat = np.zeros([int(1 / P) + 1, 22])
        for k in range(int(1 / P)):
            clfs, num1, num2 = clfcastrain(["Ran","Ran"], datatrain1[k], labeltrain1[k], datatrain2[k], labeltrain2[k],
                                           0, 0, ind, classweight)  #返回的是训练好的级联分类器，训练样本的阳性样本数、阴性样本数
            tempres = clfcastest(clfs, datatest1[k], labeltest1[k], timetest1[k],datatest2[k], labeltest2[k],timetest2[k], 0, 0,WT1,WT2)
            resi.append(tempres)
            resmat[k,0:18] = [num2, num1, tempres[0][0], tempres[0][1], tempres[0][2], tempres[1][0], tempres[1][1],
                         tempres[1][2], tempres[-1][0], tempres[-1][1], tempres[-1][2],tempres[2][0], tempres[2][1],
                         tempres[-4], tempres[-3], tempres[-2],tempres[5],tempres[6]]
            resmat[k,18] = resmat[k,8]*(resmat[k,16]+resmat[k,17])
            resmat[k,19] = resmat[k,9]*resmat[k,16]
        #结果统计矩阵，0训练集阳性样本书、1阴性样本数，2-460s测试结果、5-720s测试结果、8-10级联分类器片段测试结果、11-12事件测试结果，13预测错误的事件，
        #14错失的事件、15人工标注数据中的总的时间、16测试集中的阳性片段数、17测试集中的阴性片段数、18预测正确的片段数、19预测正确的阳性片段数。
        #20人工标注的AHI，21模型计算的AHI
        resmat[-1, :] = sum(resmat[0:-1,:]) / int(1 / P)
        resmat[-1,18] = sum(resmat[0:-1,18])
        resmat[-1,19] = sum(resmat[0:-1,19])
        resmat[-1,16] = sum(resmat[0:-1,16])
        resmat[-1,17] = sum(resmat[0:-1,17])
        resmat[-1, 8] = resmat[-1,18]/(resmat[-1,16]+resmat[-1,17])
        resmat[-1, 9] = resmat[-1,19]/resmat[-1,16]
        resmat[-1, 13:16] = sum(resmat[0:-1, 13:16])
        resmat[-1, 11] = 1 - (resmat[-1, 13] / (resmat[-1, 15] - resmat[-1, 14] + resmat[-1, 13]))
        # resmat[-1, 8] = 1 - (resmat[-1, 10] / (resmat[-1, 12] - resmat[-1, 11]))
        resmat[-1, 12] = 1 - resmat[-1, 14] / resmat[-1, 15]
        resmat[-1, 20] = resmat[-1, 15] / ((resmat[-1, 16] + resmat[-1, 17] + WT2) / 3600)
        resmat[-1, 21] = (resmat[-1, 15] - resmat[-1, 14] + resmat[-1, 13]) / (
                    (resmat[-1, 16] + resmat[-1, 17] + WT2) / 3600)
        res.append(resmat)
        resave[i] = resmat[-1]
    resave[-1, :] = sum(resave[0:-1,:]) / N
    resave[-1, 18] = sum(resave[0:-1, 18])
    resave[-1, 19] = sum(resave[0:-1, 19])
    resave[-1, 16] = sum(resave[0:-1, 16])
    resave[-1, 17] = sum(resave[0:-1, 17])
    resave[-1, 8] = resave[-1, 18] / (resave[-1, 16] + resave[-1, 17])
    resave[-1, 9] = resave[-1, 19] / resave[-1, 16]
    resave[-1, 13:16] = sum(resave[0:-1, 13:16])
    resave[-1, 11] = 1 - (resave[-1, 13] / (resave[-1, 15] - resave[-1, 14] + resave[-1, 13]))
    # resave[-1, 8] = 1 - (resave[-1, 10] / (resave[-1, 12] - resave[-1, 11]))
    resave[-1, 12] = 1 - resave[-1, 14] / resave[-1, 15]
    resave[-1,20] = resave[-1,15]/((resave[-1,16]+resave[-1,17]+WT2)/3600)
    resave[-1,21] = (resave[-1,15]-resave[-1,14]+resave[-1,13])/((resave[-1,16]+resave[-1,17]+WT2)/3600)


    return res, resave


def casdata(data1, label1, data2, label2, timeind1,timeind2,WT1, WT2, P):
    #级联分类器的数据切割获得（1/P）折训练数据集、测试数据集
    l1 = len(label1)
    l2 = len(label2)
    lte = int(l1 * P)
    k = int(1 / P)
    datatrain1 = []
    datatrain2 = []
    datatest1 = []
    datatest2 = []
    labeltrain1 = []
    labeltrain2 = []
    labeltest1 = []
    labeltest2 = []
    timetest1 = []
    timetest2 = []
    for i in range(k):
        ind = np.zeros(l1, dtype=bool)
        ind[i * lte:(i + 1) * lte] = True
        datatest1.append(data1[ind])
        labeltest1.append(label1[ind])
        timetest1.append(timeind1[ind])
        indtest2 = mapwindow(WT1,WT2,timetest1[i],timeind2)
        ind2 = np.zeros(len(data2), dtype=bool)
        ind2[indtest2] = True
        datatest2.append(data2[ind2])
        labeltest2.append(label2[ind2])
        timetest2.append(timeind2[ind2])
        ind = ~ind
        ind2 = ~ind2
        datatrain1.append(data1[ind])
        labeltrain1.append(label1[ind])
        # timetrain1.append(timeind1[ind])
        datatrain2.append(data2[ind2])
        labeltrain2.append(label2[ind2])
        # timetrain2.append(timeind2[ind2])
    return datatrain1, labeltrain1, datatest1, labeltest1, timetest1,datatrain2, labeltrain2, datatest2, labeltest2,timetest2


def dataseg(data1, label1, data2, label2, WT1, WT2, P):
    #按时间直接对数据进行单次切割
    l = len(label1)
    ltr = int(l * P)
    lte = l - ltr
    datatrain1 = data1[:ltr]
    labeltrain1 = label1[:ltr]
    datatest1 = data1[ltr:]
    labeltest1 = label1[ltr:]
    ltr2 = ltr + WT1 - WT2
    datatrain2 = data2[:ltr2]
    labeltrain2 = label2[:ltr2]
    datatest2 = data2[ltr2:]
    labeltest2 = label2[ltr2:]
    return datatrain1, labeltrain1, datatest1, labeltest1, datatrain2, labeltrain2, datatest2, labeltest2


def clfcastrain(clfs, data, label, data2, label2, sust, Y, ind=[], classweight=['balanced', 'balanced']):
    #级联分类器训练函数，返回的是级联分类器与训练集的阳性样本与阴性样本数
    tempclf = []
    count = 1
    tempk = int(len(label)/sum(label))
    if classweight[0] == 1028:
        classweight[0] = {0:1,1:(tempk+3)}
    if classweight[1] == 1028:
        classweight[1] = {0:1,1:(tempk+3)}

    # sust = 20
    for i in range(len(clfs)):
        if clfs[i] == "Knn":
            tempclf.append(KNeighborsClassifier())
        elif clfs[i] == "Gau":
            tempclf.append(GaussianNB())
        elif clfs[i] == "Dec":
            if len(ind) == 0:
                comment = 'Please input the ind' + str(count) + '[min_samples_split,min_samples_leaf,max_depth]:'
                ind1 = input(comment).split()
                ind1 = [int(num) for num in ind1]
                if i == 0:
                    tempclf.append(DecisionTreeClassifier(class_weight=classweight[0], min_samples_split=int(ind1[0]),
                                                          min_samples_leaf=int(ind1[1]),
                                                          max_depth=int(ind1[2]), random_state=0))
                else:
                    tempclf.append(DecisionTreeClassifier(class_weight=classweight[1], min_samples_split=int(ind1[0]),
                                                          min_samples_leaf=int(ind1[1]),
                                                          max_depth=int(ind1[2]), random_state=0))
                count += 1
            else:
                if i == 0:
                    tempclf.append(
                        DecisionTreeClassifier(class_weight=classweight[0], min_samples_split=int(ind[i * 3]),
                                               min_samples_leaf=int(ind[i * 3 + 1]),
                                               max_depth=int(ind[i * 3 + 2]), random_state=0))
                else:
                    tempclf.append(
                        DecisionTreeClassifier(class_weight=classweight[1], min_samples_split=int(ind[i * 3]),
                                               min_samples_leaf=int(ind[i * 3 + 1]),
                                               max_depth=int(ind[i * 3 + 2]), random_state=0))
        elif clfs[i] == "Ext":
            tempclf.append(ExtraTreeClassifier())
        elif clfs[i] == "SVC":
            tempclf.append(SVC())
        elif clfs[i] == "Ran":
            if i == 0:
                tempclf.append(RandomForestClassifier(bootstrap = True, oob_score= True, n_jobs=4, random_state=0,max_features=1,\
                                   class_weight = classweight[0],max_depth=ind[i*3+2],n_estimators=10,min_samples_split=ind[i*3],min_samples_leaf=ind[i*3+1]))
            else:
                tempclf.append(RandomForestClassifier(bootstrap = True, oob_score= True, n_jobs=4, random_state=0,max_features=1,\
                                   class_weight = classweight[1],max_depth=ind[i*3+2],n_estimators=20,min_samples_split=ind[i*3],min_samples_leaf=ind[i*3+1]))
    # 降采样
    # nump = sum(label)
    # numn = len(label)-nump
    # # tempk = int(numn/nump)
    # if numn > nump:
    #     tempn = numn-nump
    #     tempran = np.random.rand(numn)
    #     tempsort = tempran.argsort()
    #     tempind = tempsort[0:tempn]
    #     ind_n = np.where(label == 0)
    #     ind_forn = ind_n[0][tempind]
    #     data = np.delete(data,ind_forn,axis=0)
    #     label = np.delete(label,ind_forn,axis=0)

    tempclf[0].fit(data, label)
    temppre = tempclf[0].predict(data)
    ind1 = []
    ind1.append(accuracy_score(label, temppre))
    ind1.append(recall_score(label, temppre))
    ind1.append(precision_score(label, temppre))
    tempvar1, tempvar2, tempvar3, tempvar4 = AHIcal(temppre, 50)
    sec = np.zeros(len(label2), dtype=bool)
    for i in range(tempvar1):
        sec[(tempvar3[i] - sust):(tempvar4[i] + 60 - 20 + sust)] = True
    map = np.where(sec == True)

    # data2 = data[sec]
    # label2 = label[sec]

    tempclf[1].fit(data2, label2)
    if Y:
        drawtree(tempclf[0], 'tree1')
        drawtree(tempclf[1], 'tree2')
    temppre = tempclf[1].predict(data2)
    ind2 = []
    ind2.append(accuracy_score(label2, temppre))
    ind2.append(recall_score(label2, temppre))
    ind2.append(precision_score(label2, temppre))
    # datawait = data2[sec]
    # labelwait = label2[sec]
    # var1, var2, var3, var4 = AHIcal(label2, 1)
    # if len(datawait) != 0:
    #     # tempclf[1].fit(datawait, labelwait)
    #     temppre = tempclf[1].predict(datawait)
    #     ind2 = []
    #     ind2.append(accuracy_score(labelwait, temppre))
    #     ind2.append(recall_score(labelwait, temppre))
    #     ind2.append(precision_score(labelwait, temppre))
    #
    #     var5, var6, var7, var8 = AHIcal(temppre, 10)
    #     var7 = map[0][var7]
    #     var8 = map[0][var8]
    #     res = []
    #     flag = [0] * var5
    #     wrongres = []
    #     eva = []
    #     for i in range(var5):
    #         # tempflag = []
    #         for k in range(var1):
    #             if (var7[i] >= var3[k] and var7[i] < var4[k]) or (var8[i] > var3[k] and var8[i] <= var4[k]) or (
    #                     var7[i] <= var3[k] and var8[i] >= var4[k]):
    #                 res.append([i, k])
    #                 flag[i] = 1
    #         # res.append(tempflag)
    #         if flag[i] == 0:
    #             wrongres.append([var7[i], var8[i]])
    #     ########################
    #     # eva[0]是精准率
    #     if len(flag) == 0:
    #         eva.append(0)
    #     else:
    #         eva.append(100 * sum(flag) / len(flag))
    #     flag2 = [0] * var1
    #     missres = []
    #     for item in res:
    #         flag2[item[1]] = 1
    #     for i in range(var1):
    #         if flag2[i] == 0:
    #             missres.append([var3[i], var4[i]])
    #     ########################
    #     # eva[0]是召回率
    #     if var1 != 0:
    #         eva.append(100 * (var1 - len(missres)) / var1)
    #     else:
    #         eva.append(100 * (var1 - len(missres)))
    #     # wrongloc = []
    #     # for item in wrongres:
    #     #     wrongloc.append([map[0][item[0]],map[0][item[1]]])
    #     # missloc = []
    #     # for item in missres:
    #     #     missloc.append([map[0][item[0]],map[0][item[1]]])
    # else:
    #     ind2 = [0, 0, 0]
    #     if var1 != 0:
    #         eva = [0, 0]
    #         wrongres = []
    #         missres = []
    #         for k in range(var1):
    #             missres.append([var3[k], var4[k]])
    #
    #     else:
    #         eva = [100, 100]
    #         wrongres = []
    #         missres = []

    return tempclf, sum(label2), len(label2) - sum(label2)

def eventdetect(var1,var3,var4,var5,var7,var8):
    #var1，标签事件总数
    #var3，标签事件起点，时间索引
    #var4，标签事件终点，时间索引
    #var5，预测数据总数
    #var7，预测事件起点，时间索引
    #var8，预测事件终点，时间索引
    #返回的是wrongres,missres,eva,res
    #wrongres，错误的预测事件
    #missres，遗漏的预测事件
    #eva，事件的评价指标
    #res，正确的事件
    res = []
    flag = [0] * var5
    wrongres = []
    eva = []
    for i in range(var5):
        # tempflag = []
        for k in range(var1):
            if (var7[i] >= var3[k] and var7[i] < var4[k]) or (var8[i] > var3[k] and var8[i] <= var4[k]) or (
                    var7[i] <= var3[k] and var8[i] >= var4[k]):
                res.append([i, k])
                flag[i] = 1
        # res.append(tempflag)
        if flag[i] == 0:
            wrongres.append([var7[i], var8[i]])
    if var5 != 0:
        eva.append(100 * sum(flag) / len(flag))
    else:
        eva.append(0)
    flag2 = [0] * var1
    missres = []
    for item in res:
        flag2[item[1]] = 1
    for i in range(var1):
        if flag2[i] == 0:
            missres.append([var3[i], var4[i]])
    if var1 != 0:
        eva.append(100 * (var1 - len(missres)) / var1)
    else:
        eva.append(100 * (var1 - len(missres)))
    return wrongres,missres,eva,res
def clfcastest(clfs, data, label, timeind1,data2, label2,timeind2, sust, Y,WT1,WT2):
    #############################
    #clfs，级联分类器，list类型
    #data，60s测试数据
    #data2，20s测试数据
    #sust，60s窗口延长时间
    #Y，是否画出决策树标志
    #############################
    tempclf = clfs  #集成刚才训练好的分类器
    temppre = tempclf[0].predict(data)  #60s预测输出
    #event detect and smooth
    temppre = prefilter(label,timeind1,temppre,WT1,5)
    timeroi, temps, tempe = roidetect(temppre, timeind1)
    indtest = mapwindow(WT1, WT2,  timeroi, timeind2)
    #在级联分类器1的预测结果基础上寻找级联分类器2的输入
    ind1 = []   #ind1，60s数据测试性能
    ind1.append(accuracy_score(label, temppre))
    ind1.append(recall_score(label, temppre))
    ind1.append(precision_score(label, temppre))
    res20 = np.zeros(len(label2))
    sec = np.zeros(len(label2), dtype=bool)
    sec[indtest] = True
    # map = np.where(sec == True)     #获得data2中的阳性窗口的索引
    if Y:
        drawtree(tempclf[1], 'tree')
    datawait = data2[sec]   #从60s阳性窗口中继承20s阳性窗口
    labelwait = label2[sec]   #从60s阳性窗口中继承20s阳性窗口
    timewait = timeind2[sec]
    label2 = prefilter(label2,timeind2,label2,WT2,5)
    var2,var3,var4 = roidetect(label2,timeind2)
    var3 = ind2time(var2,var3)
    var4 = ind2time(var2,var4)
    var1 = len(var3)
    if len(datawait) != 0:
        temppre = tempclf[1].predict(datawait)
        tempob = tempclf[1].predict(data2)
        tempob = prefilter(label2, timeind2, tempob, WT2, 5)
        temppre = prefilter(labelwait,timewait,temppre,WT2,5)
        sec = ~sec
        tempob[sec] = 0
        ind2 = []
        ind2.append(accuracy_score(labelwait, temppre))
        ind2.append(recall_score(labelwait, temppre))
        ind2.append(precision_score(labelwait, temppre))
        ind3 = []
        ind3.append(accuracy_score(label2, tempob))
        ind3.append(recall_score(label2, tempob))
        ind3.append(precision_score(label2, tempob))
        var6,var7,var8 = roidetect(temppre,timewait)
        var5 = len(var7)
        var7 = ind2time(var6,var7)
        var8 = ind2time(var6,var8)
        var7 = time2ind(timeind2,var7)
        var8 = time2ind(timeind2,var8)
        for i in range(len(var7)):
            res20[var7[i]:var8[i]+1] = 1
        segscore = []
        segscore.append(accuracy_score(label2,res20))
        segscore.append(recall_score(label2, res20))
        segscore.append(precision_score(label2, res20))
        var7 = ind2time(timeind2,var7)
        var8 = ind2time(timeind2,var8)
        #var3，var4，var7，var8都应该是时间索引
        wrongres,missres,eva,res = eventdetect(var1,var3,var4,var5,var7,var8)

    else:
        ind2 = [0, 0, 0]
        segscore = []
        segscore.append(accuracy_score(label2,res20))
        segscore.append(recall_score(label2, res20))
        segscore.append(precision_score(label2, res20))
        if var1 != 0:
            # ind2 = [0, 0, 0]
            eva = [0, 0]
            wrongres = []
            missres = []
            for k in range(var1):
                missres.append([var3[k], var4[k]])
        else:
            eva = [100, 100]
            wrongres = []
            missres = []
    #返回的是60s窗口的测试结果、20s窗口的测试结果、事件的测试结果、错误的事件、错失的事件、测试集的阳性样本数、测试集的阴性样本数
    #错误事件的个数、错失事件的个数、人工标注数据的事件个数、级联分类器的分类结果。
    return ind1, ind2, eva, wrongres, missres, sum(label2), len(label2) - sum(label2), len(wrongres), len(missres), var1,segscore
    # # plabel,slabel,elabel,dlabel = durdetect(label,timeind1)
    # # # ob1, ob2, ob3, ob4 = AHIcal(label, 1)
    # # # tempvar1, tempvar2, tempvar3, tempvar4,temppre = smoothres(temppre, 50)    #从60s片段之中获得apnea事件
    # # clfind1 = np.where(temppre == 1)
    # # clfind1 = clfind1[0]
    # # timepre = timeind1[clfind1]
    # # indtest = mapwindow(WT1,WT2,timepre,timeind2)
    #
    # # for i in range(tempvar1):
    # #     sec[indtest] = True     #筛出60s窗口中的阳性窗口
    #
    # var1, var2, var3, var4 = AHIcal(label2, 1)  #计算人工标注中的ah事件
    # plabel, slabel, elabel, dlabel = durdetect(label2, timeind2)



def clfcas(clfs, data, label, data2, label2, sust, Y):
    tempclf = []
    count = 1
    # sust = 20
    for item in clfs:
        if item == "Knn":
            tempclf.append(KNeighborsClassifier())
        elif item == "Gau":
            tempclf.append(GaussianNB())
        elif item == "Dec":
            comment = 'Please input the ind' + str(count) + '[min_samples_split,min_samples_leaf,max_depth]:'
            ind1 = input(comment).split()
            ind1 = [int(num) for num in ind1]

            tempclf.append(DecisionTreeClassifier(class_weight='balanced', min_samples_split=int(ind1[0]),
                                                  min_samples_leaf=int(ind1[1]),
                                                  max_depth=int(ind1[2])))
            count += 1
        elif item == "Ext":
            tempclf.append(ExtraTreeClassifier())
        elif item == "SVC":
            tempclf.append(SVC())
    tempclf[0].fit(data, label)
    temppre = tempclf[0].predict(data)
    ind1 = []
    ind1.append(accuracy_score(label, temppre))
    ind1.append(recall_score(label, temppre))
    ind1.append(precision_score(label, temppre))
    tempvar1, tempvar2, tempvar3, tempvar4 = AHIcal(temppre, 5)
    sec = np.zeros(len(label2), dtype=bool)
    for i in range(tempvar1):
        sec[(tempvar3[i] - sust):(tempvar4[i] + 60 - 20 + sust)] = True
    map = np.where(sec == True)

    # data2 = data[sec]
    # label2 = label[sec]

    tempclf[1].fit(data2, label2)
    if Y:
        drawtree(tempclf[1], 'tree')
    datawait = data2[sec]
    labelwait = label2[sec]
    # tempclf[1].fit(datawait, labelwait)
    temppre = tempclf[1].predict(datawait)
    ind2 = []
    ind2.append(accuracy_score(labelwait, temppre))
    ind2.append(recall_score(labelwait, temppre))
    ind2.append(precision_score(labelwait, temppre))
    var1, var2, var3, var4 = AHIcal(labelwait, 1)
    var5, var6, var7, var8 = AHIcal(temppre, 10)

    res = []
    flag = [0] * var5
    wrongres = []
    eva = []
    for i in range(var5):
        # tempflag = []
        for k in range(var1):
            if (var7[i] >= var3[k] and var7[i] < var4[k]) or (var8[i] > var3[k] and var8[i] <= var4[k]) or (
                    var7[i] <= var3[k] and var8[i] >= var4[k]):
                res.append([i, k])
                flag[i] = 1
        # res.append(tempflag)
        if flag[i] == 0:
            wrongres.append([var7[i], var8[i]])
    eva.append(100 * sum(flag) / len(flag))
    flag2 = [0] * var1
    missres = []
    for item in res:
        flag2[item[1]] = 1
    for i in range(var1):
        if flag2[i] == 0:
            missres.append([var3[i], var4[i]])
    eva.append(100 * (var1 - len(missres)) / var1)
    wrongloc = []
    for item in wrongres:
        wrongloc.append([map[0][item[0]], map[0][item[1]]])
    missloc = []
    for item in missres:
        missloc.append([map[0][item[0]], map[0][item[1]]])

    return ind1, ind2, eva, wrongres, missres


def AHItrain(clfs, data, label):
    ##################################
    # clfs,分类器名称
    # data，带训练数据
    # label，带训练标签
    # 返回的是片段的分类评价指标，错误分类的ahi区间，区间准确度精准度，漏掉的ahi区间
    ##################################
    ind = []
    # for item in clfs:
    #     if item == "Knn":
    #         tempclf = KNeighborsClassifier()
    #     elif item == "Gau":
    #         tempclf = GaussianNB()
    #     elif item == "Dec":
    #         tempclf = DecisionTreeClassifier(class_weight='balanced', min_samples_split=50, min_samples_leaf=100,
    #                                          max_depth=5)
    #     elif item == "Ext":
    #         tempclf = ExtraTreeClassifier()
    #     elif item == "SVC":
    #         tempclf = SVC()
    tempclf = clfs
    tempclf.fit(data, label)
    # drawtree(tempclf,'tree')
    pre = tempclf.predict(data)
    ind.append(accuracy_score(label, pre))
    ind.append(recall_score(label, pre))
    ind.append(precision_score(label, pre))
    var1, var2, var3, var4 = AHIcal(label, 1)
    # var5,var6,var7,var8 = smoothres(pre,20)
    var5, var6, var7, var8 = AHIcal(pre, 5)
    res = []
    flag = [0] * var5
    wrongres = []
    eva = []
    for i in range(var5):
        # tempflag = []
        for k in range(var1):
            if (var7[i] > var3[k] and var7[i] < var4[k]) or (var8[i] > var3[k] and var8[i] < var4[k]) or (
                    var7[i] < var3[k] and var8[i] > var4[k]):
                res.append([i, k])
                flag[i] = 1
        # res.append(tempflag)
        if flag[i] == 0:
            wrongres.append([var7[i], var8[i]])
    eva.append(100 * sum(flag) / len(flag))
    flag2 = [0] * var1
    missres = []
    for item in res:
        flag2[item[1]] = 1
    for i in range(var1):
        if flag2[i] == 0:
            missres.append([var3[i], var4[i]])
    eva.append(100 * (var1 - len(missres)) / var1)
    return tempclf


def AHItest(clfs, data, label):
    ##################################
    # clfs,分类器名称
    # data，带训练数据
    # label，带训练标签
    # 返回的是片段的分类评价指标，错误分类的ahi区间，区间准确度精准度，漏掉的ahi区间
    ##################################
    ind = []
    tempclf = clfs
    # tempclf.fit(data,label)
    # drawtree(tempclf,'tree')
    pre = tempclf.predict(data)
    ind.append(accuracy_score(label, pre))
    ind.append(recall_score(label, pre))
    ind.append(precision_score(label, pre))
    var1, var2, var3, var4 = AHIcal(label, 1)
    # var5,var6,var7,var8 = smoothres(pre,20)
    var5, var6, var7, var8 = AHIcal(pre, 10)
    res = []
    flag = [0] * var5
    wrongres = []
    eva = []
    for i in range(var5):
        # tempflag = []
        for k in range(var1):
            if (var7[i] > var3[k] and var7[i] < var4[k]) or (var8[i] > var3[k] and var8[i] < var4[k]) or (
                    var7[i] < var3[k] and var8[i] > var4[k]):
                res.append([i, k])
                flag[i] = 1
        # res.append(tempflag)
        if flag[i] == 0:
            wrongres.append([var7[i], var8[i]])
    eva.append(100 * sum(flag) / len(flag))
    flag2 = [0] * var1
    missres = []
    for item in res:
        flag2[item[1]] = 1
    for i in range(var1):
        if flag2[i] == 0:
            missres.append([var3[i], var4[i]])
    eva.append(100 * (var1 - len(missres)) / var1)
    return ind, wrongres, eva, missres, var1


def AHIres(clfs, data, label):
    ##################################
    # clfs,分类器名称
    # data，带训练数据
    # label，带训练标签
    # 返回的是片段的分类评价指标，错误分类的ahi区间，区间准确度精准度，漏掉的ahi区间
    ##################################
    ind = []
    for item in clfs:
        if item == "Knn":
            tempclf = KNeighborsClassifier()
        elif item == "Gau":
            tempclf = GaussianNB()
        elif item == "Dec":
            tempclf = DecisionTreeClassifier(class_weight='balanced', min_samples_split=50, min_samples_leaf=100,
                                             max_depth=5)
        elif item == "Ext":
            tempclf = ExtraTreeClassifier()
        elif item == "SVC":
            tempclf = SVC()
    tempclf.fit(data, label)
    drawtree(tempclf, 'tree')
    pre = tempclf.predict(data)
    ind.append(accuracy_score(label, pre))
    ind.append(recall_score(label, pre))
    ind.append(precision_score(label, pre))
    var1, var2, var3, var4 = AHIcal(label, 1)
    # var5,var6,var7,var8 = smoothres(pre,20)
    var5, var6, var7, var8 = AHIcal(pre, 5)
    res = []
    flag = [0] * var5
    wrongres = []
    eva = []
    for i in range(var5):
        # tempflag = []
        for k in range(var1):
            if (var7[i] > var3[k] and var7[i] < var4[k]) or (var8[i] > var3[k] and var8[i] < var4[k]) or (
                    var7[i] < var3[k] and var8[i] > var4[k]):
                res.append([i, k])
                flag[i] = 1
        # res.append(tempflag)
        if flag[i] == 0:
            wrongres.append([var7[i], var8[i]])
    eva.append(100 * sum(flag) / len(flag))
    flag2 = [0] * var1
    missres = []
    for item in res:
        flag2[item[1]] = 1
    for i in range(var1):
        if flag2[i] == 0:
            missres.append([var3[i], var4[i]])
    eva.append(100 * (var1 - len(missres)) / var1)
    return ind, wrongres, eva, missres


def AHIval(clfs, data, label):
    ###########################################
    # clfs，待检验分类器
    # data，切分好的数据集
    # label，切分好的标签集
    ###########################################
    tempclfs = []
    ind = []
    for item in clfs:
        if item == "Knn":
            tempclf = KNeighborsClassifier()
        elif item == "Gau":
            tempclf = GaussianNB()
        elif item == "Dec":
            tempclf = DecisionTreeClassifier(class_weight='balanced', min_samples_split=50, min_samples_leaf=500,
                                             max_depth=5)
        elif item == "Ext":
            tempclf = ExtraTreeClassifier()
        elif item == "SVC":
            tempclf = SVC()
        tempclfs.append(tempclf)
    clfs = tempclfs
    respre = {}
    res = {}
    for clf in clfs:
        l = len(data)
        respre.setdefault(str(clf)[0:3], {})
        for i in range(l):
            templ = len(data[i])
            respre[str(clf)[0:3]].setdefault(i, [])
            res.setdefault(i, [])
            for k in range(templ):
                tempind = []
                datatrain = data[i][k]
                datatest = data[i][1 - k]
                labeltrain = label[i][k]
                labeltest = label[i][1 - k]
                clf.fit(datatrain, labeltrain)
                temppre = clf.predict(datatest)
                tempind.append(accuracy_score(labeltest, temppre))
                tempind.append(recall_score(labeltest, temppre))
                tempind.append(precision_score(labeltest, temppre))
                var1, var2, var3, var4 = smoothres(temppre)
                var5, var6, var7, var8 = AHIcal(labeltest)
                respre[str(clf)[0:3]][i].append([var1, var2])
                res[i].append([var5, var6])
                ind.append(tempind)
    return res, respre, ind


def AHIcal(label, thre=50):
    ###########################################
    # 计算AHI的函数
    # label按时间排列的标签
    # 返回的是aha次数以及每次持续的时长以及相应的开始以及结束的时间
    ###########################################
    templ = len(label)
    startflag = 0
    endflag = 0
    tempcount = 0
    cachelen = []
    cachestart = []
    cacheend = []
    aha = 0
    # thre = 50
    for i in range(1, templ):
        if label[i - 1] == 0 and label[i] == 1:
            startflag = 1
            endflag = 0
            tempstart = i
            # cachestart.append(i)
        if label[i - 1] == 1 and label[i] == 0:
            endflag = 1
            startflag = 0
            tempend = i
            # cacheend.append(i)
        if startflag == 1 and endflag == 0:
            tempcount += 1
        if endflag == 1 and startflag == 0:
            if tempcount >= thre:
                aha += 1
                cachelen.append(tempcount)
                cachestart.append(tempstart)
                cacheend.append(tempend)
            startflag = 0
            endflag = 0
            tempcount = 0
            tempstart = 0
            tempend = 0
    return aha, cachelen, cachestart, cacheend


def smoothres(label):
    l = len(label)
    output = label
    for i in range(4, l - 3):
        if label[i - 3] == 1 and label[i - 2] == 1 and label[i - 1] == 1 and label[i] == 0 \
                and label[i + 1] == 1 and label[i + 2] == 1 and label[i + 3] == 1:
            output[i] = 1
        if label[i - 3] == 0 and label[i - 2] == 0 and label[i - 1] == 0 and label[i] == 1 \
                and label[i + 1] == 0 and label[i + 2] == 0 and label[i + 3] == 0:
            output[i] = 0
    # var1, var2, var3, var4 = AHIcal(output, thre)
    return output


def OB(data, label, name, max_depth, min_samples_split, min_samples_leaf):
    ########################################
    # data,训练数据
    # label，训练标签
    # name，输出文件名称
    # max_depth，决策树最大深度
    # min samples split...均为决策树参数
    #########################################

    clf = DecisionTreeClassifier(class_weight='balanced', max_depth=max_depth, \
                                 min_samples_split=min_samples_split, min_samples_leaf \
                                     =min_samples_leaf, random_state=0)
    plottree(clf, data, label, name)


def plottree(clf, data, label, name):
    ##############################
    # clf,分类器
    # data，训练数据
    # label，数据标签
    ##############################
    clf.fit(data, label)

    score_a = cross_val_score(clf, data, label, scoring='accuracy', cv=5, n_jobs=4)
    score_r = cross_val_score(clf, data, label, scoring='recall', cv=5, n_jobs=4)
    score_p = cross_val_score(clf, data, label, scoring='precision', cv=5, n_jobs=4)
    res = [score_a.mean(), score_r.mean(), score_p.mean()]
    dot_data = export_graphviz(clf, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render(name)
    print(res)


def drawtree(clf, name):
    # clf.fit(data,label)
    dot_data = export_graphviz(clf, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render(name)


def load_data(filename, N):
    ########################
    # 读取特征xlsx文件
    # 第一行是文字说明
    # 23个被试存储在不同的sheet中
    ########################

    data = pd.read_excel(filename, sheet_name=None)
    data_train = []
    label = []
    timeind = []
    if N == 0:
        N = 23
        for i in range(N):
            label.append(data[str(i + 1)].iloc[:, -2].values)
            data_train.append(data[str(i + 1)].iloc[:, 0:-2].values)
            timeind.append(data[str(i + 1)].iloc[:, -1].values)
        data_train = np.array(data_train)
        label = np.array(label)
        timeind = np.array(timeind)
    else:
        label.append(data[str(N)].iloc[:, -2].values)
        data_train.append(data[str(N)].iloc[:, 0:-2].values)
        timeind.append(data[str(N)].iloc[:, -1].values)
        data_train = np.array(data_train)
        label = np.array(label)
        timeind = np.array(timeind)
    return data_train, label, timeind


########################
# 训练四种分类器，last editted 9-8
########################
def train_97(data, label, P, N):
    # lsub = len(data)
    #######################
    # 输入项：
    # data训练数据
    # label标签
    # P训练集合比例
    # N待分类的被试数目
    # 采用直接按比例划分数据集的方式
    #######################
    clf1 = SVC(kernel='rbf', C=0.5, gamma=1, max_iter=1e7, tol=1e-5, cache_size=500, shrinking=False)
    # par1 = {'C':[0.1,0.5,1,1.1,1.2,1.3,1.4,1.5,2.0,3.0]}
    par1 = {'C': [0.9, 1.0, 1.1]}
    clf2 = GaussianNB()
    par2 = {}
    clf3 = KNeighborsClassifier()
    par3 = {'algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute'), 'n_neighbors': [3, 4, 5, 6, 7]}
    clf4 = AdaBoostClassifier()
    par4 = {'n_estimators': [30, 35, 40, 45, 50, 55, 60], 'learning_rate': [0.1, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]}
    clf = [clf1, clf2, clf3, clf4]
    par = [par1, par2, par3, par4]
    res = []
    for i in range(N):
        templ = len(data[i])
        templtr = int(templ * P)
        templte = templ - templtr
        scaler = StandardScaler()
        datatrain = data[i][0:templtr]
        datatest = data[i][templtr:]
        latr = label[i][0:templtr]
        late = label[i][templtr:]
        #######################
        # 输入项标准化
        #######################
        scaler.fit(datatrain)
        temptr = scaler.transform(datatrain)
        tempte = scaler.transform(datatest)
        #######################
        # 输入项未标准化
        #######################
        # temptr = datatrain
        # tempte = datatest

        tempbin = []
        for item in clf:
            tempres = {}
            tempres.setdefault(str(item)[0:3], [])
            item.fit(temptr, latr)
            temppre = item.predict(tempte)
            tempres[str(item)[0:3]].append(accuracy_score(late, temppre))
            tempres[str(item)[0:3]].append(precision_score(late, temppre))
            tempres[str(item)[0:3]].append(recall_score(late, temppre))
            tempbin.append(tempres)

            del temppre, tempres
        res.append(tempbin)
        del templ, templtr, templte, temptr, tempte, latr, late

    return res


def train_911(data, label, P, N, N_seg):
    #######################
    # 输入项：
    # data训练数据
    # label标签
    # P训练集合比例
    # N待分类的被试数目
    # 数据切割段数
    # 针对于每个被试每一折数据的超参数寻优
    #######################
    ##########################
    # 准备分类器
    ##########################
    clf1 = SVC(class_weight='balanced')
    par1 = {'C': [0.1, 0.5, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 3.0], 'gamma': [0.1, 0.5, 1.0, 1.5, 2.0]}
    clf2 = GaussianNB()
    par2 = {}
    clf3 = AdaBoostClassifier()
    par3 = {'n_estimators': [30, 35, 40, 45, 50, 55, 60], 'learning_rate': [0.1, 0.2, 0.5, 1.0, 1.5, 2.0]}
    # clf4 = KNeighborsClassifier()
    # par4 = {'algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute'), 'n_neighbors': [3, 4, 5]}
    scaler = StandardScaler()
    # clf = [clf1,clf2,clf3]
    # pars = [par1, par2, par3]
    clf = [clf1]
    pars = [par1]
    res = {}
    res.setdefault('l', int(1 / P))
    ##########################
    # 准备训练集合与测试集合
    ##########################
    for i in range(N):
        templ = len(data[i])
        l_seg = int(templ / N_seg)
        l_segte = int(l_seg * P)
        l_segtr = l_seg - l_segte
        tempdata = []
        templabel = []
        res.setdefault(i, {})
        for k in range(N_seg):
            tempdata.append(data[i][k * l_seg:(k + 1) * l_seg])
            templabel.append(label[i][k * l_seg:(k + 1) * l_seg])

        for count in range(len(clf)):
            res[i].setdefault(str(clf[count])[0:3], {})
            for k in range(int(1 / P)):
                ind = np.zeros(l_seg)
                ind[k * l_segte:(k + 1) * l_segte] = 1
                indte = np.where(ind == 1)
                # ind = np.ones(ind.shape[0]) - ind
                indtr = np.where(ind == 0)
                res[i][str(clf[count])[0:3]].setdefault(k, {})

                for count2 in range(int(1 / P)):
                    temptrain = tempdata[count2][indtr]
                    temptest = tempdata[count2][indte]
                    templatr = templabel[count2][indtr]
                    template = templabel[count2][indte]
                    if count2 == 0:
                        dtrain = temptrain
                        dtest = temptest
                        ltrain = templatr
                        ltest = template
                    else:
                        dtrain = np.vstack((dtrain, temptrain))
                        dtest = np.vstack((dtest, temptest))
                        ltrain = np.hstack((ltrain, templatr))
                        ltest = np.hstack((ltest, template))
                    del temptrain, temptest, templatr, template
                #######################
                # 输入项标准化
                #######################
                scaler.fit(dtrain)
                temptr = scaler.transform(dtrain)
                tempte = scaler.transform(dtest)
                #########################
                # 分类器训练与分类
                #########################

                tempclf1 = GridSearchCV(clf[count], pars[count], 'recall', n_jobs=4)
                tempclf2 = GridSearchCV(clf[count], pars[count], 'roc_auc', n_jobs=4)
                tempclf1.fit(temptr, ltrain)
                tempclf2.fit(temptr, ltrain)
                # temppre = clf[i].predict(tempte)
                res[i][str(clf[count])[0:3]].setdefault(k, {})
                res[i][str(clf[count])[0:3]][k].setdefault('best_recall', tempclf1.best_score_)
                res[i][str(clf[count])[0:3]][k].setdefault('best_recall_parm', tempclf1.best_params_)
                res[i][str(clf[count])[0:3]][k].setdefault('best_precision', tempclf2.best_score_)
                res[i][str(clf[count])[0:3]][k].setdefault('best_precision_parm', tempclf2.best_params_)
                # del temppre
                # del ltrain,ltest,dtrain,dtest
    return (res)


def parop(data, label, l, N,classweight = 'balanced'):
    ###########################
    # 超参数调优，l：最后评价参数个数，N被试个数
    ###########################
    ###########################
    # 1号分类器，SVM
    ###########################
    # clf1 = SVC(class_weight='balanced')
    # par1 = {'C': [0.1, 0.5, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 3.0],'gamma':[0.1,0.5,1.0,1.5,2.0]}
    # par1 = {'C' : [1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1],'gamma':[1.5,2.0,2.5]}
    # par1 = {'C': [1, 2, 3, 4, 5, 6], 'gamma': [1.5, 2.5, 3.5, 4.5]}
    # par1 = {'C':[1.6,1.65,1.7,1.75,1.8],'gamma':[1.5,1.6,1.7,1.8,1.9,2.0,2.1]}
    # par1 = {'C':[0.05,0.1,0.15,0.2],'gamma':[1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]}
    # par1 = {'C': [0.05, 0.1, 0.15, 0.2], 'gamma': [0.5,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00]}
    ###########################
    # 二号分类器，贝叶斯
    ###########################
    # clf2 = GaussianNB()
    # par2 = {}
    ###########################
    # 三号分类器，Ada
    ###########################
    # clf3 = AdaBoostClassifier()
    # # # par3 = {'learning_rate': [0.1, 0.2, 0.5, 1.0,  1.5, 2.0],'n_estimators': [30, 35, 40, 45, 50, 55, 60]}
    # par3 = {'learning_rate': [1.5], 'n_estimators': [35], \
    #         'base_estimator': [DecisionTreeClassifier()]}
    #
    # # par3 = {'n_estimators':[50,55,60,65,70],'learning_rate':[0.1,0.2,0.3,0.4,0.5]}
    ###########################
    # 四号分类器与五号分类器，DT与ET
    ###########################
    # clf4 = DecisionTreeClassifier(class_weight=classweight, random_state=0)
    # clf5 = ExtraTreeClassifier(class_weight='balanced',random_state=0)
    # par4 = {'min_samples_split':[2,100,200],'min_samples_leaf':[1,50,100,150]}
    # par4 = {'min_samples_split':[50,150,250,350,450,550,1000],\
    #         'min_samples_leaf':[100,250,500,750,1000],'max_depth':[3,4,5]}
    # par4 = {'min_samples_split': [100], \
    #         'min_samples_leaf': [100], 'max_depth': [10]}
    # par4 = {'min_samples_split': [100,300,500,700,900, 1000], \
    #         'min_samples_leaf': [100, 150,200,250,300,400, 500,], 'max_depth': [ 4, 5]}
    # par4 = {'min_samples_split': [100, 300, 500, 700, 900, 1000], \
    #         'min_samples_leaf': [500,600,700,800,1000,1500 ], 'max_depth': [5,6,7]}
    # par4 = {'min_samples_split': [100, 300, 500, 700, 900, 1000], \
    #         'min_samples_leaf': [450,460,470,480,490,500,510,520,530,540,550], 'max_depth': [5]}
    # clf6 = KNeighborsClassifier(n_jobs=4)
    # par6 = {'n_neighbors':[3,4,5]}
    # clf7 = RadiusNeighborsClassifier()
    # par7 = {'radius':[0.5,1.0,1.5]}
    ###########################################
    # ensemble方法
    ###########################################
    # clf8 = BaggingClassifier(max_features=1.0,\
    #                          oob_score=True,n_jobs=4,random_state=0)
    # par8 = {'base_estimator':[DecisionTreeClassifier(class_weight='balanced'),\
    #                           ExtraTreeClassifier(class_weight='balanced')],\
    #         'n_estimators':[10,20,30,40,50],'max_samples':[0.2,0.4,0.5,1.0],'max_features':[0.5]}
    # clf9 = ExtraTreesClassifier(class_weight='balanced',random_state = 0,max_depth=10,\
    #                            n_jobs=4, oob_score=True,bootstrap=True)
    #
    # par9 = {'n_estimators':[20],'min_samples_split':[100],\
    #         'min_samples_leaf':[150]}
    clf10 = RandomForestClassifier(bootstrap = True, oob_score= True, n_jobs=4, random_state=0,\
                                   class_weight = 'balanced',max_depth=50)
    par10 = {'n_estimators':[20],'min_samples_split':[30],\
             'min_samples_leaf':[30]}
    # clf = [clf1,clf2,clf3]
    # pars = [par1, par2, par3]
    clf = [clf10]
    pars = [par10]
    res = {}
    scoring = ['accuracy', 'recall', 'precision']
    for count in range(len(clf)):
        res.setdefault(str(clf[count])[0:3], {})
        for i in range(N):
            res[str(clf[count])[0:3]].setdefault(i, {})
            for scoler in scoring:
                tempclf = GridSearchCV(clf[count], pars[count], cv=2, scoring=scoler, n_jobs=4, return_train_score=True)
                tempclf.fit(data[i], label[i])
                score = tempclf.cv_results_
                res[str(clf[count])[0:3]][i].setdefault("test" + scoler, score['mean_test_score'])
                res[str(clf[count])[0:3]][i].setdefault("train" + scoler, score['mean_train_score'])
    return res, pars


def createahiset(data, label, P, N):
    ##########################
    # data数据集
    # label标签集合
    # P切割集合的比例
    # N被试数目
    ##########################
    datares = []
    labelres = []
    for i in range(N):
        templ = len(data[i])
        segl = int(templ * P)
        tempdata = []
        templabel = []
        for k in range(int(1 / P)):
            tempdata.append(data[i][k * segl:(k + 1) * segl])
            templabel.append(label[i][k * segl:(k + 1) * segl])
        datares.append(tempdata)
        labelres.append(templabel)
    return datares, labelres


def createdataset(data, label, timeind, P, N, N_seg):
    #######################
    # 输入项：
    # data训练数据
    # label标签
    # timeind时间信息
    # P训练集合比例
    # N待分类的被试数目
    # 数据切割段数
    # ahiflag,是否准备计算ahi标志
    # 产生一个将原始样本按时间轴均匀切割然后按时间排布抽取的数据集
    #######################
    ##########################
    # 准备分类器
    ##########################
    dataset = []
    labelset = []
    timeset = []
    for i in range(N):
        templ = len(data[i])
        l_seg = int(templ / N_seg)
        l_segte = int(l_seg * P)
        l_segtr = l_seg - l_segte
        tempdata = []
        templabel = []
        temptime = []
        dataset.append([])
        labelset.append([])
        timeset.append([])

        for k in range(N_seg):
            tempdata.append(data[i][k * l_seg + 1:(k + 1) * l_seg])
            templabel.append(label[i][k * l_seg + 1:(k + 1) * l_seg])
            temptime.append(timeind[i][k * l_seg + 1:(k+1) * l_seg])
        for count2 in range(int(1 / P)):
            for k in range(N_seg):
                temptrain = tempdata[k][count2 * l_segte + 1:(count2 + 1) * l_segte]
                templatr = templabel[k][count2 * l_segte + 1:(count2 + 1) * l_segte]
                temptind = temptime[k][count2 * l_segte + 1:(count2 + 1) * l_segte]
                if count2 == 0 and k == 0:
                    dataset[i] = temptrain
                    labelset[i] = templatr
                    timeset[i] = temptind
                else:
                    dataset[i] = np.vstack((dataset[i], temptrain))
                    labelset[i] = np.hstack((labelset[i], templatr))
                    timeset[i] = np.hstack((timeset[i], temptind))
                del temptrain, templatr, temptind
    dataset = np.array(dataset)
    labelset = np.array(labelset)
    timeset = np.array(timeset)
    return dataset, labelset, timeset


def train_912(data, label, P, N, N_seg):
    #######################
    # 输入项：
    # data训练数据
    # label标签
    # P训练集合比例
    # N待分类的被试数目
    # 数据切割段数
    # 与MATLAB对比
    #######################
    ##########################
    # 准备分类器
    ##########################
    clf1 = SVC(kernel='rbf', gamma=1, C=1.15, shrinking=False, max_iter=1e7, class_weight='balanced')
    par1 = {'C': [0.1, 0.5, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 3.0], 'gamma': [0.1, 0.5, 1.0, 1.5, 2.0]}
    clf2 = GaussianNB()
    par2 = {}
    clf3 = AdaBoostClassifier()
    par3 = {'n_estimators': [30, 35, 40, 45, 50, 55, 60], 'learning_rate': [0.1, 0.2, 0.5, 1.0, 1.5, 2.0]}
    # clf4 = KNeighborsClassifier()
    # par4 = {'algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute'), 'n_neighbors': [3, 4, 5]}
    scaler = StandardScaler()
    # clf = [clf1,clf2,clf3]
    # pars = [par1, par2, par3]
    clf = [clf1]
    pars = [par1]
    res = {}
    res.setdefault('l', int(1 / P))
    ##########################
    # 准备训练集合与测试集合
    ##########################
    for i in range(N):
        templ = len(data[i])
        l_seg = int(templ / N_seg)
        l_segte = int(l_seg * P)
        l_segtr = l_seg - l_segte
        tempdata = []
        templabel = []
        res.setdefault(i, {})
        for k in range(N_seg):
            tempdata.append(data[i][k * l_seg:(k + 1) * l_seg])
            templabel.append(label[i][k * l_seg:(k + 1) * l_seg])

        for count in range(len(clf)):
            res[i].setdefault(str(clf[count])[0:3], {})
            for k in range(int(1 / P)):
                ind = np.zeros(l_seg)
                ind[k * l_segte:(k + 1) * l_segte] = 1
                indte = np.where(ind == 1)
                # ind = np.ones(ind.shape[0]) - ind
                indtr = np.where(ind == 0)
                res[i][str(clf[count])[0:3]].setdefault(k, {})

                for count2 in range(int(1 / P)):
                    temptrain = tempdata[count2][indtr]
                    temptest = tempdata[count2][indte]
                    templatr = templabel[count2][indtr]
                    template = templabel[count2][indte]
                    if count2 == 0:
                        dtrain = temptrain
                        dtest = temptest
                        ltrain = templatr
                        ltest = template
                    else:
                        dtrain = np.vstack((dtrain, temptrain))
                        dtest = np.vstack((dtest, temptest))
                        ltrain = np.hstack((ltrain, templatr))
                        ltest = np.hstack((ltest, template))
                    del temptrain, temptest, templatr, template
                #######################
                # 输入项标准化
                #######################
                scaler.fit(dtrain)
                temptr = scaler.transform(dtrain)

                tempte = scaler.transform(dtest)
                ########################
                # 输入不经标准化
                ########################
                # temptr = dtrain
                # tempte = dtest
                #########################
                # 分类器训练与分类
                #########################

                clf[count].fit(temptr, ltrain)
                temppre = clf[count].predict(tempte)
                tempacu = accuracy_score(ltest, temppre)
                temprecall = recall_score(ltest, temppre)
                tempprecision = precision_score(ltest, temppre)

                # temppre = clf[i].predict(tempte)
                res[i][str(clf[count])[0:3]].setdefault(k, {})
                res[i][str(clf[count])[0:3]][k].setdefault('accuracy', tempacu)
                res[i][str(clf[count])[0:3]][k].setdefault('recall', temprecall)
                res[i][str(clf[count])[0:3]][k].setdefault('precision', tempprecision)

                # del temppre
                del ltrain, ltest, dtrain, dtest

    return (res)


def resana911(res, N, index):
    #############################
    # res,超参数网格寻优结果
    # N，原始数据切割段数
    # index，评价指标内容
    # train911与train912的结果都可以使用该函数
    #############################
    l_subject = len(res) - 1
    clfs = res[0].keys()
    l_seg = N
    l_ind = len(index)

    rea = {}
    for i in range(l_subject):
        rea.setdefault(i, {})
        for item in clfs:
            rea[i].setdefault(item, [])

            for ind in index:
                temp = []
                for j in range(l_seg):
                    temp.append(res[i][item][j][ind])
                rea[i][item].append(np.mean(temp))
    return rea


def resana913(res, N, index):
    #############################
    # res,超参数网格寻优结果
    # N，原始数据切割段数
    # index，评价指标内容
    # 超参数寻有结果使用该函数
    #############################
    l_clf = len(res)
    clfs = list(res.keys())
    l_subject = len(res[clfs[0]])
    l_eva = len(res[clfs[0]][0].keys())
    item_eva = list(res[clfs[0]][0].keys())
    # l_par = len(res[clfs[0]][0][item_eva[0]])
    rea = {}
    for item in clfs:
        rea.setdefault(item, {})
        for i in range(l_subject):
            rea[item].setdefault(str(i), {})
            for count in range(len(res[item][i][item_eva[0]])):
                rea[item][str(i)].setdefault(str(count), [])
                for eva in item_eva:
                    rea[item][str(i)][str(count)].append(res[item][i][eva][count])
    return rea


########################
# 太多了，不知道写的什么
########################
def train(data_train, label, P):
    clf1 = SVC(kernel='rbf', gamma=1, C=1)
    # clf2 = KNeighborsClassifier()
    # clf3 = GaussianNB()
    # clf4 = AdaBoostClassifier()
    scaler = StandardScaler()

    N = 23
    score = []
    # clf = [clf1,clf2,clf3,clf4]
    clf = [clf1]
    score = ['accuracy', 'precision_macro', 'recall_macro']
    parameters = {'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3, 45]}
    res = {}
    best_par = {}
    kf = KFold(n_splits=5, shuffle=True)

    for item in clf:
        for i in range(1):
            # temp_data = []
            np.delete(data_train[i], [5, 12], axis=1)
            res.setdefault('recal', [])
            res.setdefault('accu', [])
            best_par.setdefault(i, [])
            ind_train = []
            ind_test = []
            temp_data = scaler.fit_transform(data_train[i])
            # temp_clf = GridSearchCV(item, parameters, cv=5, scoring='recall', n_jobs=4)
            # for train_index,test_index in kf.split(temp_data):
            #     temp_train = temp_data[train_index]
            #     label_train = label[i][train_index]
            #     temp_test = temp_data[test_index]
            #     label_test = label[i][test_index]
            # temp_clf.fit(temp_train,label_train)
            # best_par[i].append(temp_clf.best_params_)
            # res[i].append(temp_clf.best_score_)

            res['recal'].append(cross_val_score(item, temp_data, label[i], scoring='recall', cv=5, n_jobs=4))
            res['accu'].append(cross_val_score(item, temp_data, label[i], scoring='accuracy', cv=5, n_jobs=4))
            # 删除相应的特征列
            # col_ind = [8,9,10,11]
            # temp_data = np.delete(temp_data,col_ind,axis = 1)
            # col_ind = [2,3,4]
            # temp_data = np.delete(temp_data,col_ind,axis=1)

            # 计算相应数据集长度
            # l_temp = len(temp_data)
            # l_train = int(l_temp*P)
            # l_test = l_temp-l_train

            # #切分训练集以及测试集
            # temp_train = temp_data[0:l_train]
            # label_train = label[i][0:l_train]
            # temp_test = temp_data[l_train:]
            # label_test = label[i][l_train:]
            # # ran_num = np.random.random([1,l_train])
            # count_ran = np.argsort(ran_num)
            # temp_train = temp_train[count_ran]
            # label_train = label_train[count_ran]
            #
            #
            # temp_test = temp_data[l_train:]
            # label_test = label[i][l_train:]
            # ran_num = np.random.random([1, l_test])
            # count_ran = np.argsort(ran_num)
            # temp_test = temp_test[count_ran]
            # label_test = label_test[count_ran]
            #
            # #拟合模型并且进行预测
            # item.fit(temp_train[0],label_train[0])
            # label_pre = item.predict(temp_test[0])
            # label_pre = np.mat(label_pre)
            # label_test = np.mat(label_test[0])
            # temp = np.zeros([1,l_test])
            # temp[label_pre == label_test]=1

            ##以下是将原始数据集打乱
            # temp_count = np.random.random([1,l_temp])
            # num = np.argsort(temp_count)
            # data_ran = temp_data[num]
            # label[i] = label[i][num]
            # temp_train = temp_data[num[0,1:l_train]]
            # label_train = label[i][num[0,1:l_train]]
            # temp_test = temp_data[num[0,l_train:]]
            # label_test = label[i][num[0,l_train:]]
            # item.fit(temp_train,label_train)
            # label_pre = item.predict(temp_test)
            # label_pre = np.mat(label_pre)
            # label_test = np.mat(label_test)
            # temp = np.zeros([1,l_test])
            # temp[label_pre == label_test]=1

            # cross_val_score(temp_clf,temp_data,label[i],scoring="accuracy",cv = 5,n_jobs=4)

            # temp = cross_validate(best_clf,temp_data,label[i],scoring=score,cv = 5,n_jobs=4)

            # score.append([sum(temp)/len(temp),str(item)[0:3]+str(i)])

    return res, best_par


########################
# 将结果存储为txt文件，没什么用
########################
def write_res(score):
    file = open("res.txt", 'a')
    for item in score:
        for value in item:
            file.write(str(value))
            file.write("\t")
        file.write('\n')
    file.close()


########################
# 读取结果文件，没什么用
########################
def load_res(filename="res.txt"):
    file = open(filename, 'r')
    data = []
    N = 23
    i = 1
    temp = []
    for line in file.readlines():

        line = line.split('\t')
        del (line[-1])
        if i < 23:
            temp.append(float(line[0]))
            i += 1
        else:
            temp.append(float(line[0]))
            data.append(temp)
            temp = []
            i = 1

    return data


########################
# 各训练集合中阳性阴性数目统计柱状图显示
########################
def data_analysis(data, label, P):
    l = len(data)
    l_p = int(l * P)
    num = int(1 / P)
    temp = []
    temp_label = []
    count_p = []
    count_n = []
    for i in range(num):
        temp.append(data[i * l_p:(i + 1) * l_p])
        temp_label.append(label[i * l_p:(i + 1) * l_p])
        count_p.append(sum(temp_label[i]))
        count_n.append(temp_label[i].shape[0] - sum(temp_label[i]))
    xlabel = []
    for i in range(num):
        xlabel.append(i)
    fig = plt.figure(1)
    sns.barplot(x=xlabel, y=count_p)
    plt.title("count_p")
    fig = plt.figure(2)
    sns.barplot(x=xlabel, y=count_n)
    plt.title("count_n")
    plt.show()


def data_split(data, label, P):
    #################################
    # 返回DataFrame类型数据集
    #################################
    l = len(data)
    res = []
    for i in range(l):
        temp = pd.DataFrame(data[i])
        temp['dataset'] = 'Test'
        temp.iloc[0:int(len(data[i]) * P), -1] = 'Train'
        temp['label'] = pd.Series(label[i])
        res.append(temp)
    return res


########################
# 打乱数据集？
########################
def disorganize(data, label):
    l = len(data)
    res_data = []
    res_label = []
    for i in range(l):
        siz = len(data[i])
        ran = np.random.random(siz)
        ran_ind = np.argsort(ran)
        temp_data = data[i][ran_ind]
        temp_label = label[i][ran_ind]
        res_data.append(temp_data)
        res_label.append(temp_label)
    return res_data, res_label


########################
# 获得数据集
########################
def get_dataset(data, label, P):
    ##########################
    # 返回numpy.array类型数据集，不便于进行画图分析
    ##########################
    l = len(data)
    train_set = []
    test_set = []
    train_label = []
    test_label = []
    for i in range(l):
        temp_train = data[i][0:int(len(data[i]) * P)]
        temp_test = data[i][int(len(data[i]) * P):]
        temp_label_train = label[i][0:int(len(data[i]) * P)]
        temp_label_test = label[i][int(len(data[i]) * P):]
        train_set.append(temp_train)
        train_label.append(temp_label_train)
        test_set.append(temp_test)
        test_label.append(temp_label_test)
    return train_set, train_label, test_set, test_label


########################
# 训练之前首先归一化处理的训练方法
########################
def maxmintrain(data_train, label_train, data_test, label_test):
    l = len(data_train)
    scaler = []
    score = []
    for i in range(l):
        scaler.append(MinMaxScaler().fit(data_train[i]))
        data_train[i] = scaler[i].transform(data_train[i])
        data_test[i] = scaler[i].transform(data_test[i])
        clf = SVC()
        clf.fit(data_train[i], label_train[i])
        score.append(clf.score(data_test[i], label_test[i]))
    return score


########################
# 下采样，应该没什么用
########################
def downsample(data, label, fea_ind):
    res = []
    res_label = []
    l = len(data)
    for fea in fea_ind:
        for i in range(l):
            Q1 = np.percentile(data[i][:, fea], 25)
            Q3 = np.percentile(data[i][:, fea], 75)
            IQR = Q3 - Q1
            outlier_step = 1.5 * IQR
            ind_row = np.where((data[i][:, fea] < (Q1 - outlier_step)) | (data[i][:, fea] > (Q3 + outlier_step)))
            temp = np.delete(data[i], ind_row, axis=0)
            temp_label = np.delete(label[i], ind_row, axis=0)
            res.append(temp)
            res_label.append(temp_label)
    return res, res_label


############################
# 特征分析箱线图，输入读取出的特征data以及label
############################
def fea_ana(data, label):
    num_f = len(data[0][0, :])
    name = ['feature1', 'feature2', 'feature3']
    for i in range(num_f):
        fig = plt.figure(i)
        sns.boxenplot(x=label[0], y=data[0][:, i])
        plt.title(name[i])
        fig.savefig("feature" + str(i) + ".jpg")
    plt.show()


############################
# 样本散点图分布
############################
def sample_dis(data, label, N):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    tempinda = np.where(label[N] == 1)
    tempindn = np.where(label[N] == 0)
    tempa = data[N][tempinda]
    tempn = data[N][tempindn]

    tempa = ax.scatter(tempa[:, 0], tempa[:, 1], tempa[:, 2], c='r')
    tempn = ax.scatter(tempn[:, 0], tempn[:, 1], tempn[:, 2], c='b')
    ax.set_zlabel('feature3')
    ax.set_ylabel('feature2')
    ax.set_xlabel('feature1')
    ax.legend([tempa, tempn], ["Apnea", "Normal"])
    fig.savefig("sample_distribution.jpg")
    plt.show()
