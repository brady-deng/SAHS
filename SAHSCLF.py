import function
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler,Normalizer
import pandas as pd
import json
import datetime

if __name__ == '__main__':
    ################################
    #超参数评优
    ################################
    # classweight = input('Please input the classweight of the decision tree(** **):')
    # classweight = [int(item) for item in classweight.split()]
    # N = int(input('Please input the index of the subject you want to train:'))
    # if classweight[0] != 0:
    #     wei = {0:classweight[0],1:classweight[1]}
    # else:
    #     wei = 'balanced'
    # data, label = function.load_data('f20-12-ob.xlsx', N)
    # # data,label = function.createdataset(data,label,0.5,1,5)
    # res,pars = function.parop(data,label,3,1,classweight=wei)
    # rea = function.resana913(res,5,['accuracy','recall','precision'])
    # print(rea)
    ###############################
    # 926结果再现
    # ###############################
    # classweight = input('Please input the classweight of the decision tree(** ** ** **):')
    # classweight = [int(item) for item in classweight.split()]
    # if classweight[2] == 0:
    #     wei = [{0:classweight[0],1:classweight[1]},'balanced']
    # else:
    #     wei = [{0: classweight[0], 1: classweight[1]}, {0:classweight[2],1:classweight[3]}]
    # data,label = function.load_data('f60-1-notao.xlsx',1)
    # data2,label2 = function.load_data('f20-1-7.xlsx',1)
    # datatrain1, labeltrain1, datatest1, labeltest1, datatrain2, labeltrain2, datatest2, labeltest2 = function.dataseg(
    #     data[0], label[0], data2[0], label2[0], 60, 20, 0.5)
    # clfs,num1,num2 = function.clfcastrain(["Dec","Dec"],datatrain1,labeltrain1,datatrain2,labeltrain2,0,0,classweight = wei)
    # tempres = function.clfcastest(clfs,datatest1,labeltest1,datatest2,labeltest2,0,0)
    # print(tempres[0])
    # print(tempres[1])
    # print(tempres[2])
    #######################
#926
#########################
    # 级联分类器K折交叉训练
    N = int(input('Please input the index of the subjects you want to train and test:'))
    WT1 = int(input('Please input the length of window1:'))
    WT2 = int(input('Please input the length of window2:'))
    classweight = input('Please input the classweight of the decision tree(** ** ** **):')
    classweight = [int(item) for item in classweight.split()]
    wei = []
    if classweight[0] == 0:
        wei.append('balanced')
    elif classweight[0] == 1028:
        wei.append(1028)
    else:
        wei.append({0:classweight[0],1:classweight[1]})
    if classweight[2] == 0:
        wei.append('balanced')
    elif classweight[2] == 1028:
        wei.append(1028)
    else:
        wei.append({0:classweight[2],1:classweight[3]})
    data,label,timeind = function.load_data('f60-23-1119.xlsx', N)
    data2,label2,timeind2 = function.load_data('f10-23-1119.xlsx', N)

    N = len(data)
    # # data, label, timeind = function.createdataset(data, label, timeind, 0.5, N, 5)
    # # # data2, label2 = function.createdataset(data2, label2, 0.5, N, 2)
    # indall = [i for i in range(23)]
    # indall = set(indall)
    # # indfew = [4,5,6,8,10,12,14,15,16]
    # indfew = [12]
    # # indfew = []
    # indfew = set(indfew)
    # indroi = indall - indfew
    # indroi = list(indroi)
    # # # indroi = [0,1,2,7,9,11,13,14,21]
    # ind = np.array(indroi)
    # ind = np.array([0,1,2,3,6,7,9,13,17,19,20,21,22])
    # data = data[ind]
    # label = label[ind]
    # timeind = timeind[ind]
    # data2 = data2[ind]
    # label2 = label2[ind]
    # timeind2 = timeind2[ind]
    indall = [i for i in range(23)]
    indall = set(indall)
    # indfew = [4,5,6,8,10,12,14,15,16]
    indfew = [12]
    # indfew = []
    indfew = set(indfew)
    indroi = indall - indfew
    indroi = list(indroi)
    # # indroi = [0,1,2,7,9,11,13,14,21]
    ind = np.array(indroi)
    ind = np.array([0,1,2,3,4,6,7,9,13,15,17,19,20,21,22])
    data = data[ind]
    label = label[ind]
    timeind = timeind[ind]
    data2 = data2[ind]
    label2 = label2[ind]
    timeind2 = timeind2[ind]
    N = len(data)
    # res = function.clfcasopt(data, label, data2, label2, timeind, timeind2, N, WT1, WT2, classweight=wei, Y=1)

    res,resave,par = function.clfcaskfold(data,label,data2,label2,timeind,timeind2,N,WT1,WT2,classweight = wei,Y = 0)
    resse = function.seveva(resave[:-1,-2],resave[:-1,-1])
    # res, resave = function.clfloadtest(data, label, data2, label2, timeind, timeind2, N, WT1, WT2,)
    keys = []
    for i in range(N):
        keys.append('sub'+str(i))
    keys.append('ave')
    columns = ['nump','numn','acu1','recall1','pre1','acu2','recall2','pre2','acu3','recall3','pre3',\
               'pre_e','recall_e','wrong_e','miss_e','total_e','p_test','n_test','true_pre','true_p','ahi_psg',\
               'ahi_pre']
    resave = pd.DataFrame(resave,columns=columns)
    columns = ['ind_minsplit1', 'ind_minleaf1', 'ind_maxdepth1', 'ind_minsplit2', 'ind_minleaf2', 'ind_maxdepth2']
    del keys[-1]
    par = pd.DataFrame(par,columns=columns)
    resexcel = pd.concat([resave,par],axis = 1)
    keys = ['normal','slight','normal','serious','total']
    # resse2 = np.zeros((6,5))
    # resse2[1:,:] = resse
    # resse2[0,:] = np.arange(1,6)
    resse = pd.DataFrame(resse,index = keys,columns=keys)
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    var1 = nowTime[5:7]
    var2 = nowTime[8:10]
    var3 = nowTime[11:13]
    var4 = nowTime[14:16]
    time = var1+var2+var3+var4
    writer = pd.ExcelWriter('resob'+time+'.xlsx')
    resexcel.to_excel(writer,'Sheet1')
    resse.to_excel(writer,'Sheet2')
    writer.save()
    # print(res)

    # data, label = function.createdataset(data, label, 0.5, N, 2)
    # data2, label2 = function.createdataset(data2, label2, 0.5, N, 2)

    # data,label = function.load_data('f60-1-notao.xlsx', N)
    # data2,label2 = function.load_data('f20-1-7.xlsx', N)


    # 单分类器训练与事件检测
    # N = int(input('Please input the index of the subjects you want to train:'))
    # data,label,timeind = function.load_data('f10-23-1119.xlsx', N)
    # # ind = np.array([ 0,1,2,7,9,11,13,14,21])
    # # data = data[ind]
    # # label = label[ind]
    # # data2 = data2[ind]
    # # label2 = label2[ind]
    # indall = [i for i in range(23)]
    # indall = set(indall)
    # indfew = [12]
    # # indfew = []
    # indfew = set(indfew)
    # indroi = indall - indfew
    # indroi = list(indroi)
    # ind = np.array(indroi)
    # data = data[ind]
    # label = label[ind]
    # timeind = timeind[ind]
    # N = len(data)
    # ind = input('Please input the par of the Decision tree(** ** **):').split()  # 决策树的叶子节点的参数设置，最小样本切割，最小叶子样本，最大深度
    # ind = [int(item) for item in ind]
    # indpar = ind
    # WT = int(input('Please input the window length:'))
    # classweight = input('Please input the classweight of the decision tree(** **):')  # 决策树的类权重
    # classweight = [int(item) for item in classweight.split()]
    # res0 = function.clfkfold("Ran", data, label, timeind,N,indpar,classweight,WT)
    # print(res0)
    # function.clfopt(data,label,timeind,N)


    # 单分类器事件检测
    # N = int(input('Please input the index of the subjects you want to train:'))
    # data,label = function.load_data('f20-12-ob.xlsx', N)
    # N = len(data)
    # data2,label2 = function.load_data('f20-1-1008.xlsx', N)
    # data3, label3 = function.load_data('f60-23-1008.xlsx', N)
    # data4, label4 = function.load_data('f20-23-1008.xlsx', N)
    # data, label = function.createdataset(data, label, 0.5, N, 2)
    # data2, label2 = function.createdataset(data2, label2, 0.5, N, 2)
    # ind = np.array([ 0,  1,  2,  6,  7,  9, 13, 17, 18, 19, 20, 21, 22])
    # data = data[ind]
    # label = label[ind]
    # data2 = data2[ind]
    # label2 = label2[ind]

    # res1 = function.clfkfold("Ran", data2, label2, N)
    # res2 = function.clfkfold("Ran", data3, label3, N)
    # res3 = function.clfkfold("Ran", data4, label4, N)

    # # print(res1[1])
    # # print(res2[2])
    # # print(res3[3])

    #######################
    #9-15OB tree
    #######################
    # data,label = function.load_data('f60-1-3.xlsx',1)
    # data,label = function.createdataset(data,label,0.2,1,5)
    #######################
    #9-22测试
    #######################

    # # function.clfcas(["Dec", "Dec"], data[0], label[0], data2[0], label2[0], 0, 1)
    # data, label = function.load_data('f60-1-notao.xlsx', 1)
    # data,label = function.createdataset(data,label,0.5,1,5)
    # # # function.clfcas(["Dec", "Dec"], data[0], label[0], data3[0], label3[0], 0, 1)
    # res,pars = function.parop(data,label,3,1)
    # rea = function.resana913(res,5,['accuracy','recall','precision'])
    # # # res2,pars2 = function.parop(data2,label2,3,1)
    # # # rea2 = function.resana913(res2,5,['accuracy','recall','precision'])
    # print(rea)

    # res,flag = function.AHIres(["Dec"],data[0],label[0])

    #######################
    #9-17AHI计算
    #######################
    # data,label = function.load_data('f20-1.xlsx',1)
    # data,label = function.createdataset(data,label,0.5,1,4)
    # data,label = function.createahiset(data,label,0.5,1)
    # res,respre,ind = function.AHIval(["Dec"],data,label)
    # ah,cache = function.AHIcal(label[0])
    #######################
    # 9-13划分数据集以及超参数调优
    #######################
    # data,label = function.load_data('f20-1-923.xlsx',1)
    # data,label = function.createdataset(data,label,0.2,1,5)
    # res,pars = function.parop(data,label,3,1)
    # rea = function.resana913(res,5,['accuracy','recall','precision'])
    # print(rea)
    # print(pars)
    # #######################
    #9-12classify
    #######################
    ##############################
    # 超参数调优
    ##############################
    # data,label = function.load_data('f60-1.xlsx',1)
    # res = function.train_911(data,label,0.2,1,5)
    # print(res)
    # rea = function.resana911(res,['best_recall','best_precision'])
    # print(rea)
    ##############################
    #912与MATLAB进行对照
    ##############################
    # data,label = function.load_data('f60-1.xlsx',1)
    # res = function.train_912(data,label,0.2,1,5)
    # ra = function.resana911(res,['accuracy','recall','precision'])
    # print(ra)
    ######################
    #9-7classifiy
    ######################
    # data,label = function.load_data('f60-1.xlsx',1)
    # # function.fea_ana(data,label)
    # # function.sample_dis(data,label,1)
    # res = function.train_911(data,label,0.2,1,5)
    # print(res)
    # rea = function.resana911(res)
    ######################
    #classifiy
    ######################
    # data,label = function.load_data("f30-1-norm.xlsx")
    # data, label = function.downsample(data, label, [0, 1, 5, 7])
    # score,par = function.train(data,label,0.8)
    # print(score,par)
    # function.write_res(score = score)
    # data = function.load_res("res.txt")
    # l = len(data)
    # for i in range(l):
    #     print(sum(data[i])/len(data[i]))
    #######################
    #maxminclassify
    #######################
    # data_train,label_train,data_test,label_test = function.get_dataset(data,label,0.8)
    # score = function.maxmintrain(data_train,label_train,data_test,label_test)
    # print(score)
    #######################
    #analyze
    #######################
    # data,label = function.load_data("f60-1.xlsx")
    # function.data_analysis(data[0],label[0],0.2)
    # num_f = len(data[0][0,:])
    # name = ['feature1','feature2','feature3']
    # for i in range(num_f):
    #     fig = plt.figure(i)
    #     sns.boxenplot(x = label[0],y = data[0][:,i])
    #     plt.title(name[i])
    #     fig.savefig("feature"+str(i)+".jpg")
    # plt.show()
    #######################
    #Train_set and Test_set analysis
    #######################
    # data_train,label_train,data_test,label_test = function.data_split(data,label,0.8)
    # data_outor,label_outor = function.disorganize(data,label)
    # data_ana_outor = function.data_split(data_outor,label_outor,0.8)
    # data_ana_or = function.data_split(data,label,0.8)
    # num_feature = 12
    # fig = plt.figure()
    # plt.subplot(121)
    # sns.boxplot(x = 'label', y = 0,data = data_ana_or[1])
    # plt.subplot(122)
    # sns.boxplot(x = 'label', y = 0,data = data_ana_outor[1])

    # for i in range(num_feature):
    #     fig = plt.figure(i)
    #     sns.boxplot(x = 'label',y = i,hue='dataset',data=data_ana_or[1])
    #     plt.savefig("contrast"+str(i)+"inorder"+".jpg")
    #     fig = plt.figure(i+12)
    #     sns.boxplot(x='label', y=i, hue='dataset', data=data_ana_outor[1])
    #     plt.savefig("contrast" + str(i) + "outorder" + ".jpg")
    # plt.show()
    ##########################
    #analyze the way of preprocess
    ##########################
    # data_train,label_train,data_test,label_test = function.get_dataset(data,label,0.8)
    # examda_train,examla_train,examda_test,examla_test = data_train[1],label_train[1],data_test[1],label_test[1]
    # scale_minmax = MinMaxScaler().fit(examda_train)
    # temp_minmax_train = scale_minmax.transform(examda_train)
    # temp_minmax_test = scale_minmax.transform(examda_test)
    # temp_minmax_train = pd.DataFrame(temp_minmax_train)
    # temp_minmax_train['dataset'] = 'Train'
    # temp_minmax_train['label'] = pd.Series(examla_train)
    # temp_minmax_test = pd.DataFrame(temp_minmax_test)
    # temp_minmax_test['dataset'] = 'Test'
    # temp_minmax_test['label'] = pd.Series(examla_test)
    # temp_minmax = pd.concat([temp_minmax_train,temp_minmax_test])
    #
    #
    # scale_stand = StandardScaler().fit(examda_train)
    # temp_stand_train = scale_stand.transform(examda_train)
    # temp_stand_test = scale_stand.transform(examda_test)
    # temp_stand_train = pd.DataFrame(temp_stand_train)
    # temp_stand_train['dataset'] = 'Train'
    # temp_stand_train['label'] = pd.Series(examla_train)
    # temp_stand_test = pd.DataFrame(temp_stand_test)
    # temp_stand_test['dataset'] = 'Test'
    # temp_stand_test['label'] = pd.Series(examla_test)
    # temp_stand = pd.concat([temp_stand_train, temp_stand_test])
    #
    #
    #
    # scale_norm = Normalizer().fit(examda_train)
    # temp_norm_train = scale_norm.transform(examda_train)
    # temp_norm_test = scale_norm.transform(examda_test)
    # temp_norm_train = pd.DataFrame(temp_norm_train)
    # temp_norm_train['dataset'] = 'Train'
    # temp_norm_train['label'] = pd.Series(examla_train)
    # temp_norm_test = pd.DataFrame(temp_norm_test)
    # temp_norm_test['dataset'] = 'Test'
    # temp_norm_test['label'] = pd.Series(examla_test)
    # temp_norm = pd.concat([temp_norm_train, temp_norm_test])
    #
    # fig = plt.figure()
    # sns.boxplot(x = 'label',y = 1,hue = 'dataset',data=temp_minmax)
    # plt.savefig("minmax.jpg")
    #
    # fig = plt.figure()
    # sns.boxplot(x='label', y=1,hue = 'dataset', data=temp_stand)
    # plt.savefig("stand.jpg")
    #
    # fig = plt.figure()
    # sns.boxplot(x='label', y=1,hue = 'dataset', data=temp_norm)
    # plt.savefig("norm.jpg")
    ########################
    #downsample
    ########################

