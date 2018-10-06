import function
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler,Normalizer
import pandas as pd

if __name__ == '__main__':
    ################################
    #超参数评优
    ################################
    data, label = function.load_data('f60-1-notao.xlsx', 1)
    data,label = function.createdataset(data,label,0.5,1,5)
    res,pars = function.parop(data,label,3,1)
    rea = function.resana913(res,5,['accuracy','recall','precision'])
    print(rea)
    ################################
    #926结果再现
    ################################
    # data,label = function.load_data('f60-1-notao.xlsx',1)
    # data2,label2 = function.load_data('f20-1-7.xlsx',1)
    # datatrain1, labeltrain1, datatest1, labeltest1, datatrain2, labeltrain2, datatest2, labeltest2 = function.dataseg(
    #     data[0], label[0], data2[0], label2[0], 60, 20, 0.5)
    # clfs,num1,num2 = function.clfcastrain(["Dec","Dec"],datatrain1,labeltrain1,datatrain2,labeltrain2,0,0)
    # tempres, = function.clfcastest(clfs,datatest1,labeltest1,datatest2,labeltest2,0,0)
    # print(tempres)
    #######################
#926
#########################
    # # 级联分类器K折交叉训练
    N = int(input('Please input the number of the subjects you want to train and test:'))
    data,label = function.load_data('f60-7.xlsx', N)
    data2,label2 = function.load_data('f20-7.xlsx', N)
    data, label = function.createdataset(data, label, 0.5, N, 5)
    data2, label2 = function.createdataset(data2, label2, 0.5, N, 5)
    # data,label = function.load_data('f60-1-notao.xlsx', N)
    # data2,label2 = function.load_data('f20-1-7.xlsx', N)
    res,resave = function.clfcaskfold(data,label,data2,label2,N)
    print(res)


    # 单分类器训练
    # datatrain,labeltrain,datatest,labeltest = function.get_dataset(data2,label2,0.7)
    # clfs = function.AHItrain(["Dec"],datatrain[0],labeltrain[0])
    # function.AHItest(clfs,datatest[0],labeltest[0])

    #######################
    #9-15OB tree
    #######################
    # data,label = function.load_data('f60-1-3.xlsx',1)
    # data,label = function.createdataset(data,label,0.2,1,5)
    #######################
    #9-22测试
    #######################

    # function.clfcas(["Dec", "Dec"], data[0], label[0], data2[0], label2[0], 0, 1)
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

