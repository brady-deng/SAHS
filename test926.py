"""
__title__ = ''
__author__ = 'WNI10'
__mtime__ = '2018/9/27'
"""
import func
if __name__ == '__main__':
    #######################
#926
#########################
    # 级联分类器训练
    N = int(input('Please input the number of the subjects you want to train and test:'))
    data,label = func.load_data('f60-1-notao.xlsx',N)
    data2,label2 = func.load_data('f20-1-7.xlsx',N)
    res = []
    for i in range(N):
        datatrain1, labeltrain1, datatest1, labeltest1, datatrain2, labeltrain2, datatest2, labeltest2 = func.dataseg(data[0],label[0],data2[0],label2[0],60,20,0.7)
        clfs = func.clfcastrain(["Dec","Dec"],datatrain1,labeltrain1,datatrain2,labeltrain2,0,0)
        tempres = func.clfcastest(clfs,datatest1,labeltest1,datatest2,labeltest2,0,0)
        res.append(tempres)
    print(res)

    # 单分类器训练
    # datatrain,labeltrain,datatest,labeltest = func.get_dataset(data2,label2,0.7)
    # clfs = func.AHItrain(["Dec"],datatrain[0],labeltrain[0])
    # func.AHItest(clfs,datatest[0],labeltest[0])

    #######################
    #9-15OB tree
    #######################
    # data,label = func.load_data('f60-1-3.xlsx',1)
    # data,label = func.createdataset(data,label,0.2,1,5)
    #######################
    #9-22测试
    #######################
    # data,label = func.load_data('f60-1-notao.xlsx',1)
    # data2,label2 = func.load_data('f20-1-flow.xlsx',1)
    # func.clfcas(["Dec", "Dec"], data[0], label[0], data2[0], label2[0], 0, 1)
    # data3, label3 = func.load_data('f20-1-sp.xlsx', 1)
    # func.clfcas(["Dec", "Dec"], data[0], label[0], data3[0], label3[0], 0, 1)
    # res,pars = func.parop(data,label,3,1)
    # rea = func.resana913(res,5,['accuracy','recall','precision'])
    # res2,pars2 = func.parop(data2,label2,3,1)
    # rea2 = func.resana913(res2,5,['accuracy','recall','precision'])
    # print(rea)

    # res,flag = func.AHIres(["Dec"],data[0],label[0])

    #######################
    #9-17AHI计算
    #######################
    # data,label = func.load_data('f20-1.xlsx',1)
    # data,label = func.createdataset(data,label,0.5,1,4)
    # data,label = func.createahiset(data,label,0.5,1)
    # res,respre,ind = func.AHIval(["Dec"],data,label)
    # ah,cache = func.AHIcal(label[0])
    #######################
    # 9-13划分数据集以及超参数调优
    #######################
    # data,label = func.load_data('f20-1-923.xlsx',1)
    # data,label = func.createdataset(data,label,0.2,1,5)
    # res,pars = func.parop(data,label,3,1)
    # rea = func.resana913(res,5,['accuracy','recall','precision'])
    # print(rea)
    # print(pars)
    # #######################
    #9-12classify
    #######################
    ##############################
    # 超参数调优
    ##############################
    # data,label = func.load_data('f60-1.xlsx',1)
    # res = func.train_911(data,label,0.2,1,5)
    # print(res)
    # rea = func.resana911(res,['best_recall','best_precision'])
    # print(rea)
    ##############################
    #912与MATLAB进行对照
    ##############################
    # data,label = func.load_data('f60-1.xlsx',1)
    # res = func.train_912(data,label,0.2,1,5)
    # ra = func.resana911(res,['accuracy','recall','precision'])
    # print(ra)
    ######################
    #9-7classifiy
    ######################
    # data,label = func.load_data('f60-1.xlsx',1)
    # # func.fea_ana(data,label)
    # # func.sample_dis(data,label,1)
    # res = func.train_911(data,label,0.2,1,5)
    # print(res)
    # rea = func.resana911(res)
    ######################
    #classifiy
    ######################
    # data,label = func.load_data("f30-1-norm.xlsx")
    # data, label = func.downsample(data, label, [0, 1, 5, 7])
    # score,par = func.train(data,label,0.8)
    # print(score,par)
    # func.write_res(score = score)
    # data = func.load_res("res.txt")
    # l = len(data)
    # for i in range(l):
    #     print(sum(data[i])/len(data[i]))
    #######################
    #maxminclassify
    #######################
    # data_train,label_train,data_test,label_test = func.get_dataset(data,label,0.8)
    # score = func.maxmintrain(data_train,label_train,data_test,label_test)
    # print(score)
    #######################
    #analyze
    #######################
    # data,label = func.load_data("f60-1.xlsx")
    # func.data_analysis(data[0],label[0],0.2)
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
    # data_train,label_train,data_test,label_test = func.data_split(data,label,0.8)
    # data_outor,label_outor = func.disorganize(data,label)
    # data_ana_outor = func.data_split(data_outor,label_outor,0.8)
    # data_ana_or = func.data_split(data,label,0.8)
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
    # data_train,label_train,data_test,label_test = func.get_dataset(data,label,0.8)
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

