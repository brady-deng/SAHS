import pandas as pd
from sklearn.svm import SVC
from sklearn. ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder,StandardScaler,MinMaxScaler
from sklearn.model_selection import cross_val_score,cross_validate,GridSearchCV,KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



def load_data(filename):
    N = 23
    data = pd.read_excel(filename,sheet_name =None)
    data_train = []
    label = []
    for i in range(N):
        label.append(data[str(i + 1)].iloc[:, -1].values)
        data_train.append(data[str(i+1)].iloc[:,0:-1].values)


    return data_train,label
def train_97(data,label,P,N):
    # lsub = len(data)
    clf1 = SVC()
    clf2 = GaussianNB()
    clf3 = KNeighborsClassifier()
    clf4 = AdaBoostClassifier()
    clf = [clf1,clf2,clf3,clf4]
    res = []
    for i in range(N):
        templ = len(data[i])
        templtr = int(templ*P)
        templte = templ - templtr
        temptr = data[i][0:templtr]
        tempte = data[i][templtr:]
        latr = label[i][0:templtr]
        late = label[i][templtr:]
        tempbin = []
        for item in clf:

            tempres = {}
            tempres.setdefault(str(item)[0:3],[])
            item.fit(temptr,latr)
            temppre = item.predict(tempte)
            tempres[str(item)[0:3]].append(accuracy_score(late,temppre))
            tempres[str(item)[0:3]].append(precision_score(late, temppre))
            tempres[str(item)[0:3]].append(recall_score(late,temppre))
            tempbin.append(tempres)


            del temppre,tempres
        res.append(tempbin)
        del templ,templtr,templte,temptr,tempte,latr,late

    return res
def train(data_train,label,P):
    clf1 = SVC(kernel='rbf',gamma=1,C=1)
    # clf2 = KNeighborsClassifier()
    # clf3 = GaussianNB()
    # clf4 = AdaBoostClassifier()
    scaler = StandardScaler()

    N = 23
    score = []
    # clf = [clf1,clf2,clf3,clf4]
    clf = [clf1]
    score = ['accuracy','precision_macro','recall_macro']
    parameters = {'C':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.5,2.0,2.5,3,45]}
    res = {}
    best_par = {}
    kf = KFold(n_splits=5,shuffle=True)

    for item in clf:
        for i in range(1):
            # temp_data = []
            np.delete(data_train[i],[5,12],axis=1)
            res.setdefault('recal',[])
            res.setdefault('accu',[])
            best_par.setdefault(i,[])
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
            #删除相应的特征列
            # col_ind = [8,9,10,11]
            # temp_data = np.delete(temp_data,col_ind,axis = 1)
            # col_ind = [2,3,4]
            # temp_data = np.delete(temp_data,col_ind,axis=1)

            #计算相应数据集长度
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


    return res,best_par
def write_res(score):
    file = open("res.txt", 'a')
    for item in score:
        for value in item:
            file.write(str(value))
            file.write("\t")
        file.write('\n')
    file.close()
def load_res(filename="res.txt"):
    file = open(filename,'r')
    data = []
    N = 23
    i = 1
    temp = []
    for line in file.readlines():

        line = line.split('\t')
        del(line[-1])
        if i < 23:
            temp.append(float(line[0]))
            i+=1
        else:
            temp.append(float(line[0]))
            data.append(temp)
            temp = []
            i = 1


    return data
def data_analysis(data,label,P):
    l = len(data)
    l_p = int(l*P)
    num = int(1/P)
    temp = []
    temp_label = []
    count_p = []
    count_n = []
    for i in range(num):
        temp.append(data[i*l_p:(i+1)*l_p])
        temp_label.append(label[i*l_p:(i+1)*l_p])
        count_p.append(sum(temp_label[i]))
        count_n.append(temp_label[i].shape[0]-sum(temp_label[i]))
    xlabel = []
    for i in range(num):
        xlabel.append(i)
    fig = plt.figure(1)
    sns.barplot(x = xlabel,y = count_p)
    plt.title("count_p")
    fig = plt.figure(2)
    sns.barplot(x = xlabel,y = count_n)
    plt.title("count_n")
    plt.show()
def data_split(data,label,P):
    #################################
    #返回DataFrame类型数据集
    #################################
    l = len(data)
    res = []
    for i in range(l):
        temp = pd.DataFrame(data[i])
        temp['dataset'] = 'Test'
        temp.iloc[0:int(len(data[i])*P),-1] = 'Train'
        temp['label'] = pd.Series(label[i])
        res.append(temp)
    return res
def disorganize(data,label):
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
    return res_data,res_label
def get_dataset(data,label,P):
    ##########################
    #返回numpy.array类型数据集，不便于进行画图分析
    ##########################
    l = len(data)
    train_set = []
    test_set = []
    train_label = []
    test_label = []
    for i in range(l):
        temp_train = data[i][0:int(len(data[i])*P)]
        temp_test = data[i][int(len(data[i])*P):]
        temp_label_train = label[i][0:int(len(data[i])*P)]
        temp_label_test = label[i][int(len(data[i])*P):]
        train_set.append(temp_train)
        train_label.append(temp_label_train)
        test_set.append(temp_test)
        test_label.append(temp_label_test)
    return train_set,train_label,test_set,test_label
def maxmintrain(data_train,label_train,data_test,label_test):
    l = len(data_train)
    scaler = []
    score = []
    for i in range(l):
        scaler.append(MinMaxScaler().fit(data_train[i]))
        data_train[i] = scaler[i].transform(data_train[i])
        data_test[i] = scaler[i].transform(data_test[i])
        clf = SVC()
        clf.fit(data_train[i],label_train[i])
        score.append(clf.score(data_test[i],label_test[i]))
    return score
def downsample(data,label,fea_ind):
    res = []
    res_label = []
    l = len(data)
    for fea in fea_ind:
        for i in range(l):
            Q1=np.percentile(data[i][:,fea],25)
            Q3=np.percentile(data[i][:,fea],75)
            IQR=Q3-Q1
            outlier_step = 1.5*IQR
            ind_row = np.where((data[i][:,fea]<(Q1-outlier_step))|(data[i][:,fea]>(Q3+outlier_step)))
            temp = np.delete(data[i],ind_row,axis = 0)
            temp_label = np.delete(label[i],ind_row,axis = 0)
            res.append(temp)
            res_label.append(temp_label)
    return res,res_label
