from sklearn.model_selection import train_test_split,KFold,GroupKFold,GroupShuffleSplit,ShuffleSplit
import numpy as np
a = np.ones([5,20])
b = np.zeros([1,20])
b[0,10:] = 1
a = a.T
b = b.T

####################
#train_test_aplit
####################
x_train,x_test,y_train,y_test = train_test_split(a,b,test_size=0.2,shuffle=False)
print("y_train")
print(y_train)
print("y_test")
print(y_test)
####################
#KFold
####################
kf = KFold(n_splits=5)
for train_index,test_index in kf.split(a):
    print("train_index:",train_index,",test_index:",test_index)
####################
#Groupkfold
####################
kf2 = GroupKFold(n_splits=5)
res2 = kf2.get_n_splits(a,b)
for train_index,test_index in kf2.split(a,b):
    print("train_index:", train_index, ",test_index:", test_index)