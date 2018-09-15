"""
__title__ = ''
__author__ = 'WNI10'
__mtime__ = '2018/9/15'
"""
from sklearn.datasets import load_iris
import function
# from sklearn import tree

if __name__ == '__main__':
    # iris = load_iris()
    # clf = tree.DecisionTreeClassifier()
    # clf = clf.fit(iris.data,iris.target)
    #
    # import graphviz
    # dot_data = tree.export_graphviz(clf,out_file=None)
    # graph = graphviz.Source(dot_data)
    # graph.render("iris")

    # data = load_iris().data
    # label = load_iris().target
    # function.OB(data,label)
    ###################################
    #画出树形图
    ###################################
    data,label = function.load_data("f60-1-3.xlsx",1)
    data,label = function.createdataset(data,label,0.2,1,5)
    function.OB(data[0],label[0],'temp',3,50,100)
