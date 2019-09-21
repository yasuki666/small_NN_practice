import numpy as np
import matplotlib.pyplot as plt

def create_data_set():
    group = np.array([[1.0,2.0],
                      [1.2,0.1],
                      [0.1,1.4],
                      [0.3,3.5],
                      [1.1,1.0],
                      [0.5,1.5]])
    labels = np.array(['A','A','B',"B","A",'B'])
    return group,labels

def KNN_classify(k,dis,X_train,x_train,Y_test):  #k为前k个距离，M，E代表欧式距离或曼哈顿距离，X_train为训练集，x_train为训练集标签，Y_test为测试集
    assert dis == 'E'or dis=='M'   #E代表欧拉距离，M代表曼哈顿距离
    num_test = Y_test.shape[0]
    label_list = []

    if(dis=='E'): #欧拉距离
        for i in range(num_test):
            distances = np.sqrt(np.sum(((X_train-np.tile(Y_test[i],(X_train.shape[0],1)))**2),axis=1))
            nearest_k = np.argsort(distances)
            top_k = nearest_k[:k]
            classCount = {}
            for i in top_k:
                classCount[x_train[i]] = classCount.get(x_train[i],0)+1
            sortedClassCount = sorted(classCount.items(),reverse = True)
            label_list.append(sortedClassCount[0][0])
        return np.array(label_list)

    if (dis == 'M'):  # 曼哈顿距离
        for i in range(num_test):
            distances = np.sum(np.abs(X_train - Y_test[i]),axis=1)
            nearest_k = np.argsort(distances)
            top_k = nearest_k[:k]
            classCount = {}
            for i in top_k:
                classCount[x_train[i]] = classCount.get(x_train[i], 0) + 1
            sortedClassCount = sorted(classCount.items(), reverse=True)
            label_list.append(sortedClassCount[0][0])
        return np.array(label_list)

if __name__== '__main__':
    group,labels  = create_data_set()
    Y_test = np.array([[1.0,2.1],[0.4,2.0]])
    Y_test_pred = KNN_classify(5,'M',group,labels,Y_test)
    print(Y_test_pred)
    plt.scatter(group[labels == 'A',0],group[labels=='A',1],color = 'r',marker = '*')
    plt.scatter(group[labels == 'B', 0], group[labels == 'B', 1], color='g', marker='+')
    plt.scatter(Y_test[Y_test_pred == 'A', 0], Y_test[Y_test_pred == 'A', 1], color='r', marker='*')
    plt.scatter(Y_test[Y_test_pred == 'B', 0], Y_test[Y_test_pred == 'B', 1], color='g', marker='+')
    plt.show()

