import os
import matplotlib.pyplot as plt
import datetime
import numpy as np
import pandas as pd
import pydotplus
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import Image
from six import StringIO
from sklearn import tree
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import export_graphviz
from sklearn.preprocessing import MinMaxScaler

def processing(feature_cols2,train_X):
    mmScaler=MinMaxScaler()
    DataTrain1=mmScaler.fit_transform(train_X)#得到预处理的样本DataTrain1
    # 需要选择有意义的特征输入机器学习的算法和模型进行训练,舍去方差为0的特征
    Scaler2=VarianceThreshold(0)
    DataTrain2=Scaler2.fit_transform(DataTrain1)#得到提取特征后的样本DataTrain2
    # 找到仍然保留的特征值
    featureIndex=Scaler2.get_support(True)
    for index in featureIndex:
        feature_cols2.append(feature_cols[index])
    print('After the processing,the numbers of feature is',len(feature_cols2))
    print('The index of the reserved data is', featureIndex)
    # 数据采样后经过classifier来分类
    return DataTrain2

class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=16, out_features=30)
        self.fc2 = nn.Linear(in_features=30, out_features=12)
        #这里设置输出特征为3不是问题
        # Pytorch有个要求，在使用CrossEntropyLoss这个函数进行验证时label必须是以0开始的,所以设置输出为3，确保可以分类0 1 2三种类
        # 如果设置输出为2的话，所分类的只有0和1，无法识别2，
        # 如果您非要让输出特征为2，请打开第77行附近的mappings以及lamada表达式的注释
        # 打开后0代表team1，1代表Team2
        self.output = nn.Linear(in_features=12, out_features=3)
    #这里使用的函数只是简单的simoid函数
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # 设置前面一层的激活函数
        x = self.output(x)
        x = F.softmax(x,1)
        return x

def Fusion(Voteing_y_pred,probDT,probMLP,probKNN,ScoreDT,ScoreMLP,ScoreKNN):
    for num in range(0,len(probMLP)):
        # 以下建立decision profile
        #平均值数据，记录属于每种情况的可能性
        meanResult = np.zeros((3, 2))
        meanResult[0]=probDT[num]*ScoreDT/100
        meanResult[1]=probMLP[num]*ScoreMLP/100
        meanResult[2] = probKNN[num]*ScoreKNN/100
        # resultMean代表两列的平均值
        resultMean = np.mean(meanResult, axis=0)
        meanProb.append(resultMean.tolist())
        # 从概率的大小来决定该组数据属于哪一类
        if resultMean[0] > resultMean[1]:
            Voteing_y_pred.append(1)
        else:
            Voteing_y_pred.append(2)

# 读取数据结构DataFrame
LolData = pd.read_csv('./new_data.csv')
LolData = LolData.iloc[1:] # 删除第一行（也就是命名行）
LolData.head()
# mappings = {
# 1: 0 ,
# 2: 1
# }
# LolData['winner'] =LolData['winner'].apply(lambda a: mappings[a])

feature_cols=['seasonId','firstBlood',
              'firstTower','firstInhibitor','firstBaron','firstDragon','firstRiftHerald',
              't1_towerKills','t1_inhibitorKills','t1_baronKills','t1_dragonKills','t1_riftHeraldKills',
            't2_towerKills','t2_inhibitorKills','t2_baronKills','t2_dragonKills','t2_riftHeraldKills'
            ]
X=LolData[feature_cols]
Y=LolData['winner'].values
print('Data import complete******************')
print('Before the processing,the numbers of feature is',len(feature_cols))
# 对于数据进行分类，一部分用来测试，一部分用来训练
train_X, test_X, train_y, test_y = train_test_split(X, Y, test_size=0.3, random_state=0)
feature_cols2=[]
feature_cols3=[]
# # 对于数据进行预处理的函数
# 对于数据进行处理
print('for the training_x')
DataTrain2=processing(feature_cols2,train_X)
print('for the testing_x')
DataTest2=processing(feature_cols3, test_X)

print('ANN******************')
# 使用ANN神经网络
DataTrain2 = torch.FloatTensor(DataTrain2)
DataTest2 = torch.FloatTensor(DataTest2)
y_train = torch.LongTensor(train_y)
y_test = torch.LongTensor(test_y)
starttimeANN = datetime.datetime.now()
model=ANN()
criterion = nn.CrossEntropyLoss()
learning_rate=0.01
# 使用optim包来定义优化算法
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# 将神经网络重复训练400次
epochs = 400
loss_arr = []
for i in range(epochs):
    y_hat = model.forward(DataTrain2)
    loss = criterion(y_hat, y_train)
    loss_arr.append(loss)
    # 每次训练前清零之前计算的梯度(导数)
    optimizer.zero_grad()
    # 根据误差反向传播计算误差对各个权重的导数
    loss.backward()
    # 根据优化器里面的算法自动调整神经网络权重
    optimizer.step()

ANN_pred_out=model(DataTest2)
_,ANN_predict_y = torch.max(ANN_pred_out, 1)
scoreANN=accuracy_score(test_y,ANN_predict_y)
print(ANN_predict_y)
print('scoreANN:', scoreANN)
endtimeANN = datetime.datetime.now()
print('the running timr of ANN(pytorch) is ', endtimeANN-starttimeANN)

# （2）使用 k-NN算法,在这里我已经使用for与np.where()循环调试过得分较大的距离,这里直接写入距离
print("KNN******************")
starttimeKNN = datetime.datetime.now()
best_score = 0.0
best_k = -1
knn_predict_y=[]
best_probKNN=[]
for k in range(1, 50):
    knn = KNeighborsClassifier(n_neighbors=k, weights='uniform', algorithm='auto', metric='euclidean')
    # 定义一个knn分类器对象
    knn.fit(DataTrain2, train_y)
    # 调用该对象的训练方法，主要接收两个参数：训练数据集及其样本标签
    knn_predict_y = knn.predict(DataTest2)
    scoreKNN = accuracy_score(test_y, knn_predict_y)
    probKNN=knn.predict_proba(DataTest2)
    if scoreKNN > best_score:
        best_score = scoreKNN
        best_k = k
        best_probKNN =probKNN
        best_KNN=knn

print("best_k = ", best_k)
print("best_score = ", best_score)
print('knn_predict_y:',knn_predict_y)
 #调用该对象的测试方法，主要接收一个参数：测试数据集
print('scoreKNN:', best_score)
endtimeKNN = datetime.datetime.now()
print('the running time of KNN is ', endtimeKNN-starttimeKNN)

print("DT******************")
# 设定最大深度不限的分类决策树
starttimeDT = datetime.datetime.now()
tree1= tree.DecisionTreeClassifier(criterion="gini",splitter="random",max_depth=None,min_samples_split=10,min_samples_leaf=5,min_weight_fraction_leaf=0.,
        max_features=None,random_state=None,max_leaf_nodes=None,class_weight=None)
# 拟合数据
tree1.fit(DataTrain2,train_y)
DT_y_pred=tree1.predict(DataTest2)
print('DT_y_pred:', DT_y_pred)
probDT= tree1.predict_proba(DataTest2)
# print("predict_pro:",probDT)
scoreDT=accuracy_score(test_y,DT_y_pred)
print('scoreDT:', scoreDT)
endtimeDT = datetime.datetime.now()
print('the running time of DT is ', endtimeDT-starttimeDT)

# # 图象可视化处理
# os.environ["PATH"] += os.pathsep + 'D:/Graphviz2.38/bin/'
# dot_data = StringIO()
# export_graphviz(tree1, out_file=dot_data,
# filled=True, rounded=True, special_characters=True,feature_names = feature_cols2,class_names=['1','2'])
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_png('D:/DataMining/diabetes.png')
# Image(graph.create_png())
# print('图象已经输出...')

# （4）使用 MLP算法
print("MLP******************")
starttimeMLP = datetime.datetime.now()
mlp = MLPClassifier(activation='relu',
        solver='sgd',alpha=0.00001,max_iter=600,shuffle=True,learning_rate_init=0.001,
        early_stopping=False, random_state=1)
mlp.fit(DataTrain2,train_y)
MLP_y_pred=mlp.predict(DataTest2)
print('MLP_predict_y:',MLP_y_pred)
probMLP=mlp.predict_proba(DataTest2)
# print('probMLP:',probMLP)
scoreMLP=accuracy_score(test_y,MLP_y_pred)
print('scoreMLP:',scoreMLP)
endtimeMLP = datetime.datetime.now()
print('the running time of MLP is ', endtimeMLP-starttimeMLP)

# 采用Fusion method方法
print('Fusion method*************')
Voteing_y_pred1 = []
meanProb=[]
starttimeFM = datetime.datetime.now()
Fusion(Voteing_y_pred1,probDT,probMLP,best_probKNN,scoreDT,scoreMLP,best_score)
FuScore=accuracy_score(test_y,Voteing_y_pred1)
print('VoteingScore',FuScore)
endtimeFM= datetime.datetime.now()
print('the running time of Fusion method is ',
      (endtimeFM-starttimeFM)+(endtimeMLP-starttimeMLP)+(endtimeDT-starttimeDT)+(endtimeKNN-starttimeKNN))
print("***************************************")
#
# 在得到最佳的分类器后，对test的数据进行读取
feature_cols4=[]
Voteing_y_pred2 = []
df2=pd.read_csv('./test_set.csv')
df2 = df2.iloc[1:] # 删除第一行（也就是命名行）
df2.head()
feature_cols_test=['seasonId','firstBlood',
              'firstTower','firstInhibitor','firstBaron','firstDragon','firstRiftHerald',
              't1_towerKills','t1_inhibitorKills','t1_baronKills','t1_dragonKills','t1_riftHeraldKills',
            't2_towerKills','t2_inhibitorKills','t2_baronKills','t2_dragonKills','t2_riftHeraldKills'
            ]
X2=LolData[feature_cols_test]
Y2=LolData['winner'].values
# 数据预处理
FinalX=processing(feature_cols4,X2)
probDT2= tree1.predict_proba(FinalX)
probKNN2=best_KNN.predict_proba(FinalX)
probMLP2=mlp.predict_proba(FinalX)
Fusion(Voteing_y_pred2,probDT2,probMLP2,probKNN2,scoreDT,scoreMLP,best_score)
print("The predicted result is ",Voteing_y_pred2)
FuScore2=accuracy_score(Y2,Voteing_y_pred2)
print('VoteingScore',FuScore2)



# 对于预测结果画图
print("The predicted result is in result.png")
plt.xlabel('Winner')             #设置x，y轴的标签
plt.ylabel('Number of matches')
plt.barh(range(len(Voteing_y_pred2)), Voteing_y_pred2,color='rgb')
plt.savefig('./result.png')
plt.show()
# for row in Voteing_y_pred2:
#     print(row)