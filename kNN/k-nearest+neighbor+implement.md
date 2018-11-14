
# 代码实现
知道KNN的原理后，应该可以很轻易的写出代码了，这里介绍一下在距离计算上的优化，在结尾给上完整代码(代码比较乱，知道思想就好)。

函数的输入：`train_X`、`test_X`是numpy array，假设它们的shape分别为(n, 4)、(m, 4)；
要求输出的是它们两点间的距离矩阵，shape为(n, m)。

不就是计算两点之间的距离，再存起来吗，直接暴力上啊︿(￣︶￣)︿，于是就有了下面的

### 双循环暴力实现

``` Python
def euclideanDistance_two_loops(train_X, test_X):
    ''' Calculate two-point Euclidean distance
    Args:
        train_X: Numpy array type training set.
        test_X: Numpy array type test set.
    Returns:
        dists: A numpy array of shape (num_test, num_train) where dists[i, j]
        is the Euclidean distance between the ith test point and the jth training point.
    '''
    num_test = test_X.shape[0]
    num_train = train_X.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
        for j in range(num_train):
            test_line = test_X[i]
            train_line = train_X[j]
            temp = np.subtract(test_line, train_line)
            temp = np.power(temp, 2)
            dists[i][j] = np.sqrt(temp.sum())
    return dists
```

不知道你有没有想过，这里的样本数只有100多个，所以时间上感觉没有等太久，但是当样本量非常大的时候呢，双循环计算的弊端就显露出来了。解决方案是把它转换为两个矩阵之间的运算，这样就能避免使用循环了。

## 转化为矩阵运算实现
### L2 Distance

此处参考[CSDNfrankzd的博客](https://blog.csdn.net/frankzd/article/details/80251042)<br>
记测试集矩阵P的大小为$M*D$，训练集矩阵C的大小为$N*D$（测试集中共有M个点，每个点为D维特征向量。训练集中共有N个点，每个点为D维特征向量） <br><br>
记$P_{i}$是P的第i行$P_i = [ P_{i1}\quad P_{i2} \cdots P_{iD}]$，记$C_{j}$是C的第j行$C_j = [ C_{j1} C_{j2} \cdots \quad C_{jD}]$
- 首先计算Pi和Cj之间的距离dist(i,j) <br><br>
$d(P_i,C_j) = \sqrt{(P_{i1}-C_{j1})^2+(P_{i2}-C_{j2})^2+\cdots+(P_{iD}-C_{jD})^2}\\ =\sqrt{(P_{i1}^2+P_{i2}^2+\cdots+P_{iD}^2)+(C_{j1}^2+C_{j2}^2+\cdots+C_{jD}^2)-2\times(P_{i1}C_{j1}+P_{i2}C_{j2}+\cdots+P_{iD}C_{iD})}\\=\sqrt{\left \| P_i\right \|^2+\left\|C_j\right\|^2-2\times P_iC_j^T}$
<br><br>
- 我们可以推广到距离矩阵的第i行的计算公式 <br><br>
$dist[i]=\sqrt{(\left\|P_i\right\|^2\quad \left\|P_i\right\|^2 \cdots \left\|P_i\right\|^2)+(\left\|C_1\right\|^2 \quad \left\|C_2\right\|^2 \cdots \left\|C_N\right\|^2)-2\times P_i(C_1^T \quad C_2^T \cdots C_N^T)}\\=\sqrt{(\left\|P_i\right\|^2\quad \left\|P_i\right\|^2 \cdots \left\|P_i\right\|^2)+(\left\|C_1\right\|^2 \quad \left\|C_2\right\|^2 \cdots \left\|C_N\right\|^2)-2\times P_iC^T}$
<br><br>
- 继续将公式推广为整个距离矩阵(也就是完全平方公式) <br><br>
$dist=\sqrt{\begin{pmatrix}\left\|P_1\right\|^2 & \left\|P_1\right\|^2 & \cdots & \left\|P_1\right\|^2\\\left\|P_2\right\|^2 & \left\|P_2\right\|^2 & \cdots & \left\|P_2\right\|^2\\\vdots & \vdots & \ddots & \vdots \\\left\|P_M\right\|^2 & \left\|P_M\right\|^2 & \cdots & \left\|P_M\right\|^2 \end{pmatrix}+\begin{pmatrix}\left\|C_1\right\|^2 & \left\|C_2\right\|^2 & \cdots & \left\|C_N\right\|^2\\\left\|C_1\right\|^2 & \left\|C_2\right\|^2 & \cdots & \left\|C_N\right\|^2\\\vdots & \vdots & \ddots & \vdots \\\left\|C_1\right\|^2 & \left\|C_2\right\|^2 & \cdots & \left\|C_N\right\|^2 \end{pmatrix}-2\times PC^T}$

知道距离矩阵怎么算出来的之后，在代码上只需要套公式带入就能实现了。

``` Python
def euclideanDistance_no_loops(train_X, test_X):
    num_test = test_X.shape[0]
    num_train = train_X.shape[0]

    sum_train = np.power(train_X, 2)
    sum_train = sum_train.sum(axis=1)
    sum_train = sum_train * np.ones((num_test, num_train))

    sum_test = np.power(test_X, 2)
    sum_test = sum_test.sum(axis=1)
    sum_test = sum_test * np.ones((1, sum_train.shape[0]))
    sum_test = sum_test.T

    sum = sum_train + sum_test - 2 * np.dot(test_X, train_X.T)
    dists = np.sqrt(sum)
    return dists
```

是不是很简单，这里两种方法我们衡量两点间距离的标准是`欧氏距离`。如果想用其他的标准呢，比如$L_{1}$距离该怎么实现呢，这里我参照上面推导公式的思想得出了计算$L_{1}$距离的矩阵运算。

### L1 Distance

记测试集矩阵P的大小为$M*D$，训练集矩阵C的大小为$N*D$（测试集中共有M个点，每个点为D维特征向量。训练集中共有N个点，每个点为D维特征向量） <br><br>
记$P_{i}$是P的第i行$P_i = [ P_{i1}\quad P_{i2} \cdots P_{iD}]$，记$C_{j}$是C的第j行$C_j = [ C_{j1} C_{j2} \cdots \quad C_{jD}]$
- 首先计算Pi和Cj之间的距离dist(i,j)<br><br>
$d(P_{i}, C_{j}) = |P_{i1} - C_{j1}| + |P_{i2} - C_{j2}| + \cdots + |P_{iD} - C_{jD}|$


- 我们可以推广到距离矩阵的第i行的计算公式 <br><br>
$dist[i] = \left | [P_{i}\quad P_{i} \cdots P_{i}] - [C_{1}\quad C_{2}\cdots C_{N}] \right|=[|P_{i} - C_{1}|\quad |P_{i} - C_{2}| \cdots |P_{i} - C_{N}|]$


- 继续将公式推广为整个距离矩阵 <br><br>
$dist = \left |  \left[\begin{matrix}P_{1} & P_{1} & \cdots & P_{1} \cr P_{2} & P_{2} & \cdots & P_{2} \cr \vdots & \vdots & \ddots & \vdots \cr P_{M} & P_{M} & \cdots & P_{M} \end{matrix} \right]  - \left[\begin{matrix}C_{1} & C_{2} & \cdots & C_{N} \cr C_{1} & C_{2} & \cdots & C_{N} \cr \vdots & \vdots & \ddots & \vdots \cr C_{1} & C_{2} & \cdots & C_{N} \end{matrix} \right] \right | = \left [\begin{matrix} |P_{1} - C_{1}| & |P_{1} - C_{2}| & \cdots & |P_{1} - C_{N}| \cr
|P_{2} - C_{1}| & |P_{2}-C_{2}| & \cdots & |P_{2} - C_{N}| \cr
\vdots & \vdots & \ddots & \vdots \cr 
|P_{M} - C_{1}| & |P_{M} - C_{2}| & \cdots & |P_{M} - C_{N}|\end{matrix}\right ]$

<br>其中$P_i = [ P_{i1}\quad P_{i2} \cdots P_{iD}]$、$C_j = [ C_{j1} C_{j2} \cdots \quad C_{jD}]$

``` Python
def l1_distance_no_loops(train_X, test_X):
    num_test = test_X.shape[0]
    num_train = train_X.shape[0]

    test = np.tile(test_X, (num_train, 1, 1))
    train = np.tile(train_X, (num_test, 1, 1))
    train = np.transpose(train, axes=(1, 0, 2))

    sum = np.subtract(test, train)
    sum = np.abs(sum)
    sum = np.sum(sum, axis=2)
    dists = sum.T
    return dists
```

由于测试集样本数量有限，两种距离衡量标准下的准确率分别是0.94和0.98。

## 完整代码

``` Python
import pandas as pd
import numpy as np
import random
from collections import Counter


def load_data(path, rate=0.8):
    ''' Making dataSet

    Args:
        path: DataSet file path.
        rate: TrainSet percentage.
    Returns:
        trainSet: Training set.
        trainLabel: Training label.
        testSet: Test set.
        testLabel: Test label.
    '''
    data = pd.read_csv(path)
    randIndex = random.sample(range(0, len(data)), len(data))
    trainSet = data.loc[randIndex[:int(len(data) * rate)]]
    testSet = data.loc[randIndex[int(len(data) * rate):]]
    trainLabel = trainSet['name']
    testLabel = testSet['name']
    trainSet.drop('name', axis=1, inplace=True)
    testSet.drop('name', axis=1, inplace=True)
    return np.array(trainSet), np.array(trainLabel), np.array(testSet), np.array(testLabel)


def euclideanDistance_two_loops(train_X, test_X):
    ''' Calculate two-point Euclidean distance

    Args:
        train_X: Numpy array type training set.
        test_X: Numpy array type test set.
    Returns:
        dists: A numpy array of shape (num_test, num_train) where dists[i, j]
        is the Euclidean distance between the ith test point and the jth training point.
    '''

    num_test = test_X.shape[0]
    num_train = train_X.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
        for j in range(num_train):
            test_line = test_X[i]
            train_line = train_X[j]
            temp = np.subtract(test_line, train_line)
            temp = np.power(temp, 2)
            dists[i][j] = np.sqrt(temp.sum())
    return dists


def euclideanDistance_no_loops(train_X, test_X):
    '''Calculate two-point Euclidean distance without loops

    Input / Output: Same as euclideanDistance_two_loops
    '''

    num_test = test_X.shape[0]
    num_train = train_X.shape[0]

    sum_train = np.power(train_X, 2)
    sum_train = sum_train.sum(axis=1)
    sum_train = sum_train * np.ones((num_test, num_train))

    sum_test = np.power(test_X, 2)
    sum_test = sum_test.sum(axis=1)
    sum_test = sum_test * np.ones((1, sum_train.shape[0]))
    sum_test = sum_test.T

    sum = sum_train + sum_test - 2 * np.dot(test_X, train_X.T)
    dists = np.sqrt(sum)
    return dists


def l1_distance_no_loops(train_X, test_X):
    '''Calculate two-point L1 distance
    Args:
        train_X: Numpy array type training set.
        test_X: Numpy array type test set.
    Returns:
        dists: A numpy array of shape (num_test, num_train) where dists[i, j]
        is the Euclidean distance between the ith test point and the jth training point.
    '''

    num_test = test_X.shape[0]
    num_train = train_X.shape[0]

    test = np.tile(test_X, (num_train, 1, 1))
    train = np.tile(train_X, (num_test, 1, 1))
    train = np.transpose(train, axes=(1, 0, 2))

    sum = np.subtract(test, train)
    sum = np.abs(sum)
    sum = np.sum(sum, axis=2)
    dists = sum.T
    return dists


def predict_labels(dists, labels, k=1):
    ''' To predict a label for each test point.
    Args:
        dists: a numpy array of shape (num_test, num_train) where dists[i, j].
        labels: each test point real label.

    Returns:
        y_pred: knn predict target label.
    '''

    num_test = dists.shape[0]
    y_pred = []
    for i in range(num_test):
        index = np.argsort(dists[i])
        index = index[:k]
        closest_y = labels[index]
        name, _ = Counter(closest_y).most_common(1)[0]
        y_pred.append(name)
    return y_pred


def getAccuracy(y_pred, y):
    ''' Calculate the predict accuracy.
    Args:
        y_pred: knn predict target label.
        y: each test point real label.

    Returns:
        accuracy: average accuracy in test set.
    '''
    num_correct = np.sum(y_pred == y)
    accuracy = float(num_correct) / len(y)
    return accuracy


if __name__ == "__main__":
    trainData, trainLabel, testData, testLabel = load_data("irisdata.txt", 0.67)
    dists = l1_distance_no_loops(trainData, testData)
    y_pred = predict_labels(dists, trainLabel, k=3)
    accuracy = getAccuracy(y_pred, testLabel)
    print(accuracy)

```
