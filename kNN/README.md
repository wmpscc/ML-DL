
# KNN介绍

k近邻法(k-nearest neighbors)是由Cover和Hart于1968年提出的，它是懒惰学习(lazy learning)的著名代表。<br>它的工作机制比较简单：
- 给定一个测试样本
- 计算它到训练样本的距离
- 取离测试样本最近的`k`个训练样本
- “投票法”选出在这k个样本中出现最多的类别，就是预测的结果

> 距离衡量的标准有很多，常见的有：$L_p$距离、切比雪夫距离、马氏距离、巴氏距离、余弦值等。

什么意思呢？先来看这张图

![2个类](https://github.com/wmpscc/ML-DL/raw/master/kNN/img/P1.png)

我们对应上面的流程来说
- 1.给定了红色和蓝色的训练样本，绿色为测试样本
- 2.计算绿色点到其他点的距离
- 3.选取离绿点最近的k个点
- 4.选取k个点中，同种颜色最多的类。例如：k=1时，k个点全是蓝色，那预测结果就是Class 1；k=3时，k个点中两个红色一个蓝色，那预测结果就是Class 2

## 举例

这里用`欧氏距离`作为距离的衡量标准，用鸢尾花数据集举例说明。
鸢尾花数据集有三个类别，每个类有150个样本，每个样本有4个特征。

先来回顾一下欧氏距离的定义(摘自维基百科)：<br>
> 在欧几里得空间中，点 x = (x1,...,xn) 和 y = (y1,...,yn) 之间的欧氏距离为<br>
 $d(x,y):={\sqrt  {(x_{1}-y_{1})^{2}+(x_{2}-y_{2})^{2}+\cdots +(x_{n}-y_{n})^{2}}}={\sqrt  {\sum _{{i=1}}^{n}(x_{i}-y_{i})^{2}}}$<br>
 
> 向量 ${\displaystyle {\vec {x}}}$的自然长度，即该点到原点的距离为<br>
 $\|{\vec  {x}}\|_{2}={\sqrt  {|x_{1}|^{2}+\cdots +|x_{n}|^{2}}}$<br>
 它是一个纯数值。在欧几里得度量下，两点之间线段最短。

现在给出六个训练样本，分为三类，每个样本有4个特征，编号为7的`名称`是我们要预测的。

|编号|花萼长度(cm)|花萼宽度(cm)|花瓣长度(cm)|花瓣宽度(cm)|名称|
|----|----------|-----------|-----------|----------|----|
|1|4.9|3.1|1.5|0.1|Iris setosa|
|2|5.4|3.7|1.5|0.2|Iris setosa|
|3|5.2|2.7|3.9|1.4|Iris versicolor|
|4|5.0|2.0|3.5|1.0|Iris versicolor|
|5|6.3|2.7|4.9|1.8|Iris virginica|
|6|6.7|3.3|5.7|2.1|Iris virginica|
|7|5.5|2.5|4.0|1.3|    ?    |

按照之前说的步骤，我们来计算测试样本到各个训练样本的距离。例如到第一个样本的距离$d_{1}=\sqrt{(4.9 - 5.5)^2 + (3.1 - 2.5)^2 + (1.5 - 4.0)^2 + (0.1 - 1.3)^2} = 2.9$

写一个函数来执行这个操作吧


```python
import numpy as np
def calc_distance(iA,iB):
    temp = np.subtract(iA, iB)      # 对应元素相减
    temp = np.power(temp, 2)        # 元素分别平方
    distance = np.sqrt(temp.sum())  # 先求和再开方
    return distance

testSample = np.array([5.5, 2.5, 4.0, 1.3])
print("Distance to 1:", calc_distance(np.array([4.9, 3.1, 1.5, 0.1]), testSample))
print("Distance to 2:", calc_distance(np.array([5.4, 3.7, 1.5, 0.2]), testSample))
print("Distance to 3:", calc_distance(np.array([5.2, 2.7, 3.9, 1.4]), testSample))
print("Distance to 4:", calc_distance(np.array([5.0, 2.0, 3.5, 1.0]), testSample))
print("Distance to 5:", calc_distance(np.array([6.3, 2.7, 4.9, 1.8]), testSample))
print("Distance to 6:", calc_distance(np.array([6.7, 3.3, 5.7, 2.1]), testSample))
```

    Distance to 1: 2.9
    Distance to 2: 2.98496231132
    Distance to 3: 0.387298334621
    Distance to 4: 0.916515138991
    Distance to 5: 1.31909059583
    Distance to 6: 2.36854385647


如果我们把k定为3，那么离测试样本最近3个依次是：

|编号|名称|
|----|---|
|3|Iris versicolor|
|4|Iris versicolor|
|5|Iris virginica|

显然测试样本属于`Iris versicolor`类的“票数”多一点，事实上它的确属于这个类。

## 优/缺点

这里参考了[CSDN芦金宇博客上的总结](https://blog.csdn.net/ch1209498273/article/details/78440276)

**优点**
- 简单好用，容易理解，精度高，理论成熟，既可以用来做分类也可以用来做回归；
- 可用于数值型数据和离散型数据；
- 训练时间复杂度为O(n)；无数据输入假定；
- 对异常值不敏感。


**缺点**
- 计算复杂性高；空间复杂性高；
- 样本不平衡问题（即有些类别的样本数量很多，而其它样本的数量很少）；
- 一般数值很大的时候不用这个，计算量太大。但是单个样本又不能太少，否则容易发生误分。
- 最大的缺点是无法给出数据的内在含义。

补充一点：由于它属于懒惰学习，因此需要大量的空间来存储训练实例，在预测时它还需要与已知所有实例进行比较，增大了计算量。

**这里介绍一下，当样本不平衡时的影响。**

![不平衡](https://github.com/wmpscc/ML-DL/raw/master/kNN/img/P2.png)

从直观上可以看出X应该属于$\omega_{1}$，这是理所应当的。对于Y看起来应该属于$\omega_{1}$，但事实上在k范围内，更多的点属于$\omega_{2}$，这就造成了错误分类。<br>

## 一个结论

在周志华编著的《机器学习》中证明了最近邻学习器的泛化错误率不超过贝叶斯最优分类器的错误率的两倍，在原书的226页，这里就不摘抄了。


# KNN代码实现

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
