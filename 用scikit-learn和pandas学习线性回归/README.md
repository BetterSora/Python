
# 1. 获取数据，定义问题

    我们用的是一个循环发电场的数据，共有9568个样本数据，每个数据有5列，分别是:AT（温度）, V（压力）, AP（湿度）, RH（压强）, PE（输出电力)。我们不用纠结于每项具体的意思。

    我们的问题是得到一个线性的关系，对应PE是样本输出，而AT/V/AP/RH这4个是样本特征， 机器学习的目的就是得到一个线性回归模型，即:

        PE=θ0+θ1∗AT+θ2∗V+θ3∗AP+θ4∗RH

    而需要学习的，就是θ0,θ1,θ2,θ3,θ4这5个参数。

# 2. 整理数据

    下载后的数据可以发现是一个压缩文件，解压后可以看到里面有一个xlsx文件，我们先用excel把它打开，接着“另存为“”csv格式，保存下来，后面我们就用这个csv来运行线性回归。

    打开这个csv可以发现数据已经整理好，没有非法数据，因此不需要做预处理。但是这些数据并没有归一化，也就是转化为均值0，方差1的格式。也不用我们搞，后面scikit-learn在线性回归时会先帮我们把归一化搞定。

# 3. 用pandas来读取数据


```python
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
```


```python
# read_csv里面的参数是csv在你电脑上的路径，此处csv文件放在notebook运行目录下面
data = pd.read_csv('ccpp.csv')
```


```python
# 读取前五行数据，如果是最后五行，用data.tail()
data.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AT</th>
      <th>V</th>
      <th>AP</th>
      <th>RH</th>
      <th>PE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8.34</td>
      <td>40.77</td>
      <td>1010.84</td>
      <td>90.01</td>
      <td>480.48</td>
    </tr>
    <tr>
      <th>1</th>
      <td>23.64</td>
      <td>58.49</td>
      <td>1011.40</td>
      <td>74.20</td>
      <td>445.75</td>
    </tr>
    <tr>
      <th>2</th>
      <td>29.74</td>
      <td>56.90</td>
      <td>1007.15</td>
      <td>41.91</td>
      <td>438.76</td>
    </tr>
    <tr>
      <th>3</th>
      <td>19.07</td>
      <td>49.69</td>
      <td>1007.22</td>
      <td>76.79</td>
      <td>453.09</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11.80</td>
      <td>40.66</td>
      <td>1017.13</td>
      <td>97.20</td>
      <td>464.43</td>
    </tr>
  </tbody>
</table>
</div>



# 4. 准备运行算法的数据


```python
data.shape
```




    (9568, 5)




```python
X = data[['AT', 'V', 'AP', 'RH']]
X.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AT</th>
      <th>V</th>
      <th>AP</th>
      <th>RH</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8.34</td>
      <td>40.77</td>
      <td>1010.84</td>
      <td>90.01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>23.64</td>
      <td>58.49</td>
      <td>1011.40</td>
      <td>74.20</td>
    </tr>
    <tr>
      <th>2</th>
      <td>29.74</td>
      <td>56.90</td>
      <td>1007.15</td>
      <td>41.91</td>
    </tr>
    <tr>
      <th>3</th>
      <td>19.07</td>
      <td>49.69</td>
      <td>1007.22</td>
      <td>76.79</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11.80</td>
      <td>40.66</td>
      <td>1017.13</td>
      <td>97.20</td>
    </tr>
  </tbody>
</table>
</div>




```python
y = data[['PE']]
y.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>480.48</td>
    </tr>
    <tr>
      <th>1</th>
      <td>445.75</td>
    </tr>
    <tr>
      <th>2</th>
      <td>438.76</td>
    </tr>
    <tr>
      <th>3</th>
      <td>453.09</td>
    </tr>
    <tr>
      <th>4</th>
      <td>464.43</td>
    </tr>
  </tbody>
</table>
</div>



# 5. 划分训练集和测试集


```python
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
```

    F:\Anaconda3\lib\site-packages\sklearn\cross_validation.py:44: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)
    


```python
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
```

    (7176, 4)
    (7176, 1)
    (2392, 4)
    (2392, 1)
    

# 6. 运行scikit-learn的线性模型


```python
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)




```python
print(linreg.intercept_)
print(linreg.coef_)
```

    [ 447.06297099]
    [[-1.97376045 -0.23229086  0.0693515  -0.15806957]]
    

PE=447.06297099−1.97376045∗AT−0.23229086∗V+0.0693515∗AP−0.15806957∗RH

# 7. 模型评价


```python
#模型拟合测试集
y_pred = linreg.predict(X_test)
from sklearn import metrics
#用scikit-learn计算MSE
print("MSE:", metrics.mean_squared_error(y_test, y_pred))
#用scikit-learn计算RMSE
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
```

    MSE: 20.0804012021
    RMSE: 4.48111606657
    

得到了MSE或者RMSE，如果我们用其他方法得到了不同的系数，需要选择模型时，就用MSE小的时候对应的参数。

比如这次我们用AT， V，AP这3个列作为样本特征。不要RH， 输出仍然是PE。代码如下：


```python
X = data[['AT', 'V', 'AP']]
y = data[['PE']]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)
#模型拟合测试集
y_pred = linreg.predict(X_test)
from sklearn import metrics
# 用scikit-learn计算MSE
print("MSE:",metrics.mean_squared_error(y_test, y_pred))
# 用scikit-learn计算RMSE
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
```

    MSE: 23.2089074701
    RMSE: 4.81756239919
    

可以看出，去掉RH后，模型拟合的没有加上RH的好，MSE变大了。

# 8. 交叉验证


```python
#我们可以通过交叉验证来持续优化模型，我们采用10折交叉验证，即cross_val_predict中的cv参数为10
X = data[['AT', 'V', 'AP', 'RH']]
y = data[['PE']]
from sklearn.model_selection import cross_val_predict
predicted = cross_val_predict(linreg, X, y, cv=10)
# 用scikit-learn计算MSE
print("MSE:",metrics.mean_squared_error(y, predicted))
# 用scikit-learn计算RMSE
print("RMSE:",np.sqrt(metrics.mean_squared_error(y, predicted)))
```

    MSE: 20.7955974619
    RMSE: 4.56021901469
    

# 9. 画图观察结果


```python
#这里画图真实值和预测值的变化关系，离中间的直线y=x直接越近的点代表预测损失越低
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
```


![png](output_27_0.png)



```python

```
