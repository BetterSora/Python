
# 数据加载、存储与文件格式


```python
from __future__ import division
from numpy.random import randn
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
np.random.seed(12345)
plt.rc('figure', figsize=(10, 6))
from pandas import Series, DataFrame
import pandas as pd
np.set_printoptions(precision=4)
```

## 读写文本格式的数据


```python
# 以逗号分隔的csv文本文件，该文件有标题行
df = pd.read_csv('ex1.csv')
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>hello</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
      <td>12</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
# read_table需要指定分隔符
pd.read_table('ex1.csv', sep=',')
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>hello</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
      <td>12</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 该文件没有标题行
!type ex2.csv
```

    1,2,3,4,hello
    5,6,7,8,world
    9,10,11,12,foo
    


```python
# 让pandas为其分配默认列名
pd.read_csv('ex2.csv', header=None)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>hello</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
      <td>12</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 自定义列名
pd.read_csv('ex2.csv', names=['a', 'b', 'c', 'd', 'message'])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>hello</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
      <td>12</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 将message列做成DataFrame索引
names = ['a', 'b', 'c', 'd', 'message']
pd.read_csv('ex2.csv', names=names, index_col='message')
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
    <tr>
      <th>message</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>hello</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>world</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th>foo</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 将多个列做成一个层次化索引，只需传入由列编号或列名组成的列表即可
!type csv_mindex.csv
parsed = pd.read_csv('csv_mindex.csv', index_col=['key1', 'key2'])
parsed
```

    key1,key2,value1,value2
    one,a,1,2
    one,b,3,4
    one,c,5,6
    one,d,7,8
    two,a,9,10
    two,b,11,12
    two,c,13,14
    two,d,15,16
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>value1</th>
      <th>value2</th>
    </tr>
    <tr>
      <th>key1</th>
      <th>key2</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">one</th>
      <th>a</th>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>b</th>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>c</th>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>d</th>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">two</th>
      <th>a</th>
      <td>9</td>
      <td>10</td>
    </tr>
    <tr>
      <th>b</th>
      <td>11</td>
      <td>12</td>
    </tr>
    <tr>
      <th>c</th>
      <td>13</td>
      <td>14</td>
    </tr>
    <tr>
      <th>d</th>
      <td>15</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 有些表格可能不是用固定的分隔符去分隔字段的，该文件各个字段由数量不定的空白符分隔
list(open('ex3.txt'))
```




    ['            A         B         C\n',
     'aaa -0.264438 -1.026059 -0.619500\n',
     'bbb  0.927272  0.302904 -0.032399\n',
     'ccc -0.264273 -0.386314 -0.217601\n',
     'ddd -0.871858 -0.348382  1.100491\n']




```python
# 传入正则表达式来处理，因为列名的数量比列的数量少1，所以read_table推断第一列应该是DataFrame的索引
result = pd.read_table('ex3.txt', sep='\s+')
result
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>aaa</th>
      <td>-0.264438</td>
      <td>-1.026059</td>
      <td>-0.619500</td>
    </tr>
    <tr>
      <th>bbb</th>
      <td>0.927272</td>
      <td>0.302904</td>
      <td>-0.032399</td>
    </tr>
    <tr>
      <th>ccc</th>
      <td>-0.264273</td>
      <td>-0.386314</td>
      <td>-0.217601</td>
    </tr>
    <tr>
      <th>ddd</th>
      <td>-0.871858</td>
      <td>-0.348382</td>
      <td>1.100491</td>
    </tr>
  </tbody>
</table>
</div>




```python
!type ex4.csv
# skiprows跳过文件的第一行、第三行和第四行
pd.read_csv('ex4.csv', skiprows=[0, 2, 3])
```

    # hey!
    a,b,c,d,message
    # just wanted to make things more difficult for you
    # who reads CSV files with computers, anyway?
    1,2,3,4,hello
    5,6,7,8,world
    9,10,11,12,foo
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>hello</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
      <td>12</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
# pandas会用一组经常出现的标记值识别缺失值
!type ex5.csv
result = pd.read_csv('ex5.csv')
pd.isnull(result)
```

    something,a,b,c,d,message
    one,1,2,3,4,NA
    two,5,6,,8,world
    three,9,10,11,12,foo
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>something</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
# na_values可以接受一组用于表示缺失值的字符串
result = pd.read_csv('ex5.csv', na_values=['NULL'])
result
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>something</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one</td>
      <td>1</td>
      <td>2</td>
      <td>3.0</td>
      <td>4</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>two</td>
      <td>5</td>
      <td>6</td>
      <td>NaN</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>three</td>
      <td>9</td>
      <td>10</td>
      <td>11.0</td>
      <td>12</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 可以用一个字典为各列指定不同的NA标记值
sentinels = {'message': ['foo', 'NA'], 'something': ['two']}
pd.read_csv('ex5.csv', na_values=sentinels)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>something</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one</td>
      <td>1</td>
      <td>2</td>
      <td>3.0</td>
      <td>4</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>5</td>
      <td>6</td>
      <td>NaN</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>three</td>
      <td>9</td>
      <td>10</td>
      <td>11.0</td>
      <td>12</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### 逐块读取文本文件


```python
result = pd.read_csv('ex6.csv')
result
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>two</th>
      <th>three</th>
      <th>four</th>
      <th>key</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.467976</td>
      <td>-0.038649</td>
      <td>-0.295344</td>
      <td>-1.824726</td>
      <td>L</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.358893</td>
      <td>1.404453</td>
      <td>0.704965</td>
      <td>-0.200638</td>
      <td>B</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.501840</td>
      <td>0.659254</td>
      <td>-0.421691</td>
      <td>-0.057688</td>
      <td>G</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.204886</td>
      <td>1.074134</td>
      <td>1.388361</td>
      <td>-0.982404</td>
      <td>R</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.354628</td>
      <td>-0.133116</td>
      <td>0.283763</td>
      <td>-0.837063</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.817480</td>
      <td>0.742273</td>
      <td>0.419395</td>
      <td>-2.251035</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.776764</td>
      <td>0.935518</td>
      <td>-0.332872</td>
      <td>-1.875641</td>
      <td>U</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-0.913135</td>
      <td>1.530624</td>
      <td>-0.572657</td>
      <td>0.477252</td>
      <td>K</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.358480</td>
      <td>-0.497572</td>
      <td>-0.367016</td>
      <td>0.507702</td>
      <td>S</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-1.740877</td>
      <td>-1.160417</td>
      <td>-1.637830</td>
      <td>2.172201</td>
      <td>G</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.240564</td>
      <td>-0.328249</td>
      <td>1.252155</td>
      <td>1.072796</td>
      <td>8</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.764018</td>
      <td>1.165476</td>
      <td>-0.639544</td>
      <td>1.495258</td>
      <td>R</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.571035</td>
      <td>-0.310537</td>
      <td>0.582437</td>
      <td>-0.298765</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2.317658</td>
      <td>0.430710</td>
      <td>-1.334216</td>
      <td>0.199679</td>
      <td>P</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1.547771</td>
      <td>-1.119753</td>
      <td>-2.277634</td>
      <td>0.329586</td>
      <td>J</td>
    </tr>
    <tr>
      <th>15</th>
      <td>-1.310608</td>
      <td>0.401719</td>
      <td>-1.000987</td>
      <td>1.156708</td>
      <td>E</td>
    </tr>
    <tr>
      <th>16</th>
      <td>-0.088496</td>
      <td>0.634712</td>
      <td>0.153324</td>
      <td>0.415335</td>
      <td>B</td>
    </tr>
    <tr>
      <th>17</th>
      <td>-0.018663</td>
      <td>-0.247487</td>
      <td>-1.446522</td>
      <td>0.750938</td>
      <td>A</td>
    </tr>
    <tr>
      <th>18</th>
      <td>-0.070127</td>
      <td>-1.579097</td>
      <td>0.120892</td>
      <td>0.671432</td>
      <td>F</td>
    </tr>
    <tr>
      <th>19</th>
      <td>-0.194678</td>
      <td>-0.492039</td>
      <td>2.359605</td>
      <td>0.319810</td>
      <td>H</td>
    </tr>
    <tr>
      <th>20</th>
      <td>-0.248618</td>
      <td>0.868707</td>
      <td>-0.492226</td>
      <td>-0.717959</td>
      <td>W</td>
    </tr>
    <tr>
      <th>21</th>
      <td>-1.091549</td>
      <td>-0.867110</td>
      <td>-0.647760</td>
      <td>-0.832562</td>
      <td>C</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.641404</td>
      <td>-0.138822</td>
      <td>-0.621963</td>
      <td>-0.284839</td>
      <td>C</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1.216408</td>
      <td>0.992687</td>
      <td>0.165162</td>
      <td>-0.069619</td>
      <td>V</td>
    </tr>
    <tr>
      <th>24</th>
      <td>-0.564474</td>
      <td>0.792832</td>
      <td>0.747053</td>
      <td>0.571675</td>
      <td>I</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1.759879</td>
      <td>-0.515666</td>
      <td>-0.230481</td>
      <td>1.362317</td>
      <td>S</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.126266</td>
      <td>0.309281</td>
      <td>0.382820</td>
      <td>-0.239199</td>
      <td>L</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1.334360</td>
      <td>-0.100152</td>
      <td>-0.840731</td>
      <td>-0.643967</td>
      <td>6</td>
    </tr>
    <tr>
      <th>28</th>
      <td>-0.737620</td>
      <td>0.278087</td>
      <td>-0.053235</td>
      <td>-0.950972</td>
      <td>J</td>
    </tr>
    <tr>
      <th>29</th>
      <td>-1.148486</td>
      <td>-0.986292</td>
      <td>-0.144963</td>
      <td>0.124362</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9970</th>
      <td>0.633495</td>
      <td>-0.186524</td>
      <td>0.927627</td>
      <td>0.143164</td>
      <td>4</td>
    </tr>
    <tr>
      <th>9971</th>
      <td>0.308636</td>
      <td>-0.112857</td>
      <td>0.762842</td>
      <td>-1.072977</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9972</th>
      <td>-1.627051</td>
      <td>-0.978151</td>
      <td>0.154745</td>
      <td>-1.229037</td>
      <td>Z</td>
    </tr>
    <tr>
      <th>9973</th>
      <td>0.314847</td>
      <td>0.097989</td>
      <td>0.199608</td>
      <td>0.955193</td>
      <td>P</td>
    </tr>
    <tr>
      <th>9974</th>
      <td>1.666907</td>
      <td>0.992005</td>
      <td>0.496128</td>
      <td>-0.686391</td>
      <td>S</td>
    </tr>
    <tr>
      <th>9975</th>
      <td>0.010603</td>
      <td>0.708540</td>
      <td>-1.258711</td>
      <td>0.226541</td>
      <td>K</td>
    </tr>
    <tr>
      <th>9976</th>
      <td>0.118693</td>
      <td>-0.714455</td>
      <td>-0.501342</td>
      <td>-0.254764</td>
      <td>K</td>
    </tr>
    <tr>
      <th>9977</th>
      <td>0.302616</td>
      <td>-2.011527</td>
      <td>-0.628085</td>
      <td>0.768827</td>
      <td>H</td>
    </tr>
    <tr>
      <th>9978</th>
      <td>-0.098572</td>
      <td>1.769086</td>
      <td>-0.215027</td>
      <td>-0.053076</td>
      <td>A</td>
    </tr>
    <tr>
      <th>9979</th>
      <td>-0.019058</td>
      <td>1.964994</td>
      <td>0.738538</td>
      <td>-0.883776</td>
      <td>F</td>
    </tr>
    <tr>
      <th>9980</th>
      <td>-0.595349</td>
      <td>0.001781</td>
      <td>-1.423355</td>
      <td>-1.458477</td>
      <td>M</td>
    </tr>
    <tr>
      <th>9981</th>
      <td>1.392170</td>
      <td>-1.396560</td>
      <td>-1.425306</td>
      <td>-0.847535</td>
      <td>H</td>
    </tr>
    <tr>
      <th>9982</th>
      <td>-0.896029</td>
      <td>-0.152287</td>
      <td>1.924483</td>
      <td>0.365184</td>
      <td>6</td>
    </tr>
    <tr>
      <th>9983</th>
      <td>-2.274642</td>
      <td>-0.901874</td>
      <td>1.500352</td>
      <td>0.996541</td>
      <td>N</td>
    </tr>
    <tr>
      <th>9984</th>
      <td>-0.301898</td>
      <td>1.019906</td>
      <td>1.102160</td>
      <td>2.624526</td>
      <td>I</td>
    </tr>
    <tr>
      <th>9985</th>
      <td>-2.548389</td>
      <td>-0.585374</td>
      <td>1.496201</td>
      <td>-0.718815</td>
      <td>D</td>
    </tr>
    <tr>
      <th>9986</th>
      <td>-0.064588</td>
      <td>0.759292</td>
      <td>-1.568415</td>
      <td>-0.420933</td>
      <td>E</td>
    </tr>
    <tr>
      <th>9987</th>
      <td>-0.143365</td>
      <td>-1.111760</td>
      <td>-1.815581</td>
      <td>0.435274</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9988</th>
      <td>-0.070412</td>
      <td>-1.055921</td>
      <td>0.338017</td>
      <td>-0.440763</td>
      <td>X</td>
    </tr>
    <tr>
      <th>9989</th>
      <td>0.649148</td>
      <td>0.994273</td>
      <td>-1.384227</td>
      <td>0.485120</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>9990</th>
      <td>-0.370769</td>
      <td>0.404356</td>
      <td>-1.051628</td>
      <td>-1.050899</td>
      <td>8</td>
    </tr>
    <tr>
      <th>9991</th>
      <td>-0.409980</td>
      <td>0.155627</td>
      <td>-0.818990</td>
      <td>1.277350</td>
      <td>W</td>
    </tr>
    <tr>
      <th>9992</th>
      <td>0.301214</td>
      <td>-1.111203</td>
      <td>0.668258</td>
      <td>0.671922</td>
      <td>A</td>
    </tr>
    <tr>
      <th>9993</th>
      <td>1.821117</td>
      <td>0.416445</td>
      <td>0.173874</td>
      <td>0.505118</td>
      <td>X</td>
    </tr>
    <tr>
      <th>9994</th>
      <td>0.068804</td>
      <td>1.322759</td>
      <td>0.802346</td>
      <td>0.223618</td>
      <td>H</td>
    </tr>
    <tr>
      <th>9995</th>
      <td>2.311896</td>
      <td>-0.417070</td>
      <td>-1.409599</td>
      <td>-0.515821</td>
      <td>L</td>
    </tr>
    <tr>
      <th>9996</th>
      <td>-0.479893</td>
      <td>-0.650419</td>
      <td>0.745152</td>
      <td>-0.646038</td>
      <td>E</td>
    </tr>
    <tr>
      <th>9997</th>
      <td>0.523331</td>
      <td>0.787112</td>
      <td>0.486066</td>
      <td>1.093156</td>
      <td>K</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>-0.362559</td>
      <td>0.598894</td>
      <td>-1.843201</td>
      <td>0.887292</td>
      <td>G</td>
    </tr>
    <tr>
      <th>9999</th>
      <td>-0.096376</td>
      <td>-1.012999</td>
      <td>-0.657431</td>
      <td>-0.573315</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>10000 rows × 5 columns</p>
</div>




```python
# 通过nrows指定读取几行
pd.read_csv('ex6.csv', nrows=5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>two</th>
      <th>three</th>
      <th>four</th>
      <th>key</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.467976</td>
      <td>-0.038649</td>
      <td>-0.295344</td>
      <td>-1.824726</td>
      <td>L</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.358893</td>
      <td>1.404453</td>
      <td>0.704965</td>
      <td>-0.200638</td>
      <td>B</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.501840</td>
      <td>0.659254</td>
      <td>-0.421691</td>
      <td>-0.057688</td>
      <td>G</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.204886</td>
      <td>1.074134</td>
      <td>1.388361</td>
      <td>-0.982404</td>
      <td>R</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.354628</td>
      <td>-0.133116</td>
      <td>0.283763</td>
      <td>-0.837063</td>
      <td>Q</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 要逐块读取文件，需要设置chunksize(行数)
chunker = pd.read_csv('ex6.csv', chunksize=1000)
# read_csv所返回的这个TextParser对象使你可以根据chunksize对文件进行逐块迭代
chunker
```




    <pandas.io.parsers.TextFileReader at 0x261a4c3d9e8>




```python
chunker = pd.read_csv('ex6.csv', chunksize=1000)

tot = Series([])
for piece in chunker:
    tot = tot.add(piece['key'].value_counts(), fill_value=0)

tot = tot.order(ascending=False)
```

    F:\Anaconda3\lib\site-packages\ipykernel\__main__.py:7: FutureWarning: order is deprecated, use sort_values(...)
    


```python
tot[:10]
```




    E    368.0
    X    364.0
    L    346.0
    O    343.0
    Q    340.0
    M    338.0
    J    337.0
    F    335.0
    K    334.0
    H    330.0
    dtype: float64



### 将数据写出到文本格式


```python
data = pd.read_csv('ex5.csv')
data
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>something</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one</td>
      <td>1</td>
      <td>2</td>
      <td>3.0</td>
      <td>4</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>two</td>
      <td>5</td>
      <td>6</td>
      <td>NaN</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>three</td>
      <td>9</td>
      <td>10</td>
      <td>11.0</td>
      <td>12</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.to_csv('out.csv')
!type out.csv
```

    ,something,a,b,c,d,message
    0,one,1,2,3.0,4,
    1,two,5,6,,8,world
    2,three,9,10,11.0,12,foo
    


```python
# 将数据写出到控制台
data.to_csv(sys.stdout, sep='|')
```

    |something|a|b|c|d|message
    0|one|1|2|3.0|4|
    1|two|5|6||8|world
    2|three|9|10|11.0|12|foo
    


```python
# 缺失值在输出结果中会被表示为空字符串，na_rep可以指定标记值
data.to_csv(sys.stdout, na_rep='NULL')
```

    ,something,a,b,c,d,message
    0,one,1,2,3.0,4,NULL
    1,two,5,6,NULL,8,world
    2,three,9,10,11.0,12,foo
    


```python
# 禁用行列标签
data.to_csv(sys.stdout, index=False, header=False)
```

    one,1,2,3.0,4,
    two,5,6,,8,world
    three,9,10,11.0,12,foo
    


```python
# 写出一部分的列并指定排序顺序
data.to_csv(sys.stdout, index=False, columns=['a', 'b', 'c'])
```

    a,b,c
    1,2,3.0
    5,6,
    9,10,11.0
    


```python
# Series也有to_csv方法
# 生成DatetimeIndex对象
dates = pd.date_range('1/1/2000', periods=7)
ts = Series(np.arange(7), index=dates)
ts.to_csv('tseries.csv')
!type tseries.csv
```

    2000-01-01,0
    2000-01-02,1
    2000-01-03,2
    2000-01-04,3
    2000-01-05,4
    2000-01-06,5
    2000-01-07,6
    


```python
# 读取Series(无header行，第一列作为索引)
Series.from_csv('tseries.csv', parse_dates=True)
```




    2000-01-01    0
    2000-01-02    1
    2000-01-03    2
    2000-01-04    3
    2000-01-05    4
    2000-01-06    5
    2000-01-07    6
    dtype: int64



### 手工处理分隔符形式


```python
# 含有畸形行的文件
!type ex7.csv
```

    "a","b","c"
    "1","2","3"
    "1","2","3","4"
    


```python
import csv
f = open('ex7.csv')

reader = csv.reader(f)
```


```python
for line in reader:
    print(line)
```

    ['a', 'b', 'c']
    ['1', '2', '3']
    ['1', '2', '3', '4']
    


```python
lines = list(csv.reader(open('ex7.csv')))
header, values = lines[0], lines[1:]
data_dict = {h: v for h, v in zip(header, zip(*values))}
data_dict
```




    {'a': ('1', '1'), 'b': ('2', '2'), 'c': ('3', '3')}




```python
# csv文件的形式有很多，只需定义csv.Dialect的一个子类即可定义出新格式
# lineterminator 行结束符 delimiter 分隔符 quotechar 字符串引用约定 quoting 引用约定
class my_dialect(csv.Dialect):
    lineterminator = '\n'
    delimiter = ';'
    quotechar = '"'
    quoting = csv.QUOTE_MINIMAL
```


```python
# 通过csv.writer手工输出分隔符文件
with open('mydata.csv', 'w') as f:
    writer = csv.writer(f, dialect=my_dialect)
    writer.writerow(('one', 'two', 'three'))
    writer.writerow(('1', '2', '3'))
    writer.writerow(('4', '5', '6'))
    writer.writerow(('7', '8', '9'))
```


```python
!type mydata.csv
```

    one;two;three
    1;2;3
    4;5;6
    7;8;9
    

### JSON数据


```python
obj = """
{"name": "Wes",
 "places_lived": ["United States", "Spain", "Germany"],
 "pet": null,
 "siblings": [{"name": "Scott", "age": 25, "pet": "Zuko"},
              {"name": "Katie", "age": 33, "pet": "Cisco"}]
}
"""
```


```python
import json
result = json.loads(obj)
result
```




    {'name': 'Wes',
     'pet': None,
     'places_lived': ['United States', 'Spain', 'Germany'],
     'siblings': [{'age': 25, 'name': 'Scott', 'pet': 'Zuko'},
      {'age': 33, 'name': 'Katie', 'pet': 'Cisco'}]}




```python
asjson = json.dumps(result)
```


```python
# 将JSON对象转换为DataFrame，并选取字段的子集
siblings = DataFrame(result['siblings'], columns=['name', 'age'])
siblings
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Scott</td>
      <td>25</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Katie</td>
      <td>33</td>
    </tr>
  </tbody>
</table>
</div>



## 二进制数据格式


```python
# pandas对象有一个用于将数据以pickle形式保存到磁盘上的to_pickle方法
frame = pd.read_csv('ex1.csv')
frame.to_pickle('frame_pickle')
```


```python
pd.read_pickle('frame_pickle')
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>hello</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
      <td>12</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>



### 使用HDF5格式


```python
store = pd.HDFStore('mydata.h5')
store['obj1'] = frame
store['obj1_col'] = frame['a']
store
```




    <class 'pandas.io.pytables.HDFStore'>
    File path: mydata.h5
    /obj1                frame        (shape->[3,5])
    /obj1_col            series       (shape->[3])  




```python
store['obj1']
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>hello</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
      <td>12</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
store.close()
os.remove('mydata.h5')
```
