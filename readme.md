“波士顿房价预测”是关于机器学习的一个经典问题，在解决这个问题的方法上，我们选用的第一种模型是决策树模型，下面就是我对于该模型的实际应用的步骤及相关算法：
1.首先，我们我们需要将已知的房价数据导入，其中用来衡量房价的几个主要的参数包括有‘PM’、'LSTAT'、‘PTRATIO’,分别代表了房屋房间的平均数量、低收入业主的占比和老师和学生群体的人数占比。
数据导入的代码如下所示：

# Import libraries necessary for this project
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit

# Import supplementary visualizations code visuals.py
import visuals as vs

# Pretty display for notebooks
%matplotlib inline

# Load the Boston housing dataset
data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)
    
# Success
print("Boston housing dataset has {} data points with {} variables each.".format(*data.shape))
data[:500:80]

根据上面导入的489个数据，我们可以的得到：

          RM     LSTAT    PTRATIO      MEDV
0        7.287    4.08     12.6       699300
80       5.787    10.24    16.1       407400
160      5.928    15.76    17.8       384300
240      6.619    7.22     19.0       501900
320      6.425    12.03    20.2       338100
400      6.649    23.24    20.2       291900
480      5.693    17.19    21.2       340200

通过对这些数据进行处理，得出相对应的最大值、最小值、平均值和标准差，代码如下：
a = np.array([prices])
# TODO: Minimum price of the data
minimum_price =(a.min()) 

# TODO: Maximum price of the data
maximum_price =(a.max()) 

# TODO: Mean price of the data
mean_price = (a.mean())

# TODO: Median price of the data
median_price = (np.median(a))

# TODO: Standard deviation of prices of the data
std_price = (a.std())

# Show the calculated statistics
print("Statistics for Boston housing dataset:\n")
print("Minimum price: ${}".format(minimum_price)) 
print("Maximum price: ${}".format(maximum_price))
print("Mean price: ${}".format(mean_price))
print("Median price ${}".format(median_price))
print("Standard deviation of prices: ${}".format(std_price))

输出数据：
Statistics for Boston housing dataset:

Minimum price: $105000
Maximum price: $1024800
Mean price: $454342.9447852761
Median price $438900.0
Standard deviation of prices: $165171.13154429474

以上的数据能够大致的给我们提供一个参考，就针对这几个影响因素的对于房价的影响来看，‘RM’和‘MEDV’成正比例关系，房间数越多，则房价售价越高，‘LSTAT’对于‘MEDV’的影响则不然，比例越高，则售价越低。此外，‘PTRATIO’则更多的呈现一个正态分布，随着比例达到一定的数值之后，房价会下降，在此之前则呈现上升趋势。


一个模型的质量好坏需要对他进行训练和测试，我们可以通过决定系数R2来对于模型性能进行衡量。R2的值范围从0到1，它捕获目标变量的预测值和实际值之间的平方相关性的百分比。R2为0的模型并不比总是预测目标变量均值的模型好，而R2为1的模型则完美地预测了目标变量。当使用这个模型时，0到1之间的任何值表示目标变量中有多少百分比可以被特征解释。一个模型的R2也可以是负的，这表明该模型比总是预测平均值的模型糟糕得多。

下面是对数据点性能分数进行分配：
# TODO: Import 'r2_score'
from sklearn.metrics import r2_score

def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """
    
    # TODO: Calculate the performance score between 'y_true' and 'y_predict'
    score = r2_score(y_true, y_predict)
    
    # Return the score
    return score

# Calculate the performance of this model
score = performance_metric([3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3])
print("Model has a coefficient of determination, R^2, of {:.3f}.".format(score))


通过上述的程序，我们可以计算出R2的值为0.923.
可以发现这个值已经极度的贴近1，这说明我们的这个模型的预测优异。


接下来我们需要对数据集进行划分，分为训练集和测试集，数据集划分需要两个数据集都能够很好的反应数据的趋势和状况，一般来说，训练集约占8成，测试集占2成。将数据集划分为训练集和测试集的好处:既可以进行训练，又可以进行测试，不受干扰，训练模型可以有效验证。使用部分训练集进行测试的缺点:该模型是基于训练集的，使用训练集进行测试肯定会得到较好的结果，不能判断训练模型的优劣。具体代码如下：

# TODO: Import 'train_test_split'
from sklearn.model_selection import train_test_split
# TODO: Shuffle and split the data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2,random_state=1)

# Success
print("Training and testing split was successful.")

下面我们要生成不同深度下的决策树图形，在这里我们分别选择了最大深度为1,3,6，10四种情况，来分别分析测试集和训练集的学习曲线，阴影区域表示其不确定性。挑选最大深度3这组曲线进行分析可以发现，随着训练数据的增加，训练集曲线的得分趋于稳定，在0.8左右，验证集的得分也趋于接近0.8。可以看出，训练集数据得分趋于稳定，增加训练数据并不能提高模型的性能。
当模型的最大深度为1时，模型的预测偏差较大，因为R2得分较低，表明拟合不足。模型的最大深度为10时,有一个大的方差模型的预测,因为成绩的训练集与测试集的图像中,红色和绿色线之间的距离随着深度增加而延长。

在项目的最后一部分中，我们使用来自fit_model的优化模型构建一个模型，并对目标的特性集进行预测。在将所有内容结合在一起后，使用决策树算法训练一个模型。为了确保生成了一个优化的模型，可以使用网格搜索技术对模型进行训练，以优化决策树的'max_depth'参数。'max_depth'参数可以被认为是决策树算法在做出预测之前允许询问关于数据的多少个问题。决策树是被称为监督学习算法的一类算法的一部分。下面是应用fit_model的具体算法：
# TODO: Import 'make_scorer', 'DecisionTreeRegressor', and 'GridSearchCV'
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer

def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data
    cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 0)

    # TODO: Create a decision tree regressor object
    regressor = DecisionTreeRegressor()

    # TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params =  {'max_depth':[1,2,3,4,5,6,7,8,9,10]}

    # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(score_func = performance_metric)

    # TODO: Create the grid search cv object --> GridSearchCV()
    # Make sure to include the right parameters in the object:
    # (estimator, param_grid, scoring, cv) which have values 'regressor', 'params', 'scoring_fnc', and 'cv_sets' respectively.
    grid =  GridSearchCV(estimator = regressor,
                        param_grid = params,
                        scoring = scoring_fnc,
                        cv = cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_


    通过以上步骤，我们就已经成功的对一个模型进行了训练，后面当我们进行训练时，可以根据输入的数据，做出预测。下面是根据这个模型做的一个售价预测：
    # Produce a matrix for client data
client_data = [[5, 17, 15], # Client 1
               [4, 32, 22], # Client 2
               [8, 3, 12]]  # Client 3

# Show predictions
for i, price in enumerate(reg.predict(client_data)):
    print("Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price))

    最后输出的结果如下：
Predicted selling price for Client 1's home: $421,369.57
Predicted selling price for Client 2's home: $227,141.86
Predicted selling price for Client 3's home: $933,975.00

预测的结果和实际房价差距很小，预测精度很高。