# 刪除過去內容
#reset
 

# 匯入所需的模組
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error,r2_score


# 載入波士頓房價的數據集
boston = load_boston()
# 切割出資料集的Ｘ＆ｙ
X = boston.data
y = boston.target
# 切分訓練&測試資料
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=8765) #random_state 種子值


############################### 設定參數搜索範圍 ##############################

# 設定cv個數
cv_num=5
#設定Lasso & Ridge & RandomForest&XGBoost的參數搜索範圍(只搜索指定的参数)
lasso_alphas=[0.1, 1.0, 10.0]
ridge_alphas=[0.1, 1.0, 10.0]

forest_param = {'n_estimators': [30, 50, 100],
                'max_features': [2, 3, 4],
                'max_depth': [3, 4, 5], 
                'min_samples_leaf': [20, 30, 40, 50], 
                'n_jobs': [-1],
                'random_state': [8765]}

xgb_param = {
 'learning_rate': [0.05, 0.1],           # 學習率，也就是梯度下降法中的步長。太小的话，训练速度太慢，而且容易陷入局部最優点。通常是0.0001到0.1之间 
 'n_estimators' : [30, 50, 100],         # 樹的個數。并非越多越好，通常是50到1000之间。
 'max_depth' : [3, 4, 5],                # 每棵樹的最大深度。太小会欠擬合，太大过擬合。正常值是3到10。
 'min_child_weight' : [3, 4, 5],         # 決定最小葉子節點樣本權重和。當它的值較大時，可以避免模型學習到局部的特殊樣本。但如果這個值過高，會導致欠擬合。
 'subsample' : [0.7, 0.8],               # 随機抽樣的比例
 'colsample_bytree' : [0.4, 0.6, 0.8],   # 訓練每個樹時用的特徵的數量。1表示使用全部特徵，0.5表示使用一半的特徵
 'gamma' : [0],                          # 在節點分裂時，只有在分裂後損失函數的值下降了，才會分裂這個節點。
 'reg_alpha' : [1],                      # L1 正則化項的權重係數，越大模型越保守，用来防止过拟合。一般是0到1之间。(和Lasso regression類似)。
 'reg_lambda' : [3, 4],                  # L2 正則化項的權重係數，越大模型越保守，用来防止过拟合。一般是0到1之间。(和Ridge regression類似)。
 'objective' : ['reg:linear'],           # 定義學習任務及相應的學習目標
 'nthread' : [-1],                       # cpu 線程数
  #'scale_pos_weight' : [1],             # 各類樣本十分不平衡時，把這個參數設置為一個正數，可以使算法更快收斂。典型值是sum(negative cases) / sum(positive cases)
 'seed' : [8765]}                        # 隨機種子


############################### Lasso regression ##############################

# 使用Lasso的交叉验证来选择参数
lasso = linear_model.LassoCV(alphas=lasso_alphas,cv=cv_num)
lasso.fit(X_train,y_train)
#print("最优alpha:",lasso.alpha_)

# 再利用最佳參數訓練模型
blasso = linear_model.Lasso(alpha = lasso.alpha_)
blasso.fit(X_train,y_train) #训练模型
y_pred = blasso.predict(X_test) #预测模型

###############################################################################


############################### Ridge regression ##############################

# 使用Ridge的交叉验证来选择参数
ridge = linear_model.RidgeCV(alphas=ridge_alphas,cv=cv_num)
ridge.fit(X_train,y_train)
#print("最优alpha:",ridge.alpha_)

# 再利用最佳參數訓練模型
bridge = linear_model.Ridge(alpha = ridge.alpha_)
bridge.fit(X_train,y_train) #训练模型
y_pred2 = bridge.predict(X_test) #预测模型

###############################################################################


############################### RandomForest ##############################

#分类器使用 RandomForest
rfr = RandomForestRegressor()

# 使用RandomizedSearch的交叉验证来选择参数
randomized = RandomizedSearchCV(rfr,forest_param,iid=True,scoring='r2',cv=cv_num,n_jobs=-1)
randomized.fit(X_train, y_train)
#print('最佳參數',randomized.best_estimator_)

# 再利用最佳參數訓練模型
brfr = randomized.best_estimator_
#训练模型
brfr.fit(X_train,y_train)
#预测模型
y_pred3 = brfr.predict(X_test)

###########################################################################


############################### XGBoost ##############################

#分类器使用 XGBoost
xgbr = xgb.XGBRegressor()

# 使用RandomizedSearch的交叉验证来选择参数
randomized = RandomizedSearchCV(xgbr,xgb_param,iid=True,scoring='r2',cv=cv_num,n_jobs=-1)
randomized.fit(X_train, y_train)
#print('最佳參數',randomized.best_estimator_)

# 再利用最佳參數訓練模型
bxgbr = randomized.best_estimator_
#训练模型
bxgbr.fit(X_train,y_train)
#预测模型
y_pred4 = bxgbr.predict(X_test)

###########################################################################


# 針對實際值和xgb lasso ridge forest模型的預測值reshape
y_test=pd.DataFrame(y_test).values.reshape(-1,)
y_pred=pd.DataFrame(y_pred).values.reshape(-1,)
y_pred2=pd.DataFrame(y_pred2).values.reshape(-1,)
y_pred3=pd.DataFrame(y_pred3).values.reshape(-1,)
y_pred4=pd.DataFrame(y_pred4).values.reshape(-1,)

#　計算各模型的rmse
lasso_rmse=mean_squared_error(y_pred,y_test)**0.5
ridge_rmse=mean_squared_error(y_pred2,y_test)**0.5
forest_rmse=mean_squared_error(y_pred3,y_test)**0.5
xgb_rmse=mean_squared_error(y_pred4,y_test)**0.5

# print出各模型的rmse
print('lasso_rmse=',lasso_rmse)
print('ridge_rmse=',ridge_rmse)
print('forest_rmse=',forest_rmse)
print('xgb_rmse=',xgb_rmse)