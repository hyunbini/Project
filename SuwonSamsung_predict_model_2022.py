#SuwonSamsung_predict_model_2022
#Objective of the model : 다른 것은 없고 2022시즌 강등플레이오프의 승리를 기원하며 제작

#About dataset)
#data_suwon : 2022 수원삼성의 정규시즌 결과를 바탕으로 홈/원정 구분, 정규시즌 순위에 기반한 전력 가중치, 홈 스코어, 원정 스코어로 구성
#test_suwon_1 : 원정경기인 1경기의 상황을 가정한 데이터
#test_suwon_2 : 홈경기인 2경기의 상황을 가정한 데이터

#About ML models to use : 승리 혹은 패배를 구분하는 것이 목적이기에 classification의 기본모델인 Linear Regression과 Binary classification에 적합한 Decision tree, ensemble learning의 RandomForest와 Decision tree에 기반을 둔 Bagging model 사용

#Import
import pandas as pd
import numpy as np
import random
import os
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestClassifier

#Set the Seed
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(42)
#Check the Dirty Data
def check_nan(Filename):
    result = sum(Filename.isna().sum())
    if result != 0:
        print('It is error data.')

#Data Curation - Load the data
train_df = pd.read_csv('C:/Users/yckhb/Desktop/data_suwon.csv')
train_x = train_df.iloc[:,0:2]
train_y = train_df.iloc[:,2:]
test_1 = pd.read_csv('C:/Users/yckhb/Desktop/test_suwon_1.csv')#About first round
test_2 = pd.read_csv('C:/Users/yckhb/Desktop/test_suwon_2.csv')#About second round

#Data Preparation
check_nan(train_x)
check_nan(train_y)

#Data Analysis
LR_original_model = MultiOutputRegressor(LinearRegression()).fit(train_x,train_y)
randomforest_model = RandomForestClassifier(n_estimators=380).fit(train_x,train_y)
decision_tree_model = MultiOutputRegressor(DecisionTreeRegressor()).fit(train_x,train_y)
bagging_model = BaggingRegressor(base_estimator=decision_tree_model,n_estimators=380,verbose=0).fit(train_x,train_y)

#Predict First round score
pred_LR = LR_original_model.predict(test_1)
pred_RF = randomforest_model.predict(test_1)
pred_DT = decision_tree_model.predict(test_1)
pred_BAG = bagging_model.predict(test_1)
predict_LR = pd.DataFrame(pred_LR)
predict_RF = pd.DataFrame(pred_RF)
predict_DT = pd.DataFrame(pred_DT)
predict_BAG = pd.DataFrame(pred_BAG)

#Calculate the First round result
total_result_1 = pd.concat([predict_LR,predict_RF,predict_DT,predict_BAG])
pred_result_score_home_1 = total_result_1.iloc[:,0:1]
pred_result_score_away_1 = total_result_1.iloc[:,1:2]
result_home_1 = np.round(np.average(pred_result_score_home_1))
result_away_1 = np.round(np.average(pred_result_score_away_1))  

#Predict Second round result
pred_LR_2 = LR_original_model.predict(test_2)
pred_RF_2 = randomforest_model.predict(test_2)
pred_DT_2 = decision_tree_model.predict(test_2)
pred_BAG_2 = bagging_model.predict(test_2)
predict_LR_2 = pd.DataFrame(pred_LR_2)
predict_RF_2 = pd.DataFrame(pred_RF_2)
predict_DT_2 = pd.DataFrame(pred_DT_2)
predict_BAG_2 = pd.DataFrame(pred_BAG_2)

#Calculate the Second round result
total_result_2 = pd.concat([predict_LR_2,predict_RF_2,predict_DT_2,predict_BAG_2])
pred_result_score_home_2 = total_result_2.iloc[:,0:1]
pred_result_score_away_2 = total_result_2.iloc[:,1:2]
result_home_2 = np.round(np.average(pred_result_score_home_2))
result_away_2 = np.round(np.average(pred_result_score_away_2))  

#Implementation about First round
if result_away_1 > result_home_1:
    print('In wednesday, First round SuwonSamsung will win, score is Anyang ' + str(result_home_1).rstrip(".0") + ' : ' + 'SuwonSamsung ' + str(result_away_1).rstrip(".0"))
elif result_away_1== result_home_1:
    print('In wednesday, First round SuwonSamsung will draw, score is Anyang ' + str(result_home_1).rstrip(".0") + ' : ' + 'SuwonSamsung ' + str(result_away_1).rstrip(".0"))
else:
    print('In wednesday, First round SuwonSamsung will lose, score is Anyang ' + str(result_home_1).rstrip(".0") + ' : ' + 'SuwonSamsung ' + str(result_away_1).rstrip(".0"))

#Implementation about second round
if result_home_2 > result_away_2:
    print('In saturday, Second round SuwonSamsung will win, score is SuwonSamsung ' + str(result_home_2).rstrip(".0") + ' : ' + 'Anyang ' + str(result_away_2).rstrip(".0"))
elif result_home_2 == result_away_2:
    print('In saturday, Second round SuwonSamsung will draw, score is SuwonSamsung ' + str(result_home_2).rstrip(".0") + ' : ' + 'Anyang ' + str(result_away_2).rstrip(".0"))
else:
    print('In saturday, Second round SuwonSamsung will lose, score is SuwonSamsung ' + str(result_home_2).rstrip(".0") + ' : ' + 'Anyang ' + str(result_away_2).rstrip(".0"))

#Calculate the total score
total_result_suwon = result_away_1+result_home_2
total_result_anyang =result_home_1+result_away_2

#Print result
if total_result_suwon > total_result_anyang:
    print("Suwon Samsung will stay in K League 1, 우리는 절대 포기하지 않아!")
elif total_result_suwon == total_result_anyang:
    print("Oh...God will decide this")
else:
    print("Suwon Samsung will go to K League 2....Bye")