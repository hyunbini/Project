
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
train_df = pd.read_csv('C:/Users/yckhb/Desktop/data_seoul.csv')
train_x = train_df.iloc[:,0:2]
train_y = train_df.iloc[:,2:]
test_1 = pd.read_csv('C:/Users/yckhb/Desktop/test_seoul_1.csv')#About first round
test_2 = pd.read_csv('C:/Users/yckhb/Desktop/test_seoul_2.csv')#About second round

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
print(total_result_1)
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
print(total_result_2)
pred_result_score_home_2 = total_result_2.iloc[:,0:1]
pred_result_score_away_2 = total_result_2.iloc[:,1:2]
result_home_2 = np.round(np.average(pred_result_score_home_2))
result_away_2 = np.round(np.average(pred_result_score_away_2))  

#Implementation about First round
if result_home_1 > result_away_1:
    print('First round Seoul will win, score is Seoul ' + str(result_home_1).rstrip(".0") + ' : ' + 'MaeBuk ' + str(result_away_1).rstrip(".0"))
elif result_away_1== result_home_1:
    print('First round Seoul will draw, score is Seoul ' + str(result_home_1).rstrip(".0") + ' : ' + 'MaeBuk ' + str(result_away_1).rstrip(".0"))
else:
    print('First round Seoul will lose, score is Seoul ' + str(result_home_1)+ ' : ' + 'MaeBuk ' + str(result_away_1).rstrip(".0"))

#Implementation about second round
if result_home_2 > result_away_2:
    print('Second round Naebuk will win, score is Maebuk ' + str(result_home_2).rstrip(".0") + ' : ' + 'seoul ' + str(result_away_2).rstrip(".0"))
elif result_home_2 == result_away_2:
    print('Second round MaeBuk will draw, score is Maebuk ' + str(result_home_2).rstrip(".0") + ' : ' + 'seoul ' + str(result_away_2).rstrip(".0"))
else:
    print('Second round Maebuk will lose, score is Maebuk ' + str(result_home_2).rstrip(".0") + ' : ' + 'seoul ' + str(result_away_2).rstrip(".0"))

#Calculate the total score
total_result_seoul = result_home_1+result_away_2
total_result_maebuk =result_away_1+result_home_2

#Print result
if total_result_seoul > total_result_maebuk:
    print("Seoul will go to AFC!")
elif total_result_seoul == total_result_maebuk:
    print("Oh...God will decide this")
else:
    print("Incheon will go to AFC.")