#The function that handle the classification which Decision tree, logistic regression, SVM
#Made at 2022_2_ML

def ML(classification_algortihm_name, dataset_non_target, dataset_target, num, l):
    result_save_1 = [] #To save accuracy score about First alogrithm
    result_save_2 = [] #To save accuracy score about second algorithm
    result_save_3 = [] #To save accuracy score about third algorithm
    result_save_4 = [] #To save accuracy score about fourth algorithm
    result_total = {} #To save best alogrtihm name and accuracy mean score 

    kfold = KFold(n_splits=num[l])
    m=0 #To change the algortihm
    for classification_algortihm_name[m] in classification_algorithm_list: #Check all algorithms in the algorithm list
        for train_index, test_index in kfold.split(dataset_non_target): #Repeat as KFOLD
            train_features, test_features = dataset_non_target[train_index], dataset_non_target[test_index] #Split the non_target data
            train_target, test_target = dataset_target[train_index], dataset_target[test_index] #Split the target data
            if classification_algortihm_name[m] == 'decision_tree(entropy)':            
                #Make decision tree using entropy
                model_tree_entropy = DecisionTreeClassifier(criterion='entropy')
                model_tree_entropy_result = model_tree_entropy.fit(train_features,train_target)
                pred_1 = model_tree_entropy_result.predict(test_features)
                accuracy = np.round(accuracy_score(test_target, pred_1), 4) 
                result_save_1.append(accuracy)
            elif classification_algortihm_name[m] == 'decision_tree(gini_index)':  
                #Make decision tree using gini index        
                model_tree_gini = DecisionTreeClassifier(criterion='gini')
                model_tree_gini_result = model_tree_entropy.fit(train_features,train_target)
                pred_2=model_tree_gini_result.predict(test_features)
                accuracy = np.round(accuracy_score(test_target, pred_2), 4)
                result_save_2.append(accuracy)
            elif classification_algortihm_name[m] == 'logistic_regression':
                #Make logisticRegression model            
                model_logistic = LogisticRegression()
                model_logistic_result = model_logistic.fit(train_features,train_target)
                pred_3=model_logistic_result.predict(test_features)
                accuracy = np.round(accuracy_score(test_target, pred_3), 4)
                result_save_3.append(accuracy)
            elif classification_algortihm_name[m] == 'support_vector_machine': 
                #Make svm model           
                model_svm = svm.SVC(kernel='linear')
                model_svm_result = model_svm.fit(train_features,train_target)
                pred_4=model_svm_result.predict(test_features)
                accuracy = np.round(accuracy_score(test_target, pred_4), 4) 
                result_save_4.append(accuracy)
        m = m+1 #repeated it as much as KFOLD, so added it one to change the algorithm
    result_total['decision_tree_entropy']=np.round((sum(result_save_1)/len(result_save_1)),4) #Calculate the mean and add at dictionary
    result_total['decision_tree_gini']=np.round((sum(result_save_2)/len(result_save_2)),4) #Calculate the mean and add at dictionary
    result_total['logistic']=np.round((sum(result_save_3)/len(result_save_3)),4) #Calculate the mean and add at dictionary
    result_total['SVM']=np.round((sum(result_save_4)/len(result_save_4)),4) #Calculate the mean and add at dictionary
    result_total = sorted(result_total.items(), key = lambda x:x[1],reverse=True) #Sort in descending order
    return result_total[0] #Return the largest mean score