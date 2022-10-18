# Libarary Imports

# Basic Library
from random import randrange
import random
import numpy as np
import pandas as pd
from collections import defaultdict
# Sci-kit
from sklearn.model_selection import train_test_split #Use to separate datasets for train data and test data
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# Plot
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

class MultiColLabelEncoder:
  def __init__(self):
    self.encoder_dict = defaultdict(LabelEncoder)
#Make each LabelEncoder class instance and do fit_transform function
  def fit_transform(self, X: pd.DataFrame, columns: list):
    if not isinstance(columns, list):
      columns = [columns]#Use at inverse_transform() to change original label later
    output = X.copy()
    output[columns] = X[columns].apply(lambda x: self.encoder_dict[x.name].fit_transform(x)) #x of lambda x becomes a series structure consisting of each column, x.You can find out the name of the column by name
    return output
  def inverse_transform(self, X: pd.DataFrame, columns: list):
    if not isinstance(columns, list):
      columns = [columns]
    #Previously not encoded
    if not all(key in self.encoder_dict for key in columns):
      raise KeyError(f'At lesast one of {columns} is not encoded before')
    output = X.copy()
    try:
      output[columns] = X[columns].apply(lambda x: self.encoder_dict[x.name].inverse_transform(x))
    #If fit_transform() function is not work
    except ValueError:
      print(f'Need assignment when do "fit_transform" function')
      raise
    return output

#The Function which can handle preprocessing, modeling, evaluation in order
def Preprocessing(df: pd.DataFrame, **kwargs):
    method_na = kwargs.get('na')
    if method_na == 'dropany': #If Nan is handled by dropany
        df.dropna(how='any', inplace=True) #call the how parameters 'any'
    elif method_na == 'dropall':
        df.dropna(how='all', inplace=True)
    elif method_na == 'ffill':
        df.fillna(method='ffill', inplace=True)
    elif method_na == 'bfill':
        df.dropna(method='bfill', inplace=True)

    #Change the categorical data to numerical data
    if kwargs.get('mapping') is not None:
        for key, di in kwargs.get('mapping').items():
            df[key] = df[key].map(di)

    if kwargs.get('ctg_encoder') is not None:
        # extract categorical features list
        categorical_features = list(df.select_dtypes(include=['object']).columns)

        # label other encoding of categorical features
        mcle = kwargs.get('ctg_encoder')
        df = mcle.fit_transform(df, columns=categorical_features)

    df_list = [df]
    #Make scaled dataset
    if (kwargs.get('scaler')) is not None:
        for scalar_model in kwargs.get('scaler'):
            scalar_model.fit(df)
            scaled = scalar_model.transform(df)
            df_list.append(pd.DataFrame(scaled, columns=df.columns))
            
    return df_list
#Do the Hierarchical clustering and evaluation
def Hierarchical_Evaluate(linkage_list, data, data_name):
  scores = {}
  fig, axes = plt.subplots(nrows=len(linkage_list), ncols=len(data), figsize=(16, 35))
  for i in range(len(linkage_list)):
    for j in range(len(data)):
    #Make dendrogram
      hierarchical_single = linkage(data[j], method=linkage_list[i])
      dn = dendrogram(hierarchical_single, ax=axes[i][j])
    #Calculate silhouette score and do agglomerative cluster 
      text = '{}({})'.format(linkage_list[i], data_name[j])
      axes[i][j].title.set_text(text)
      scores[text] = silhouette_score(data[j], AgglomerativeClustering(n_clusters=5, linkage=linkage_list[i]).fit_predict(data[j]))
  return scores
#Do the k-means clustering and evaluation
def kMeans_Evaluate(n_clusters, data):
    sum_of_squared_distance = []
    score_list = []
    for n_cluster in n_clusters:
        kmeans = KMeans(n_clusters=n_cluster)
        kmeans.fit(data)
        sum_of_squared_distance.append(kmeans.inertia_)
        score_list.append(silhouette_score(data, kmeans.predict(data)))
    return sum_of_squared_distance, score_list

survey_df = pd.read_csv('dataset/young_people_survey/responses.csv')
#Spotify music dataset
music_df = pd.read_csv('/content/drive/MyDrive/GitHub/2022_termproject_ds/dataset/spotify_multigenre_data/merged_music_data.csv', encoding='cp949')
#Show features
survey_df.columns
#Show features
music_df.columns
# remove useless features
temp_df_1 = music_df.iloc[:,:16]#Use dataset which index until "tempo"
temp_df_2 = music_df.iloc[:,20:22]#Use dataset which Index "duration_ms" and "time_signature"
music_df = pd.concat([temp_df_1,temp_df_2],axis=1) # The dataset which excepts 'id','url','track_href', 'analysis_url column that not used
music_df
#Show the young people survey dataset
survey_df
#Check if there is null values in dataset
music_df.isnull().sum()
#Check if there is null values in dataset
survey_df.isnull().sum()
#Drop null values
music_df.dropna(how='any', inplace=True)
#Check again if there is null values in dataset
music_df.isnull().sum()
#Drop null values
survey_df.dropna(how='any', inplace=True)
#Check again if there is null values in dataset
survey_df.isnull().sum()
#Check the index which have '[]' in genre index
idx = music_df[music_df['Genres']=='[]'].index
#drop row that has empty genre value from music dataframe
music_df.drop(idx, inplace=True)
music_df
#Show young people survey dataset statistics
survey_df.describe()
#Show sportify music dataset statistics
music_df.describe()
#Make a boxplot to find out outliers in music popularity
boxplot = music_df.boxplot(column=['Popularity'])
boxplot.plot()
plt.show()
#drop music data that has outlined popularity
idx = music_df[music_df['Popularity'] >= 150].index #To set outlier which popularity is more than 150
music_df.drop(idx, inplace=True) #Drop the outlier data
idx = music_df[music_df['Popularity'] <= 0].index #To set outlier which popularity is lower than 0
music_df.drop(idx, inplace=True) #Drop the outlier data
music_df.describe() #Check the dataset
#check for correlation among personality features on survey data
survey_column_list = list(survey_df.columns)
corr_features = survey_column_list[19:76]
survey_corr = survey_df[corr_features].corr()
#To make a cluster map
sns.clustermap(survey_corr, annot=True, cmap="RdYlBu_r")
#Check the correlation to make heatmap
survey_corr = survey_corr.apply(lambda x: round(x,2))
survey_corr.style.background_gradient(cmap='viridis')

#To change the categorical data 'punctuality' to numerical data
punctuality_mapping = {'i am often running late':0, 'i am always on time':1, 'i am often early': 2}
survey_df['Punctuality'] = survey_df['Punctuality'].map(punctuality_mapping)
#To change the categorical data 'lying' to numerical data
lying_mapping = {'never':0, 'only to avoid hurting someone': 1, 'sometimes': 2, 'everytime it suits me':3}
survey_df['Lying'] = survey_df['Lying'].map(lying_mapping)
#To change the categorical data 'internet' to numerical data
internet_mapping = {'less than an hour a day': 0, 'few hours a day':1, 'most of the day':2}
survey_df['Internet usage'] = survey_df['Internet usage'].map(internet_mapping)
#To change the categorical data 'education' to numerical data
education_mapping = {'currently a primary school pupil':0,'primary school': 1, 'secondary school':2, 'college/bachelor degree':3, 'masters degree':4,'doctorate degree': 5}
survey_df['Education'] = survey_df['Education'].map(education_mapping)
#Check the data
survey_df[['Punctuality', 'Lying', 'Internet usage', 'Education']]
#Extract categorical features list
categorical_features = list(survey_df.select_dtypes(include=['object']).columns)
categorical_features
#Make the  multicollabel encoder using Label encoder
mcle = MultiColLabelEncoder() 
#Using multicolLabel encoder 
encoded_survey_df = mcle.fit_transform(survey_df, columns=categorical_features)
encoded_survey_df[categorical_features]
#Standard Scaling
std_scaler = StandardScaler()
#Use standard scaling 
std_scaler.fit(encoded_survey_df)
std_scaled_survey = std_scaler.transform(encoded_survey_df)
std_scaled_survey_df = pd.DataFrame(std_scaled_survey, columns=encoded_survey_df.columns)
#Check dataset
std_scaled_survey_df
#MinMax Scaling
minmax_scaler = MinMaxScaler()
minmax_scaler.fit(encoded_survey_df)
#Use MinMax Scaling
minmax_scaled_survey = minmax_scaler.transform(encoded_survey_df)
minmax_scaled_survey_df = pd.DataFrame(minmax_scaled_survey, columns=encoded_survey_df.columns)
#Check the dataset
minmax_scaled_survey_df

# df, df2 can be selected from encoded_survey_df, std_scaled_survey_df, and minmax_scaled_survey_df
df = encoded_survey_df
df2 = std_scaled_survey_df
#Check the gender ratio
print('---------- gender ratio ----------')
print(df['Gender'].value_counts() / df.shape[0])
print('-------------------------------')
plt.figure(figsize=(12, 5))
#Get the result that male data is little bit more
sns.countplot(df['Gender'], palette='Set2')
plt.xticks([0, 1], ['Male', 'Female'])
#Check the Distribution over age
plt.figure(figsize=(12, 5))
#Get the result that distribution over age takes the form of a slightly longer right tail
sns.kdeplot(df['Age'])
print('mean   >', df['Age'].mean())
print('median >', df['Age'].median())
#Sample features for visualization. x_value, y_value are characteristic and z_value is Genre.
#Can change the x_value and y_value
x_value = 'Mood swings'
y_value = 'Number of siblings'
z_value = 'Rock'
# According to the change in x_value, the change in y_value was confirmed by line plot
plt.figure(figsize=(12, 5))
# To show the lineplot
sns.lineplot(x= x_value, y= y_value, data=df)
# In order, single linkage, complete linkage, average linkage, and ward linkage were applied.
linkage_list = ['single', 'complete', 'average', 'ward']
# Data without scaling on the left and data with scaling on the right
data, data_name = [df, df2], ["Original", "Scaled"]
# Proximity matrix is created based on distance, and clusters are grown accordingly, but the method of measuring the distance between clusters has changed, and the results of clustering have also been significantly different.
result = Hierarchical_Evaluate(linkage_list, data, data_name)
for key, value in result.items():
  print('{}: {}'.format(key, value))
plt.show()
# Best method of hierarchical in eye inspection
method_hier_sil = 'ward'
# Best method of hierarchical in silhuette score
method_hier_sil = 'average'
# Among the results of the dendrogram above, the ward seems to have become relatively uniform
agg_clustering = AgglomerativeClustering(n_clusters=5, linkage='ward') #Using agglomerative clustering which linkage is 'ward' and Get the n_clusters becomes 5 doing clustering repeatly
labels = agg_clustering.fit_predict(df)#To predict the clustering result
# Hierarchical clustering (method='ward', n_cluster=5) results with data before scaling
plt.figure(figsize=(20, 6))
#Select and specify number
plt.subplot(131)
sns.scatterplot(x=x_value, y=y_value, data=df, hue=labels, palette='Set2')
#Select and specify number
plt.subplot(132)
sns.scatterplot(x=x_value, y=y_value, data=df, hue=labels, palette='Set2')
#Select and specify number
plt.subplot(133)
sns.scatterplot(x=x_value, y=y_value, data=df, hue=labels, palette='Set2')
# 3D visualization of the clustering results above
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d') 
x = df[x_value]
y = df[y_value]
z = df[z_value]
ax.scatter(x, y, z, c=labels, s=20, alpha=0.5, cmap='rainbow')
# Hierarchical clustering result conducted with scaling data
agg_clustering2 = AgglomerativeClustering(n_clusters=5, linkage='ward')
labels2 = agg_clustering2.fit_predict(df2)
plt.figure(figsize=(20, 6))
plt.subplot(131)
sns.scatterplot(x=x_value, y=y_value, data=df2, hue=labels2, palette='Set2')
plt.subplot(132)
sns.scatterplot(x=x_value, y=y_value, data=df2, hue=labels2, palette='Set2')
plt.subplot(133)
sns.scatterplot(x=x_value, y=y_value, data=df2, hue=labels2, palette='Set2')
# 3D visualization of the clustering results after scaling dataset
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d') 
x = df2[x_value]
y = df2[y_value]
z = df2[z_value]
ax.scatter(x, y, z, c = labels2, s= 20, alpha=0.5, cmap='rainbow')
#Make the list about n_clusters
n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# Using Data which is not scaled and do k-means evaluation
sum_of_squared_distance, scores = kMeans_Evaluate(n_clusters, df)
plt.figure(1 , figsize = (12, 6))
plt.plot(n_clusters , sum_of_squared_distance , 'o')
plt.plot(n_clusters , sum_of_squared_distance , '-' , alpha = 0.5)
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
print('Silhuette Score:')
#Repeat the k-means evaluation until finish the n_clusters list
for n, score in zip(n_clusters, scores):
  print('{} : {}'.format(n, score))
# Using Data which is scaled and do k-means evaluation
sum_of_squared_distance, scores = kMeans_Evaluate(n_clusters, df2)
plt.figure(1 , figsize = (12, 6))
plt.plot(n_clusters , sum_of_squared_distance , 'o')
plt.plot(n_clusters , sum_of_squared_distance , '-' , alpha = 0.5)
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
print('Silhuette Score:[')
#Repeat the k-means evaluation until finish the n_clusters list
for n, score in zip(n_clusters, scores):
  print('{} : {}'.format(n, score))
# Best k value in elbow method (inertia_)
k_kmeans_elbow = 4
# Best k value in silhuette score
k_means_sil = 2
# Learning and visualizing unscaled data with kmeans (k=4)
kmeans = KMeans(n_clusters=4)
kmeans.fit(df)
plt.figure(figsize=(20, 6))
#Set the same number used before
plt.subplot(131)
sns.scatterplot(x= x_value, y= y_value, data= df, hue=kmeans.labels_,palette='coolwarm')
plt.scatter(kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:, 3], c='red', alpha=0.5, s=150)
#Set the same number used before
plt.subplot(132)
sns.scatterplot(x= x_value, y= y_value, data=df, hue=kmeans.labels_, palette='coolwarm')
plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 3], c='red', alpha=0.5, s=150)
#Set the same number used before
plt.subplot(133)
sns.scatterplot(x= x_value, y= y_value, data=df, hue=kmeans.labels_, palette='coolwarm')
plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], c='red', alpha=0.5, s=150)
# 3D visualization of the clustering results used not scaled data
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d') 
x = df[x_value]
y = df[y_value]
z = df[z_value]
ax.scatter(x, y, z, c = kmeans.labels_, s= 20, alpha=0.5, cmap='rainbow')
# Learning and visualizing scaled data with kmeans (k=4)
kmeans2 = KMeans(n_clusters=4)
kmeans2.fit(df2)
plt.figure(figsize=(20, 6))
#Set the same number used before
plt.subplot(131)
sns.scatterplot(x= x_value, y= y_value, data=df2, hue=kmeans2.labels_,palette='coolwarm')
plt.scatter(kmeans2.cluster_centers_[:, 2], kmeans2.cluster_centers_[:, 3], c='red', alpha=0.5, s=150)
#Set the same number used before
plt.subplot(132)
sns.scatterplot(x= x_value, y= y_value, data=df2, hue=kmeans2.labels_, palette='coolwarm')
plt.scatter(kmeans2.cluster_centers_[:, 1], kmeans2.cluster_centers_[:, 3], c='red', alpha=0.5, s=150)
#Set the same number used before
plt.subplot(133)
sns.scatterplot(x= x_value, y= y_value, data=df2, hue=kmeans2.labels_, palette='coolwarm')
plt.scatter(kmeans2.cluster_centers_[:, 1], kmeans2.cluster_centers_[:, 2], c='red', alpha=0.5, s=150)
# 3D visualization of the clustering results used scaled data
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d') 
x = df[x_value]
y = df[y_value]
z = df[z_value]
ax.scatter(x, y, z, c = kmeans.labels_, s= 20, alpha=0.5, cmap='rainbow')
#Predict Result k-means clustering which is scaled data 
label0_string = df2[kmeans.labels_ == 0].mean().sort_values(ascending=False).head(10)
label1_string = df2[kmeans.labels_ == 1].mean().sort_values(ascending=False).head(10)
label2_string = df2[kmeans.labels_ == 2].mean().sort_values(ascending=False).head(10)
label3_string = df2[kmeans.labels_ == 3].mean().sort_values(ascending=False).head(10)