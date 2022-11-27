#The Function which can handle preprocessing, modeling, evaluation in order
#Made at 2022_1_DS

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