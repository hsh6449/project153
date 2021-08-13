# -*- coding: utf-8 -*-
"""
Created on Mon Aug  12 09:51:59 2021

script for k-means clustering (for anlaysis)

author : sanha
"""
#%% clustering 1 : K-means Example
from sklearn.preprocessing import scale
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

iris = load_iris()
irisDF = pd.DataFrame(data=iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
#%% ex kmeans moeld
kmeans = KMeans(n_clusters=3, init = 'k-means++', max_iter= 300, random_state = 40).fit(irisDF)
irisDF['cluster'] = kmeans.labels_

#%% ex compare
irisDF['target'] = iris.target
iris_result = irisDF.groupby(['target', 'cluster'])['sepal_length'].count()

#%% ex visualization
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(iris.data)

irisDF['pca_x'] = pca_transformed[:,0]
irisDF['pca_y'] = pca_transformed[:,1] 

marker0_ind = irisDF[irisDF['cluster'] == 0].index
marker1_ind = irisDF[irisDF['cluster'] == 1].index
marker2_ind = irisDF[irisDF['cluster'] == 2].index

plt.scatter(x = irisDF.loc[marker0_ind, 'pca_x'], y = irisDF.loc[marker0_ind, 'pca_y'], marker = 'o')
plt.scatter(x = irisDF.loc[marker1_ind, 'pca_x'], y = irisDF.loc[marker1_ind, 'pca_y'], marker = 's')
plt.scatter(x = irisDF.loc[marker2_ind, 'pca_x'], y = irisDF.loc[marker2_ind, 'pca_y'], marker = '^')

plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('Clusters Visualization')
plt.show

#%% pickle data load
import pickle
with open('gdf_6000_won', 'rb') as f :
    a = pickle.load(f)
with open('gdf_7000_won', 'rb') as f :
    b = pickle.load(f2)
#%% data 1 : 6000원
from sklearn.preprocessing import scale
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
%matplotlib inline

six_bf = a.gdf_before
six_af = a.gdf_after

six_bf_nc = six_bf[['sum_rest', 'sum_school', 'mean_rest']]
six_af_nc = six_af[['sum_rest', 'sum_school', 'mean_rest']]

kmeans = KMeans(n_clusters=3, init = 'k-means++', max_iter= 300, random_state = 40).fit(six_bf_nc)
six_bf_nc['cluster'] = kmeans.labels_ 
kmeans_2 = KMeans(n_clusters=3, init = 'k-means++', max_iter= 300, random_state = 40).fit(six_af_nc)
six_af_nc['cluster'] = kmeans_2.labels_ 
#%% 6000 k_means result

six_bf_nc['target'] = six_bf.hotspot_class
six_cluster_result = six_bf_nc.groupby(['target', 'cluster'])['mean_rest'].count()
six_af_nc['target'] = six_af.hotspot_class
six_cluster_result2 = six_af_nc.groupby(['target', 'cluster'])['mean_rest'].count()

print(six_cluster_result)
print(six_cluster_result2)
#%% visualization six_bf_nc
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(six_bf_nc)

six_bf_nc['pca_x'] = pca_transformed[:,0]
six_bf_nc['pca_y'] = pca_transformed[:,1] 

marker0_ind = six_bf_nc[six_bf_nc['cluster'] == 0].index
marker1_ind = six_bf_nc[six_bf_nc['cluster'] == 1].index
marker2_ind = six_bf_nc[six_bf_nc['cluster'] == 2].index

plt.scatter(x = six_bf_nc.loc[marker0_ind, 'pca_x'], y = six_bf_nc.loc[marker0_ind, 'pca_y'], marker = 'o', label= 'very vulnerable')
plt.scatter(x = six_bf_nc.loc[marker1_ind, 'pca_x'], y = six_bf_nc.loc[marker1_ind, 'pca_y'], marker = 's', label= 'even')
plt.scatter(x = six_bf_nc.loc[marker2_ind, 'pca_x'], y = six_bf_nc.loc[marker2_ind, 'pca_y'], marker = '^', label= 'good')

plt.legend()
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('Clusters Visualization')
plt.show()

#%% visualization six_af_nc
pca = PCA(n_components=2)
pca2_transformed = pca.fit_transform(six_af_nc)

six_af_nc['pca_x'] = pca2_transformed[:,0]
six_af_nc['pca_y'] = pca2_transformed[:,1] 

marker0_ind = six_af_nc[six_af_nc['cluster'] == 0].index
marker1_ind = six_af_nc[six_af_nc['cluster'] == 1].index
marker2_ind = six_af_nc[six_af_nc['cluster'] == 2].index

plt.scatter(x = six_bf_nc.loc[marker0_ind, 'pca_x'], y = six_bf_nc.loc[marker0_ind, 'pca_y'], marker = 'o', label= 'very vulnerable')
plt.scatter(x = six_bf_nc.loc[marker1_ind, 'pca_x'], y = six_bf_nc.loc[marker1_ind, 'pca_y'], marker = 's', label= 'even')
plt.scatter(x = six_bf_nc.loc[marker2_ind, 'pca_x'], y = six_bf_nc.loc[marker2_ind, 'pca_y'], marker = '^', label= 'good')

plt.legend()
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('Clusters Visualization')
plt.show()

#%% 7000 k-means result
sev_bf = b.gdf_before
sev_af = b.gdf_after

sev_bf_nc = sev_bf[['sum_rest', 'sum_school', 'mean_rest']]
sev_af_nc = sev_af[['sum_rest', 'sum_school', 'mean_rest']]

kmeans_3 = KMeans(n_clusters=3, init = 'k-means++', max_iter= 300, random_state = 40).fit(sev_bf_nc)
sev_bf_nc['cluster'] = kmeans_3.labels_ 
kmeans_4 = KMeans(n_clusters=3, init = 'k-means++', max_iter= 300, random_state = 40).fit(sev_af_nc)
sev_af_nc['cluster'] = kmeans_4.labels_ 

sev_bf_nc['target'] = sev_bf.hotspot_class
sev_cluster_result = sev_bf_nc.groupby(['target', 'cluster'])['mean_rest'].count()
sev_af_nc['target'] = sev_af.hotspot_class
sev_cluster_result2 = sev_af_nc.groupby(['target', 'cluster'])['mean_rest'].count()

print(sev_cluster_result)
print(sev_cluster_result2)
#%% visualization  sev_bf_nc
pca = PCA(n_components=2)
pca3_transformed = pca.fit_transform(sev_bf_nc)

sev_bf_nc['pca_x'] = pca3_transformed[:,0]
sev_bf_nc['pca_y'] = pca3_transformed[:,1] 

marker0_ind = sev_bf_nc[sev_bf_nc['cluster'] == 0].index
marker1_ind = sev_bf_nc[sev_bf_nc['cluster'] == 1].index
marker2_ind = sev_bf_nc[sev_bf_nc['cluster'] == 2].index

plt.scatter(x = sev_bf_nc.loc[marker0_ind, 'pca_x'], y = sev_bf_nc.loc[marker0_ind, 'pca_y'], marker = 'o', label= 'very vulnerable')
plt.scatter(x = sev_bf_nc.loc[marker1_ind, 'pca_x'], y = sev_bf_nc.loc[marker1_ind, 'pca_y'], marker = 's', label= 'even')
plt.scatter(x = sev_bf_nc.loc[marker2_ind, 'pca_x'], y = sev_bf_nc.loc[marker2_ind, 'pca_y'], marker = '^', label= 'good')

plt.legend()
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('Clusters Visualization')
plt.show()
#%% visualization  sev_af_nc
pca = PCA(n_components=2)
pca4_transformed = pca.fit_transform(sev_af_nc)

sev_af_nc['pca_x'] = pca4_transformed[:,0]
sev_af_nc['pca_y'] = pca4_transformed[:,1] 

marker0_ind = sev_af_nc[sev_af_nc['cluster'] == 0].index
marker1_ind = sev_af_nc[sev_af_nc['cluster'] == 1].index
marker2_ind = sev_af_nc[sev_af_nc['cluster'] == 2].index

plt.scatter(x = sev_af_nc.loc[marker0_ind, 'pca_x'], y = sev_af_nc.loc[marker0_ind, 'pca_y'], marker = 'o', label= 'very vulnerable')
plt.scatter(x = sev_af_nc.loc[marker1_ind, 'pca_x'], y = sev_af_nc.loc[marker1_ind, 'pca_y'], marker = 's', label= 'even')
plt.scatter(x = sev_af_nc.loc[marker2_ind, 'pca_x'], y = sev_af_nc.loc[marker2_ind, 'pca_y'], marker = '^', label= 'good')

plt.legend()
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('Clusters Visualization')
plt.show()
#%%