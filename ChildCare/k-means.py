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
from Spatial_Analysis import GdfCompare

with open('gdf_6000_won', 'rb') as f :
    a = pickle.load(f)   
with open('gdf_7000_won', 'rb') as f :
    b = pickle.load(f)
with open('법정동_지도_정보_추가', 'rb') as f:
    c = pickle.load(f)
#%% data 1 : 6000원
from sklearn.preprocessing import scale
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import silhouette_score ,silhouette_samples
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
plt.title('6000 Before policy: Clusters Visualization')
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
plt.title('6000 After policy:Clusters Visualization')
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
sev_cluster_result = sev_bf_nc.groupby(['target', 'cluster'])['sum_school'].count()
sev_af_nc['target'] = sev_af.hotspot_class
sev_cluster_result2 = sev_af_nc.groupby(['target', 'cluster'])['sum_school'].count()

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
#%% add 공시지가
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

merge = pd.merge(six_bf, c.loc[:,['ADM_DR_NM','공시지가','초등학생수','중학생수','고등학생수']], how ='inner', on= 'ADM_DR_NM')
merge_2 = pd.merge(six_af, c.loc[:,['ADM_DR_NM','공시지가','초등학생수','중학생수','고등학생수']], how ='inner', on= 'ADM_DR_NM')

six_bf_merge = merge[['sum_rest', 'sum_school', 'mean_rest','공시지가']]
six_af_merge = merge_2[['sum_rest', 'sum_school', 'mean_rest','공시지가']]
sc.fit(six_bf_merge)
sc.fit(six_af_merge)

kmeans = KMeans(n_clusters=3, init = 'k-means++', max_iter= 300, random_state = 40).fit(sc.transform(six_bf_merge))
six_bf_merge['cluster'] = kmeans.labels_ 
kmeans_2 = KMeans(n_clusters=3, init = 'k-means++', max_iter= 300, random_state = 40).fit(sc.transform(six_af_merge))
six_af_merge['cluster'] = kmeans_2.labels_ 

six_bf_merge['target'] = six_bf.hotspot_class
six_merge_result = six_bf_merge.groupby(['target', 'cluster'])['sum_school'].count()
six_af_merge['target'] = six_af.hotspot_class
six_merge_result2 = six_af_merge.groupby(['target', 'cluster'])['sum_school'].count()


six_bf_score = silhouette_score(six_bf_merge, kmeans.labels_,metric='euclidean')
six_af_score = silhouette_score(six_af_merge, kmeans_2.labels_,metric='euclidean')

print(six_merge_result)
print(six_merge_result2)
print('6000 before silhouette score: %.3f' %six_bf_score)
print('6000 after silhouette score: %.3f' %six_af_score)

#%% 시각화
pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(six_bf_merge[['sum_rest', 'sum_school', 'mean_rest','공시지가']])

six_bf_merge['pca_x'] = pca_transformed[:,0]
six_bf_merge['pca_y'] = pca_transformed[:,1] 

marker0_ind = six_bf_merge[six_bf_merge['cluster'] == 0].index
marker1_ind = six_bf_merge[six_bf_merge['cluster'] == 1].index
marker2_ind = six_bf_merge[six_bf_merge['cluster'] == 2].index

plt.scatter(x = six_bf_merge.loc[marker0_ind, 'pca_x'], y = six_bf_merge.loc[marker0_ind, 'pca_y'], marker = 'o', label= 'very vulnerable')
plt.scatter(x = six_bf_merge.loc[marker1_ind, 'pca_x'], y = six_bf_merge.loc[marker1_ind, 'pca_y'], marker = 's', label= 'even')
plt.scatter(x = six_bf_merge.loc[marker2_ind, 'pca_x'], y = six_bf_merge.loc[marker2_ind, 'pca_y'], marker = '^', label= 'good')

plt.legend()
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('Clusters Visualization')
plt.show()

print('공시지가 cor :',stats.pointbiserialr(x = six_bf_merge['공시지가'], y = six_bf_merge['cluster']))
print('식당 총 수 cor :',stats.pointbiserialr(x = six_bf_merge['sum_rest'], y = six_bf_merge['cluster']))
print('식당수 평균 cor :',stats.pointbiserialr(x = six_bf_merge['mean_rest'], y = six_bf_merge['cluster']))
print('학교 총 수 cor :',stats.pointbiserialr(x = six_bf_merge['sum_school'], y = six_bf_merge['cluster']))
print("-----------------------------------------------------------------------------------------------------")
print('공시지가 cor :',stats.pointbiserialr(x = six_bf_merge['공시지가'], y = six_bf_merge['target']))
print('식당 총 수 cor :',stats.pointbiserialr(x = six_bf_merge['sum_rest'], y = six_bf_merge['target']))
print('식당수 평균 cor :',stats.pointbiserialr(x = six_bf_merge['mean_rest'], y = six_bf_merge['target']))
print('학교 총 수 cor :',stats.pointbiserialr(x = six_bf_merge['sum_school'], y = six_bf_merge['target']))
#%% 
pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(six_af_merge[['sum_rest', 'sum_school', 'mean_rest','공시지가']])

six_af_merge['pca_x'] = pca_transformed[:,0]
six_af_merge['pca_y'] = pca_transformed[:,1] 

marker0_ind = six_af_merge[six_af_merge['cluster'] == 0].index
marker1_ind = six_af_merge[six_af_merge['cluster'] == 1].index
marker2_ind = six_af_merge[six_af_merge['cluster'] == 2].index

plt.scatter(x = six_af_merge.loc[marker0_ind, 'pca_x'], y = six_af_merge.loc[marker0_ind, 'pca_y'], marker = 'o', label= 'very vulnerable')
plt.scatter(x = six_af_merge.loc[marker1_ind, 'pca_x'], y = six_af_merge.loc[marker1_ind, 'pca_y'], marker = 's', label= 'even')
plt.scatter(x = six_af_merge.loc[marker2_ind, 'pca_x'], y = six_af_merge.loc[marker2_ind, 'pca_y'], marker = '^', label= 'good')

plt.legend()
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('Clusters Visualization')
plt.show()

print('공시지가 cor :',stats.pointbiserialr(x = six_af_merge['공시지가'], y = six_af_merge['cluster']))
print('식당 총 수 cor :',stats.pointbiserialr(x = six_af_merge['sum_rest'], y = six_af_merge['cluster']))
print('식당수 평균 cor :',stats.pointbiserialr(x = six_af_merge['mean_rest'], y = six_af_merge['cluster']))
print('학교 총 수 cor :',stats.pointbiserialr(x = six_af_merge['sum_school'], y = six_af_merge['cluster']))
print("-----------------------------------------------------------------------------------------------------")
print('공시지가 cor :',stats.pointbiserialr(x = six_af_merge['공시지가'], y = six_af_merge['target']))
print('식당 총 수 cor :',stats.pointbiserialr(x = six_af_merge['sum_rest'], y = six_af_merge['target']))
print('식당수 평균 cor :',stats.pointbiserialr(x = six_af_merge['mean_rest'], y = six_af_merge['target']))
print('학교 총 수 cor :',stats.pointbiserialr(x = six_af_merge['sum_school'], y = six_af_merge['target']))
#%%
merge = pd.merge(sev_bf, c.loc[:,['ADM_DR_NM','공시지가','초등학생수','중학생수','고등학생수']], how ='inner', on= 'ADM_DR_NM')
merge_2 = pd.merge(sev_af, c.loc[:,['ADM_DR_NM','공시지가','초등학생수','중학생수','고등학생수']], how ='inner', on= 'ADM_DR_NM')

sev_bf_merge = merge[['sum_rest', 'sum_school', 'mean_rest','공시지가']]
sev_af_merge = merge_2[['sum_rest', 'sum_school', 'mean_rest','공시지가']]

sc.fit(sev_bf_merge)
sc.fit(sev_af_merge)

kmeans_3 = KMeans(n_clusters=3, init = 'k-means++', max_iter= 300, random_state = 40).fit(sc.transform(sev_bf_merge))
sev_bf_merge['cluster'] = kmeans.labels_ 
kmeans_4 = KMeans(n_clusters=3, init = 'k-means++', max_iter= 300, random_state = 40).fit(sc.transform(sev_af_merge))
sev_af_merge['cluster'] = kmeans_2.labels_ 

sev_bf_merge['target'] = sev_bf.hotspot_class
sev_merge_result = sev_bf_merge.groupby(['target', 'cluster'])['mean_rest'].count()
sev_af_merge['target'] = sev_af.hotspot_class
sev_merge_result2 = sev_af_merge.groupby(['target', 'cluster'])['mean_rest'].count()


sev_bf_score = silhouette_score(sev_bf_merge, kmeans_3.labels_,metric='euclidean')
sev_af_score = silhouette_score(sev_af_merge, kmeans_4.labels_,metric='euclidean')

print(sev_merge_result)
print(sev_merge_result2)
print('7000 before silhouette score: %.3f' %sev_bf_score)
print('7000 after silhouette score: %.3f' %sev_af_score)

#%%
pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(sev_bf_merge)

sev_bf_merge['pca_x'] = pca_transformed[:,0]
sev_bf_merge['pca_y'] = pca_transformed[:,1] 

marker0_ind = sev_bf_merge[sev_bf_merge['cluster'] == 0].index
marker1_ind = sev_bf_merge[sev_bf_merge['cluster'] == 1].index
marker2_ind = sev_bf_merge[sev_bf_merge['cluster'] == 2].index

plt.scatter(x = sev_bf_merge.loc[marker0_ind, 'pca_x'], y = sev_bf_merge.loc[marker0_ind, 'pca_y'], marker = 'o', label= 'very vulnerable')
plt.scatter(x = sev_bf_merge.loc[marker1_ind, 'pca_x'], y = sev_bf_merge.loc[marker1_ind, 'pca_y'], marker = 's', label= 'even')
plt.scatter(x = sev_bf_merge.loc[marker2_ind, 'pca_x'], y = sev_bf_merge.loc[marker2_ind, 'pca_y'], marker = '^', label= 'good')

plt.legend()
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('7000 Befoe policy Clusters Visualization')
plt.show()

print('공시지가 cor :',stats.pointbiserialr(x = sev_bf_merge['공시지가'], y = sev_bf_merge['cluster']))
print('식당 총 수 cor :',stats.pointbiserialr(x = sev_bf_merge['sum_rest'], y = sev_bf_merge['cluster']))
print('식당수 평균 cor :',stats.pointbiserialr(x = sev_bf_merge['mean_rest'], y = sev_bf_merge['cluster']))
print('학교 총 수 cor :',stats.pointbiserialr(x = sev_bf_merge['sum_school'], y = sev_bf_merge['cluster']))
print("-----------------------------------------------------------------------------------------------------")
print('공시지가 cor :',stats.pointbiserialr(x = sev_bf_merge['공시지가'], y = sev_bf_merge['target']))
print('식당 총 수 cor :',stats.pointbiserialr(x = sev_bf_merge['sum_rest'], y = sev_bf_merge['target']))
print('식당수 평균 cor :',stats.pointbiserialr(x = sev_bf_merge['mean_rest'], y = sev_bf_merge['target']))
print('학교 총 수 cor :',stats.pointbiserialr(x = sev_bf_merge['sum_school'], y = sev_bf_merge['target']))

#%%
pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(sev_af_merge)

sev_af_merge['pca_x'] = pca_transformed[:,0]
sev_af_merge['pca_y'] = pca_transformed[:,1] 

marker0_ind = sev_af_merge[sev_af_merge['cluster'] == 0].index
marker1_ind = sev_af_merge[sev_af_merge['cluster'] == 1].index
marker2_ind = sev_af_merge[sev_af_merge['cluster'] == 2].index

plt.scatter(x = sev_af_merge.loc[marker0_ind, 'pca_x'], y = sev_af_merge.loc[marker0_ind, 'pca_y'], marker = 'o', label= 'very vulnerable')
plt.scatter(x = sev_af_merge.loc[marker1_ind, 'pca_x'], y = sev_af_merge.loc[marker1_ind, 'pca_y'], marker = 's', label= 'even')
plt.scatter(x = sev_af_merge.loc[marker2_ind, 'pca_x'], y = sev_af_merge.loc[marker2_ind, 'pca_y'], marker = '^', label= 'good')

plt.legend()
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('7000 After policy Clusters Visualization')
plt.show()

print('공시지가 cor :',stats.pointbiserialr(x = sev_af_merge['공시지가'], y = sev_af_merge['cluster']))
print('식당 총 수 cor :',stats.pointbiserialr(x = sev_af_merge['sum_rest'], y = sev_af_merge['cluster']))
print('식당수 평균 cor :',stats.pointbiserialr(x = sev_af_merge['mean_rest'], y = sev_af_merge['cluster']))
print('학교 총 수 cor :',stats.pointbiserialr(x = sev_af_merge['sum_school'], y = sev_af_merge['cluster']))
print("-----------------------------------------------------------------------------------------------------")
print('공시지가 cor :',stats.pointbiserialr(x = sev_af_merge['공시지가'], y = sev_af_merge['target']))
print('식당 총 수 cor :',stats.pointbiserialr(x = sev_af_merge['sum_rest'], y = sev_af_merge['target']))
print('식당수 평균 cor :',stats.pointbiserialr(x = sev_af_merge['mean_rest'], y = sev_af_merge['target']))
print('학교 총 수 cor :',stats.pointbiserialr(x = sev_af_merge['sum_school'], y = sev_af_merge['target']))

#%% same diff 시각화
six_bf_merge['target'].loc[six_bf_merge['target'] == 1] = 2
six_bf_merge['target'].loc[six_bf_merge['target'] == 0] = 1
six_bf_merge['target'].loc[six_bf_merge['target'] == int(-1)] = 0

pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(six_bf_merge[['sum_rest','sum_school', 'mean_rest','공시지가']])

six_bf_merge['pca_x'] = pca_transformed[:,0]
six_bf_merge['pca_y'] = pca_transformed[:,1] 

diffidx = six_bf_merge[six_bf_merge['cluster'] != six_bf_merge['target']].index
sameidx = six_bf_merge[six_bf_merge['cluster'] == six_bf_merge['target']].index
total = (six_bf_merge['cluster'] != six_bf_merge['target']).sum() + (six_bf_merge['cluster'] == six_bf_merge['target']).sum()
sm = (six_bf_merge['cluster'] == six_bf_merge['target']).sum()

plt.scatter(x = six_bf_merge.loc[sameidx, 'pca_x'], y = six_bf_merge.loc[sameidx, 'pca_y'], marker = 'o', label= 'same')
plt.scatter(x = six_bf_merge.loc[diffidx, 'pca_x'], y = six_bf_merge.loc[diffidx, 'pca_y'], marker = 's', label= 'Different')
plt.legend()
plt.text(3, 30, sm/total)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('6000 Befoe policy same/diff Clusters Visualization')
plt.show()
#%%
six_af_merge['target'].loc[six_af_merge['target'] == 1] = 2
six_af_merge['target'].loc[six_af_merge['target'] == 0] = 1
six_af_merge['target'].loc[six_af_merge['target'] == int(-1)] = 0



pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(six_af_merge[['sum_rest','sum_school', 'mean_rest','공시지가']])

six_af_merge['pca_x'] = pca_transformed[:,0]
six_af_merge['pca_y'] = pca_transformed[:,1] 

diffidx = six_af_merge[six_bf_merge['cluster'] != six_af_merge['target']].index
sameidx = six_af_merge[six_bf_merge['cluster'] == six_af_merge['target']].index
total = (six_af_merge['cluster'] != six_af_merge['target']).sum() + (six_af_merge['cluster'] == six_af_merge['target']).sum()
sm = (six_af_merge['cluster'] == six_af_merge['target']).sum()

plt.scatter(x = six_af_merge.loc[sameidx, 'pca_x'], y = six_af_merge.loc[sameidx, 'pca_y'], marker = 'o', label= 'same')
plt.scatter(x = six_af_merge.loc[diffidx, 'pca_x'], y = six_af_merge.loc[diffidx, 'pca_y'], marker = 's', label= 'Different')
plt.text(3, 30, sm/total)
plt.legend()
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('6000 After policy same/diff Clusters Visualization')
plt.show()
#%%
sev_bf_merge['target'].loc[sev_bf_merge['target'] == 1] = 2
sev_bf_merge['target'].loc[sev_bf_merge['target'] == 0] = 1
sev_bf_merge['target'].loc[sev_bf_merge['target'] == int(-1)] = 0

pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(sev_bf_merge[['sum_rest','sum_school', 'mean_rest','공시지가']])

sev_bf_merge['pca_x'] = pca_transformed[:,0]
sev_bf_merge['pca_y'] = pca_transformed[:,1] 

diffidx = sev_bf_merge[sev_bf_merge['cluster'] != sev_bf_merge['target']].index
sameidx = sev_bf_merge[sev_bf_merge['cluster'] == sev_bf_merge['target']].index
total = (sev_bf_merge['cluster'] != sev_bf_merge['target']).sum() + (sev_bf_merge['cluster'] == sev_bf_merge['target']).sum()
sm = (sev_bf_merge['cluster'] == sev_bf_merge['target']).sum()

plt.scatter(x = sev_bf_merge.loc[sameidx, 'pca_x'], y = sev_bf_merge.loc[sameidx, 'pca_y'], marker = 'o', label= 'same')
plt.scatter(x = sev_bf_merge.loc[diffidx, 'pca_x'], y = sev_bf_merge.loc[diffidx, 'pca_y'], marker = 's', label= 'Different')

plt.text(3, 30, sm/total)
plt.legend()
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('7000 Before policy same/diff Clusters Visualization')
plt.show()
#%%
sev_af_merge['target'].loc[sev_af_merge['target'] == 1] = 2
sev_af_merge['target'].loc[sev_af_merge['target'] == 0] = 1
sev_af_merge['target'].loc[sev_af_merge['target'] == int(-1)] = 0

pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(sev_af_merge[['sum_rest','sum_school', 'mean_rest','공시지가']])

sev_af_merge['pca_x'] = pca_transformed[:,0]
sev_af_merge['pca_y'] = pca_transformed[:,1] 

diffidx = sev_af_merge[sev_af_merge['cluster'] != sev_af_merge['target']].index
sameidx = sev_af_merge[sev_af_merge['cluster'] == sev_af_merge['target']].index
total = (sev_af_merge['cluster'] != sev_af_merge['target']).sum() + (sev_af_merge['cluster'] == sev_af_merge['target']).sum()
sm = (sev_af_merge['cluster'] == sev_af_merge['target']).sum()

plt.scatter(x = sev_af_merge.loc[sameidx, 'pca_x'], y = sev_af_merge.loc[sameidx, 'pca_y'], marker = 'o', label= 'same')
plt.scatter(x = sev_af_merge.loc[diffidx, 'pca_x'], y = sev_af_merge.loc[diffidx, 'pca_y'], marker = 's', label= 'Different')

plt.text(3, 30, sm/total)
plt.legend()
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('7000 After policy same/diff Clusters Visualization')
plt.show()

#%% 공시지가 없는 모델 시각화
six_bf_nc['target'].loc[six_bf_nc['target'] == 1] = 2
six_bf_nc['target'].loc[six_bf_nc['target'] == 0] = 1
six_bf_nc['target'].loc[six_bf_nc['target'] == int(-1)] = 0

pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(six_bf_nc[['sum_rest','sum_school', 'mean_rest']])

six_bf_nc['pca_x'] = pca_transformed[:,0]
six_bf_nc['pca_y'] = pca_transformed[:,1] 

diffidx = six_bf_nc[six_bf_nc['cluster'] != six_bf_nc['target']].index
sameidx = six_bf_nc[six_bf_merge['cluster'] == six_bf_nc['target']].index
total = (six_bf_nc['cluster'] != six_bf_nc['target']).sum() + (six_bf_nc['cluster'] == six_bf_nc['target']).sum()
sm = (six_bf_nc['cluster'] == six_bf_nc['target']).sum()

plt.scatter(x = six_bf_nc.loc[sameidx, 'pca_x'], y = six_bf_nc.loc[sameidx, 'pca_y'], marker = 'o', label= 'same')
plt.scatter(x = six_bf_nc.loc[diffidx, 'pca_x'], y = six_bf_nc.loc[diffidx, 'pca_y'], marker = 's', label= 'Different')
plt.legend()
plt.text(30,5, sm/total)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('Clusters Visualization')
plt.show()
sev_bf_merge['target'].loc[sev_bf_merge['target'] == 1] = 2
sev_bf_merge['target'].loc[sev_bf_merge['target'] == 0] = 1
sev_bf_merge['target'].loc[sev_bf_merge['target'] == int(-1)] = 0
#%%
from sklearn.metrics import silhouette_samples

silhouette_samples(six_bf_merge, kmeans.labels_,metric='euclidean')

f, axes = plt.subplots(1, 5, sharex=True, sharey=True)
f.set_size_inches(15, 3)
for i, ax in enumerate(axes):
    sil_samples = silhouette_samples(six_bf_merge, kmeans.labels_,metric='euclidean')
    sil_score = silhouette_score(six_bf_merge, kmeans.labels_,metric='euclidean')
    ax.plot(sorted(sil_samples), color='red',linestyle='dashed', linewidth=2)
    ax.set_title("silhouette_score: {}".format(round(sil_score, 2)))
    
#%%
f, axes = plt.subplots(1, 5, sharex=True, sharey=True)
f.set_size_inches(15, 3)
for i, ax in enumerate(axes):
    sil_samples = silhouette_samples(sev_bf_merge, kmeans_3.labels_,metric='euclidean')
    sil_score = silhouette_score(sev_bf_merge, kmeans_3.labels_,metric='euclidean')
    ax.plot(sorted(sil_samples), color='red',linestyle='dashed', linewidth=2)
    ax.set_title("silhouette_score: {}".format(round(sil_score, 2)))
#%%