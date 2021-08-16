# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 15:40:49 2021

@author: user
"""
#%% Import module
from sklearn.preprocessing import scale
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import silhouette_score ,silhouette_samples
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#%% pickle data load
import pickle
from Spatial_Analysis import GdfCompare

with open('gdf_6000_won', 'rb') as f :
    a = pickle.load(f)   
with open('gdf_7000_won', 'rb') as f :
    b = pickle.load(f)
with open('법정동_지도_정보_추가', 'rb') as f:
    c = pickle.load(f)

six_bf = a.gdf_before
six_af = a.gdf_after

six_bf_nc = six_bf[['sum_rest', 'sum_school', 'mean_rest']]
six_af_nc = six_af[['sum_rest', 'sum_school', 'mean_rest']]

merge = pd.merge(six_bf, c.loc[:,['ADM_DR_NM','공시지가','초등학생수','중학생수','고등학생수']], how ='inner', on= 'ADM_DR_NM')
merge_2 = pd.merge(six_af, c.loc[:,['ADM_DR_NM','공시지가','초등학생수','중학생수','고등학생수']], how ='inner', on= 'ADM_DR_NM')

six_bf_merge = merge[['sum_rest', 'sum_school','mean_rest','공시지가']]
six_af_merge = merge_2[['sum_rest', 'sum_school','mean_rest', '공시지가']]
#%% Add Official Price

sc = StandardScaler() #정확한 거리를 산정하기 위한 스케일링

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

#%% Before 6000 Policy Visualization

pca = PCA(n_components=2) #2차원 평면에 시각화를 위해 차원축소
pca_transformed = pca.fit_transform(six_bf_merge[['sum_rest','sum_school','mean_rest','공시지가']])

six_bf_merge['pca_x'] = pca_transformed[:,0]
six_bf_merge['pca_y'] = pca_transformed[:,1] 

marker0_ind = six_bf_merge[six_bf_merge['cluster'] == 0].index
marker1_ind = six_bf_merge[six_bf_merge['cluster'] == 1].index
marker2_ind = six_bf_merge[six_bf_merge['cluster'] == 2].index

plt.scatter(x = six_bf_merge.loc[marker0_ind, 'pca_x'], y = six_bf_merge.loc[marker0_ind, 'pca_y'], marker = 'o', label= 'Bad')
plt.scatter(x = six_bf_merge.loc[marker1_ind, 'pca_x'], y = six_bf_merge.loc[marker1_ind, 'pca_y'], marker = 's', label= 'Nomal')
plt.scatter(x = six_bf_merge.loc[marker2_ind, 'pca_x'], y = six_bf_merge.loc[marker2_ind, 'pca_y'], marker = '^', label= 'Good')

plt.legend()
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()

print('공시지가 cor :',stats.pointbiserialr(x = six_bf_merge['공시지가'], y = six_bf_merge['cluster']))
print('식당 총 수 cor :',stats.pointbiserialr(x = six_bf_merge['sum_rest'], y = six_bf_merge['cluster']))
print('식당수 평균 cor :',stats.pointbiserialr(x = six_bf_merge['mean_rest'], y = six_bf_merge['cluster']))
print('학교 총 수 cor :',stats.pointbiserialr(x = six_bf_merge['sum_school'], y = six_bf_merge['cluster']))
#%% After 6000 Policy Visualization

pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(six_af_merge[['sum_rest', 'sum_school', 'mean_rest','공시지가']])

six_af_merge['pca_x'] = pca_transformed[:,0]
six_af_merge['pca_y'] = pca_transformed[:,1] 

marker0_ind = six_af_merge[six_af_merge['cluster'] == 0].index
marker1_ind = six_af_merge[six_af_merge['cluster'] == 1].index
marker2_ind = six_af_merge[six_af_merge['cluster'] == 2].index

plt.scatter(x = six_af_merge.loc[marker0_ind, 'pca_x'], y = six_af_merge.loc[marker0_ind, 'pca_y'], marker = 'o', label= 'Bad')
plt.scatter(x = six_af_merge.loc[marker1_ind, 'pca_x'], y = six_af_merge.loc[marker1_ind, 'pca_y'], marker = 's', label= 'Nomal')
plt.scatter(x = six_af_merge.loc[marker2_ind, 'pca_x'], y = six_af_merge.loc[marker2_ind, 'pca_y'], marker = '^', label= 'Good')

plt.legend()
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()

print('공시지가 cor :',stats.pointbiserialr(x = six_af_merge['공시지가'], y = six_af_merge['cluster']))
print('식당 총 수 cor :',stats.pointbiserialr(x = six_af_merge['sum_rest'], y = six_af_merge['cluster']))
print('식당수 평균 cor :',stats.pointbiserialr(x = six_af_merge['mean_rest'], y = six_af_merge['cluster']))
print('학교 총 수 cor :',stats.pointbiserialr(x = six_af_merge['sum_school'], y = six_af_merge['cluster']))

#%% Silhouette Sample Distribution : Before 6000 Policy 
f, axes = plt.subplots(1, 5, sharex=True, sharey=True)
f.set_size_inches(15, 3)
for i, ax in enumerate(axes):
    sil_samples = silhouette_samples(six_bf_merge, kmeans.labels_,metric='euclidean')
    sil_score = silhouette_score(six_bf_merge, kmeans.labels_,metric='euclidean')
    ax.plot(sorted(sil_samples), color='red',linestyle='dashed', linewidth=2)
    ax.set_title("silhouette_score: {}".format(round(sil_score, 2)))

#%% Silhouette Sample Distribution : After 6000 Policy 
f, axes = plt.subplots(1, 5, sharex=True, sharey=True)
f.set_size_inches(15, 3)
for i, ax in enumerate(axes):
    sil_samples = silhouette_samples(six_af_merge, kmeans_2.labels_,metric='euclidean')
    sil_score = silhouette_score(six_af_merge, kmeans_2.labels_,metric='euclidean')
    ax.plot(sorted(sil_samples), color='red',linestyle='dashed', linewidth=2)
    ax.set_title("silhouette_score: {}".format(round(sil_score, 2)))
#%% Vulnerable Location List
six_bf_idx = six_bf_merge[six_bf_merge['cluster'] == 0 ].index
six_af_idx = six_af_merge[six_af_merge['cluster'] == 0 ].index

vul_location_bf = merge.loc[six_bf_idx, 'ADM_DR_NM']
vul_location_af = merge.loc[six_af_idx, 'ADM_DR_NM']

(len(vul_location_bf) - len(vul_location_af)) /len(vul_location_bf) * 100