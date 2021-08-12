# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 14:14:38 2021

script for data EDA

author : Daegun Kim
"""

#%% Import modules
import itertools as it
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,\
    mean_squared_log_error
from copy import deepcopy
from sklearn.inspection import permutation_importance
from tqdm import tqdm
from shapely.geometry import Point

#%% Fix random seed for reproduction
def fix_random_seed(seed=42):
    import random
    import numpy as np 
    import os

    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        tf.random.set_seed(seed)
    except:
        pass
    
fix_random_seed()

#%% Custom functions
def str_sim(str1, str2):
    list_str1 = list(str1); list_str2 = list(str2)
    
    for i in range(list_str1.count(' ')):
        list_str1.remove(' ')
        
    for i in range(list_str2.count(' ')):
        list_str2.remove(' ')
        
    sim1 = 0
    for s1 in list_str1:
        if s1 in list_str2:
            sim1 += 1
    sim1 = sim1/len(list_str1)
    
    sim2 = 0
    for s2 in list_str2:
        if s2 in list_str1:
            sim2 += 1
    sim2 = sim2/len(list_str2)
    
    return max(sim1, sim2)


def save_fig(save_dir, figure, title=None, close_fig=True):
    dir_org = os.getcwd()
    os.chdir(save_dir)
    
    figure.savefig('{}.{}'.format(title, fig_file_ext),
                   dpi=high_dpi)
    if close_fig:
        plt.close(figure)
    
    os.chdir(dir_org)
    

def load_map_rest_join(pri_thld=6000):
    """
    Parameters
    ----------
    pri_thld : TYPE, optional
        DESCRIPTION. The default is 6000.

    Returns
    -------
    gdf_emd_jj_frchs : GeoDataFrame
        GeoDataFrame based on eup_myeon_dong of JeonJu merged with child meal
        card franchise restaurants in JeonJu
    gdf_emd_jj : GeoDataFrame
        GeoDataFrame based on eup_myeon_dong of JeonJu merged with every
        restaurant in JeonJu
    """
    # Import packages
    import os
    import geopandas as gpd
    import pyproj
    
    # Load preprocessed restaurants data of Jeonju
    dir_org = os.getcwd()
    os.chdir('data')
    
    df_rest_menu = pd.read_csv('J_rnm.csv', encoding='utf-8', index_col=0)
    
    os.chdir(dir_org)
      
    # Load map shp data
    dir_org = os.getcwd()
    os.chdir('data/SHP파일모음/BUBJUNG')
    
    # gdf_emd = gpd.read_file('Z_SOP_BND_ADM_DONG_PG.shp', encoding='euc-kr')
    gdf_emd = gpd.read_file('BUBJUNG.shp', encoding='utf-8')
    gdf_emd = gdf_emd.rename(columns={'name':'ADM_DR_NM'})
    
    os.chdir(dir_org)    
        
    # Load school data
    dir_org = os.getcwd()
    os.chdir('data/SHP파일모음/학교shp')
    
    gdf_sch = gpd.read_file('school.shp', encoding='euc-kr')
    
    os.chdir(dir_org)    
        
    # Load Jeonju franchise restaurants data
    dir_org = os.getcwd()
    os.chdir('data/SHP파일모음/전주시_가맹점shp')

    df_rest_jj_frchs = pd.read_csv('jeonju_franchise.csv', encoding='euc-kr',
                                   index_col=1).iloc[:, 1:]
    
    os.chdir(dir_org)
    
    # Menu data preprocess by prices of menu
    df_rest_jj = df_rest_menu[df_rest_menu.price <= pri_thld]
    df_rest_jj = df_rest_jj.groupby('res_name').max().reset_index()
    df_rest_jj = df_rest_jj.rename(columns={'lat(y)':'위도', 
                                            'lon(x)':'경도'})
       
    #% Join the price column of df_rest_jj to franchise based on str simility
    df_rest_jj_frchs[['price', 'menu', 'name']] = np.nan
    
    for idx, row in tqdm(df_rest_jj_frchs.iterrows(),
                         total=len(df_rest_jj_frchs.index),
                         desc='Merging Menu and Price'):
        nm_org = df_rest_jj_frchs.at[idx, 'm_name']
        str_sims = df_rest_jj.res_name.apply(lambda x : str_sim('롯데리아', x))
        
        best_sim = np.max(str_sims)
        best_sim_idx = np.argmax(str_sims)
        if best_sim > 0.8:
            df_rest_jj_frchs.loc[idx, 'name'] = \
                df_rest_jj.at[best_sim_idx, 'res_name']   
            df_rest_jj_frchs.loc[idx, 'price'] = \
                df_rest_jj.at[best_sim_idx, 'price']
            df_rest_jj_frchs.loc[idx, 'menu'] = \
            df_rest_jj.at[best_sim_idx, 'menu_name']
                
    #% Assign price manually
    list_res_names = ['마트', 'CU', '롯데리아', '감탄', '본도시락',
                      '본죽', '얌스', '뚜레쥬르', '롯데슈퍼', '이삭',
                      '봉구스밥버거', '신포우리만두']
    list_menu_names = ['라면', '도시락', '핫크리스피버거', '허브탕수육', '흑미밥',
                       '흰죽', '해산물짬뽕밥', '정통크라상', '라면',
                       '햄치즈포테이토', '햄치즈', '해물부추만두']
    list_prices = [6000, 6000, 6000, 5500, 5900, 6000, 6000, 2300, 6000, 4500,
                   5000, 6000]
    
    for res_name, menu, price in zip(list_res_names,
                                     list_menu_names,
                                     list_prices):  
        df_rest_jj_frchs.loc[
            df_rest_jj_frchs.m_name.str.contains(res_name)==True,
            'price'] = price
        df_rest_jj_frchs.loc[
            df_rest_jj_frchs.m_name.str.contains(res_name)==True,
            'menu'] = menu
    
    df_rest_jj_frchs = df_rest_jj_frchs.dropna(subset=['price'])
    
    # Select map data of Jeonju and copy it for franchies restaurants
    gdf_emd_jj = deepcopy(gdf_emd)
    gdf_emd_jj_frchs = deepcopy(gdf_emd)
        
    # Select school data of Jeonju
    gdf_sch_jj = \
        gdf_sch[gdf_sch['소재지도로'].str.contains('전라북도 전주시')]
    
    # Change epsg value of restaurants in Jeonju
    in_proj = pyproj.Proj(init='epsg:4326')
    out_proj = pyproj.Proj(init='epsg:5181')
    out_proj_str = 'epsg:5181'
    
    df_rest_jj['x'], df_rest_jj['y'] = pyproj.transform(
        in_proj, out_proj,
        df_rest_jj['경도'].tolist(), 
        df_rest_jj['위도'].tolist())
    
    gdf_rest_jj = gpd.GeoDataFrame()
    
    gdf_rest_jj = gpd.GeoDataFrame(
            df_rest_jj.loc[:, :], 
            geometry=gpd.points_from_xy(df_rest_jj.loc[:, 'x'],
                                        df_rest_jj.loc[:, 'y']),
            crs=out_proj_str
            )   
    
    # Change epsg value of franchise restaurants in Jeonju
    in_proj = pyproj.Proj(init='epsg:4326')
    out_proj = pyproj.Proj(init='epsg:5181')
    out_proj_str = 'epsg:5181'
    
    df_rest_jj_frchs['x'], df_rest_jj_frchs['y'] = pyproj.transform(
        in_proj, out_proj,
        df_rest_jj_frchs['경도'].tolist(), 
        df_rest_jj_frchs['위도'].tolist())
    
    gdf_rest_jj_frchs = gpd.GeoDataFrame()
    
    gdf_rest_jj_frchs = gpd.GeoDataFrame(
            df_rest_jj_frchs.loc[:, :], 
            geometry=gpd.points_from_xy(df_rest_jj_frchs.loc[:, 'x'],
                                        df_rest_jj_frchs.loc[:, 'y']),
            crs=out_proj_str
            )     
    
    # Calculate summation of all restaurants near school
    gdf_sch_jj['num_rest'] = np.zeros(len(gdf_sch_jj.index))
    
    for idx, row in gdf_sch_jj.iterrows():
        sch_class = row.at['학교급구분']
        if sch_class == '초등학교':
            dist = 400
        elif sch_class == '중학교':
            dist = 700
        else:
            dist = 700
            
        geo_sch = row.at['geometry']
        num_rest = 0
        
        for geo_rest in gdf_rest_jj.geometry:
            if geo_sch.distance(geo_rest) <= dist:
                num_rest += 1
                
        gdf_sch_jj.at[idx, 'num_rest'] = num_rest
            
    # Calculate summation of franchise restaurants near school
    gdf_sch_jj_frchs = deepcopy(gdf_sch_jj)
    gdf_sch_jj_frchs['num_rest'] = np.zeros(len(gdf_sch_jj_frchs.index))
    
    for idx, row in gdf_sch_jj_frchs.iterrows():
        sch_class = row.at['학교급구분']
        if sch_class == '초등학교':
            dist = 400
        elif sch_class == '중학교':
            dist = 700
        else:
            dist = 700
            
        geo_sch = row.at['geometry']
        num_rest = 0
        
        for geo_rest in gdf_rest_jj_frchs.geometry:
            if geo_sch.distance(geo_rest) <= dist:
                num_rest += 1
                
        gdf_sch_jj_frchs.at[idx, 'num_rest'] = num_rest   
        
    # Calculate total restaurants near school in terms of district
    gdf_emd_jj['sum_rest'] = np.zeros(len(gdf_emd_jj.index))
    gdf_emd_jj['sum_school']= np.zeros(len(gdf_emd_jj.index))
    
    for idx, row in gdf_emd_jj.iterrows():
        geo_emd = row.geometry
        num_rest = 0
        num_sch = 0
        
        for _, row_s in gdf_sch_jj.iterrows():
            if geo_emd.intersects(row_s.geometry):
                gdf_emd_jj.at[idx, 'sum_school'] += 1
                gdf_emd_jj.at[idx, 'sum_rest'] += row_s.num_rest
                
    gdf_emd_jj['mean_rest'] = gdf_emd_jj.sum_rest/gdf_emd_jj.sum_school
    gdf_emd_jj.mean_rest = gdf_emd_jj.mean_rest.fillna(np.finfo(float).eps)
    
    # Calculate franchise restaurants near school in terms of district
    gdf_emd_jj_frchs['sum_rest'] = np.zeros(len(gdf_emd_jj_frchs.index))
    gdf_emd_jj_frchs['sum_school']= np.zeros(len(gdf_emd_jj_frchs.index))
    
    for idx, row in gdf_emd_jj_frchs.iterrows():
        geo_emd = row.geometry
        num_rest = 0
        num_sch = 0
        
        for _, row_s in gdf_sch_jj_frchs.iterrows():
            if geo_emd.intersects(row_s.geometry):
                gdf_emd_jj_frchs.at[idx, 'sum_school'] += 1
                gdf_emd_jj_frchs.at[idx, 'sum_rest'] += row_s.num_rest
                
    gdf_emd_jj_frchs['mean_rest'] = \
        gdf_emd_jj_frchs.sum_rest/gdf_emd_jj_frchs.sum_school
    gdf_emd_jj_frchs.mean_rest = \
        gdf_emd_jj_frchs.mean_rest.fillna(np.finfo(float).eps)

    return gdf_emd_jj_frchs, gdf_emd_jj

    
#%% Main script
if __name__ == '__main__':
    
    #% Import modules
    import os
    import time
    import sys
    import matplotlib as mpl
    import geopandas as gpd
    import multiprocessing
    import seaborn as sns
    import math
    import pyproj
    import re
    
    from tqdm import tqdm
    from matplotlib import pyplot as plt
    from shapely.geometry import Polygon, LineString, Point

    tot_run_time_start = time.time()
    
    #%% Make dir for saving figures
    dir_nm_save_fig = 'EDA_plots'
    try:
        os.mkdir(dir_nm_save_fig)
    except FileExistsError:
        pass  
    
    #%% Plot style
    plt.style.use('default')
    # possible options : 
    # seaborn, dark_background, bmh, ggplot, fivethirtyeight... 
    # sns.set(font_scale=0.5)   # Control whole font size
    
    # Apply korean to matplotlib
    from matplotlib import font_manager, rc
    font_name = font_manager.FontProperties(
        fname="c:/Windows/Fonts/malgun.ttf").get_name()
    rc('font', family=font_name)
    
    # 그래프에서 마이너스 폰트 깨지는 문제에 대한 대처
    mpl.rcParams['axes.unicode_minus'] = False
    
    # Matplotlib fontsize change
    SMALL_SIZE = 10
    MEDIUM_SIZE = 15
    LARGE_SIZE = 20
    
    plt.rc('font', size=SMALL_SIZE, weight='bold')
    plt.rc('axes', titlesize=LARGE_SIZE, titleweight='bold')
    plt.rc('axes', labelsize=MEDIUM_SIZE, labelweight='bold')
    plt.rc('axes', titleweight='bold')
    plt.rc('xtick', labelsize=MEDIUM_SIZE)
    plt.rc('ytick', labelsize=MEDIUM_SIZE)
    plt.rc('legend', fontsize=SMALL_SIZE)
    plt.rc('figure', titlesize=LARGE_SIZE)
    
    # Save figure settings
    high_dpi = 150
    fig_file_ext = 'png'
    
    #%% Set multiprocessing process number
    cpu_use = round(multiprocessing.cpu_count()*3/4)
    
    #%% Load preprocessed restaurants data of Jeonju
    dir_org = os.getcwd()
    os.chdir('data')
    
    df_rest_menu = pd.read_csv('J_rnm.csv', encoding='utf-8', index_col=0)
    df_rest_menu_add = pd.read_csv('J_rnm_add.csv')
    
    df_rest_menu = pd.concat([df_rest_menu, df_rest_menu_add], axis=0)
    
    os.chdir(dir_org)
      
    #%% Load map shp data
    dir_org = os.getcwd()
    # os.chdir('data/Z_SOP_BND_ADM_DONG_PG')
    os.chdir('data/SHP파일모음/BUBJUNG')
    
    # gdf_emd = gpd.read_file('Z_SOP_BND_ADM_DONG_PG.shp', encoding='euc-kr')
    gdf_emd = gpd.read_file('BUBJUNG.shp', encoding='utf-8')
    gdf_emd = gdf_emd.rename(columns={'name':'ADM_DR_NM'})
    
    os.chdir(dir_org)    
        
    #%% Load school data
    dir_org = os.getcwd()
    os.chdir('data/SHP파일모음/학교shp')
    
    gdf_sch = gpd.read_file('school.shp', encoding='euc-kr')
    
    os.chdir(dir_org)    
        
    #%% Load Jeonju franchise restaurants data
    dir_org = os.getcwd()
    os.chdir('data/SHP파일모음/전주시_가맹점shp')

    df_rest_jj_frchs = pd.read_csv('jeonju_franchise.csv', encoding='euc-kr',
                                   index_col=1).iloc[:, 1:]
    
    os.chdir(dir_org)   
    
    #%% Menu data preprocess by prices of menu
    df_rest_jj = df_rest_menu[df_rest_menu.price <= 6000]
    df_rest_jj = df_rest_jj.groupby('res_name').max().reset_index()
    df_rest_jj = df_rest_jj.rename(columns={'lat(y)':'위도', 
                                            'lon(x)':'경도'})
    
    set_nm_frchs = set(df_rest_jj_frchs.m_name)
    set_nm_menu = set(df_rest_menu.res_name)
    
    set_corr_menu = set([(x, y) for x, y in zip(df_rest_jj['경도'],
                                                 df_rest_jj['위도'])])
    set_corr_frchs = set([(x, y) for x, y in zip(df_rest_jj_frchs['경도'],
                                                 df_rest_jj_frchs['위도'])]) 
    
    #%% Join the price column of df_rest_jj to franchise based on distance
    '''
    df_rest_jj_frchs[['price', 'menu', 'name']] = np.nan
    
    for idx, row in df_rest_jj_frchs.iterrows():
        lon, lat = row['경도'], row['위도']
        dists = (df_rest_jj['경도'] - lon)**2 + (df_rest_jj['위도'] - lat)**2
        idx_min = np.argmin(dists)
     
        df_rest_jj_frchs.loc[idx, 'name'] = \
            df_rest_jj.at[idx_min, 'res_name']   
     
        if str_sim(df_rest_jj_frchs.loc[idx, 'm_name'],
                   df_rest_jj.at[idx_min, 'res_name']) > 0.8:           
            df_rest_jj_frchs.loc[idx, 'price'] = \
                df_rest_jj.at[idx_min, 'price']
            df_rest_jj_frchs.loc[idx, 'menu'] = \
                df_rest_jj.at[idx_min, 'menu_name']
    '''
    
    #%% Join the price column of df_rest_jj to franchise based on str simility
    df_rest_jj_frchs[['price', 'menu', 'name']] = np.nan
    
    for idx, row in tqdm(df_rest_jj_frchs.iterrows(),
                         total=len(df_rest_jj_frchs.index)):
        nm_org = df_rest_jj_frchs.at[idx, 'm_name']
        str_sims = df_rest_jj.res_name.apply(lambda x : str_sim('롯데리아', x))
        
        best_sim = np.max(str_sims)
        best_sim_idx = np.argmax(str_sims)
        if best_sim > 0.8:
            df_rest_jj_frchs.loc[idx, 'name'] = \
                df_rest_jj.at[best_sim_idx, 'res_name']   
            df_rest_jj_frchs.loc[idx, 'price'] = \
                df_rest_jj.at[best_sim_idx, 'price']
            df_rest_jj_frchs.loc[idx, 'menu'] = \
            df_rest_jj.at[best_sim_idx, 'menu_name']
                
    #%% Assign price manually
    list_res_names = ['마트', 'CU', '롯데리아', '감탄', '본도시락',
                      '본죽', '얌스', '뚜레쥬르', '롯데슈퍼', '이삭',
                      '봉구스밥버거', '신포우리만두']
    list_menu_names = ['라면', '도시락', '핫크리스피버거', '허브탕수육', '흑미밥',
                       '흰죽', '해산물짬뽕밥', '정통크라상', '라면',
                       '햄치즈포테이토', '햄치즈', '해물부추만두']
    list_prices = [6000, 6000, 6000, 5500, 5900, 6000, 6000, 2300, 6000, 4500,
                   5000, 6000]
    
    for res_name, menu, price in zip(list_res_names,
                                     list_menu_names,
                                     list_prices):  
        df_rest_jj_frchs.loc[
            df_rest_jj_frchs.m_name.str.contains(res_name)==True,
            'price'] = price
        df_rest_jj_frchs.loc[
            df_rest_jj_frchs.m_name.str.contains(res_name)==True,
            'menu'] = menu
    
    df_test = df_rest_jj_frchs.loc[:, ['m_name', 'name', 'menu', 'price']]
    df_test2 = pd.DataFrame()
    for r_names in list_res_names:
        df_test2 = pd.concat([
            df_test2,
            df_rest_jj_frchs[df_rest_jj_frchs.m_name.str.contains(r_names)]\
            .loc[:, ['m_name', 'name', 'menu', 'price']]],
            axis=0)
    df_test3 = df_rest_jj_frchs.loc[~df_rest_jj_frchs.index.isin(df_test2.index), 
                                    ['m_name', 'name', 'menu', 'price']]
    
    df_rest_jj_frchs = df_rest_jj_frchs.dropna(subset=['price'])
    
    #%% Select map data of Jeonju and copy it for franchies restaurants
    gdf_emd_jj = deepcopy(gdf_emd)
    gdf_emd_jj_frchs = deepcopy(gdf_emd)
    
    #%% Select school data of Jeonju
    gdf_sch_jj = \
        gdf_sch[gdf_sch['소재지도로'].str.contains('전라북도 전주시')]
    
    #%% Change epsg value of restaurants in Jeonju
    in_proj = pyproj.Proj(init='epsg:4326')
    out_proj = pyproj.Proj(init='epsg:5181')
    out_proj_str = 'epsg:5181'
    
    df_rest_jj['x'], df_rest_jj['y'] = pyproj.transform(
        in_proj, out_proj,
        df_rest_jj['경도'].tolist(), 
        df_rest_jj['위도'].tolist())
    
    gdf_rest_jj = gpd.GeoDataFrame()
    
    gdf_rest_jj = gpd.GeoDataFrame(
            df_rest_jj.loc[:, :], 
            geometry=gpd.points_from_xy(df_rest_jj.loc[:, 'x'],
                                        df_rest_jj.loc[:, 'y']),
            crs=out_proj_str
            )   
    
    #%% Change epsg value of franchise restaurants in Jeonju
    in_proj = pyproj.Proj(init='epsg:4326')
    out_proj = pyproj.Proj(init='epsg:5181')
    out_proj_str = 'epsg:5181'
    
    df_rest_jj_frchs['x'], df_rest_jj_frchs['y'] = pyproj.transform(
        in_proj, out_proj,
        df_rest_jj_frchs['경도'].tolist(), 
        df_rest_jj_frchs['위도'].tolist())
    
    gdf_rest_jj_frchs = gpd.GeoDataFrame()
    
    gdf_rest_jj_frchs = gpd.GeoDataFrame(
            df_rest_jj_frchs.loc[:, :], 
            geometry=gpd.points_from_xy(df_rest_jj_frchs.loc[:, 'x'],
                                        df_rest_jj_frchs.loc[:, 'y']),
            crs=out_proj_str
            )   
       
    #%% Calculate summation of all restaurants near school
    gdf_sch_jj['num_rest'] = np.zeros(len(gdf_sch_jj.index))
    
    for idx, row in tqdm(gdf_sch_jj.iterrows(),
                         total=gdf_sch_jj.shape[0]):
        sch_class = row.at['학교급구분']
        if sch_class == '초등학교':
            dist = 400
        elif sch_class == '중학교':
            dist = 700
        else:
            dist = 1000
            
        geo_sch = row.at['geometry']
        num_rest = 0
        
        for geo_rest in gdf_rest_jj.geometry:
            if geo_sch.distance(geo_rest) <= dist:
                num_rest += 1
                
        gdf_sch_jj.at[idx, 'num_rest'] = num_rest
            
    #%% Calculate summation of franchise restaurants near school
    gdf_sch_jj_frchs = deepcopy(gdf_sch_jj)
    gdf_sch_jj_frchs['num_rest'] = np.zeros(len(gdf_sch_jj_frchs.index))
    
    for idx, row in tqdm(gdf_sch_jj_frchs.iterrows(),
                         total=gdf_sch_jj_frchs.shape[0]):
        sch_class = row.at['학교급구분']
        if sch_class == '초등학교':
            dist = 400
        elif sch_class == '중학교':
            dist = 700
        else:
            dist = 700
            
        geo_sch = row.at['geometry']
        num_rest = 0
        
        for geo_rest in gdf_rest_jj_frchs.geometry:
            if geo_sch.distance(geo_rest) <= dist:
                num_rest += 1
                
        gdf_sch_jj_frchs.at[idx, 'num_rest'] = num_rest        
        
    #%% Calculate total restaurants near school in terms of district
    gdf_emd_jj['sum_rest'] = np.zeros(len(gdf_emd_jj.index))
    gdf_emd_jj['sum_school']= np.zeros(len(gdf_emd_jj.index))
    
    for idx, row in tqdm(gdf_emd_jj.iterrows(), total=gdf_emd_jj.shape[0]):
        geo_emd = row.geometry
        num_rest = 0
        num_sch = 0
        
        for _, row_s in gdf_sch_jj.iterrows():
            if geo_emd.intersects(row_s.geometry):
                gdf_emd_jj.at[idx, 'sum_school'] += 1
                gdf_emd_jj.at[idx, 'sum_rest'] += row_s.num_rest
                
    gdf_emd_jj['mean_rest'] = gdf_emd_jj.sum_rest/gdf_emd_jj.sum_school
    
    #%% Calculate franchise restaurants near school in terms of district
    gdf_emd_jj_frchs['sum_rest'] = np.zeros(len(gdf_emd_jj_frchs.index))
    gdf_emd_jj_frchs['sum_school']= np.zeros(len(gdf_emd_jj_frchs.index))
    
    for idx, row in tqdm(gdf_emd_jj_frchs.iterrows(), 
                         total=gdf_emd_jj_frchs.shape[0]):
        geo_emd = row.geometry
        num_rest = 0
        num_sch = 0
        
        for _, row_s in gdf_sch_jj_frchs.iterrows():
            if geo_emd.intersects(row_s.geometry):
                gdf_emd_jj_frchs.at[idx, 'sum_school'] += 1
                gdf_emd_jj_frchs.at[idx, 'sum_rest'] += row_s.num_rest
                
    gdf_emd_jj_frchs['mean_rest'] = \
        gdf_emd_jj_frchs.sum_rest/gdf_emd_jj_frchs.sum_school
       
    #%% Plot - Data geometry sync test
    fig_test, ax_test = plt.subplots(1, 1, figsize=(12, 12))
    
    # gdf.plot(ax=ax_test)
    gdf_emd_jj.plot(ax=ax_test, alpha=0.5,
                    edgecolor='k', linewidth=2)
    gdf_sch_jj.plot(column='num_rest', ax=ax_test, alpha=0.3, color='red',
                    label='school')
    gdf_rest_jj_frchs.plot(ax=ax_test, alpha=0.3, color='blue',
                           label='franchise rest')
    gdf_rest_jj.plot(ax=ax_test, alpha=0.3, color='orange', 
                     label='total rest')
    ax_test.legend()
    
    #%% Plot - Mapping number of all restaurants considering child behavior
    fig_num_rest_all, ax_num_rest_all = \
        plt.subplots(1, 1, figsize=(12, 12))
        
    c_cmap = 'Blues'
    vmin_tot_r = gdf_emd_jj.mean_rest.min()
    vmax_tot_r = gdf_emd_jj.mean_rest.max();    
    c_norm_tot_r = plt.Normalize(vmin=vmin_tot_r, vmax=vmax_tot_r)
    c_cbar_tot_r = plt.cm.ScalarMappable(norm=c_norm_tot_r, cmap=c_cmap)    
    
    gdf_emd_jj.plot(ax=ax_num_rest_all, alpha=0.5,
                    edgecolor='k', linewidth=2, column='mean_rest',
                    cmap=c_cmap, legend=False, norm=c_norm_tot_r)    
    
    ax_num_rest_all.axis('off')
    ax_num_rest_all.set_title(
        '활동반경을 고려한 접근가능 전체음식점 지수 분포')
    
    ax_num_rest_all_cbar = \
        fig_num_rest_all.colorbar(c_cbar_tot_r, ax=ax_num_rest_all,
                                      fraction=0.018)
    ax_num_rest_all_cbar.set_label(label='접근가능 음식점 지수',
                                       size=25 ,weight='bold')
    
    for idx, r in gdf_emd_jj.iterrows():
        if (idx in gdf_emd_jj.sum_rest.nlargest(5).index) or \
            (idx in gdf_emd_jj.sum_rest.nsmallest(5).index):
            x, y = r.geometry.centroid.x, r.geometry.centroid.y
            plt.text(x, y, r.ADM_DR_NM, fontsize=10)
        
    save_fig(dir_nm_save_fig, fig_num_rest_all, 
             '활동반경을 고려한 접근가능 전체음식점 지수 분포(지역이름 무)',
             close_fig=False)
        
    #%% Plot - Mapping number of franchise rests considering child behavior
    fig_num_rest_frchs, ax_num_rest_frchs = \
        plt.subplots(1, 1, figsize=(12, 12))  
        
    vmin_frchs_r = gdf_emd_jj_frchs.mean_rest.min()
    vmax_frchs_r = gdf_emd_jj_frchs.mean_rest.max();    
    c_norm_frchs_r = plt.Normalize(vmin=vmin_frchs_r, vmax=vmax_frchs_r)
    c_cbar_frchs_r = plt.cm.ScalarMappable(norm=c_norm_frchs_r, cmap=c_cmap)  
    
    gdf_emd_jj_frchs.plot(ax=ax_num_rest_frchs, alpha=0.5,
                          edgecolor='k', linewidth=2, column='mean_rest',
                          cmap=c_cmap, legend=False, norm=c_norm_frchs_r)    
    
    ax_num_rest_frchs.axis('off')
    ax_num_rest_frchs.set_title(
        '활동반경을 고려한 접근가능 가맹음식점 지수 분포')
    
    ax_num_rest_frchs_cbar = \
        fig_num_rest_frchs.colorbar(c_cbar_frchs_r, ax=ax_num_rest_frchs,
                                    fraction=0.018)
    ax_num_rest_frchs_cbar.set_label(label='접근가능 음식점 지수',
                                     size=25 ,weight='bold')
    
    # for idx, r in gdf_emd_jj_frchs.iterrows():
    #     if (idx in gdf_emd_jj_frchs.sum_rest.nlargest(5).index) or \
    #         (idx in gdf_emd_jj_frchs.sum_rest.nsmallest(5).index):
    #         x, y = r.geometry.centroid.x, r.geometry.centroid.y
    #         plt.text(x, y, r.ADM_DR_NM, fontsize=10)
        
    save_fig(dir_nm_save_fig, fig_num_rest_frchs, 
             '활동반경을 고려한 접근가능 가맹음식점 지수 분포(지역이름 무)',
             close_fig=False)    

    #%% Plot - Mapping number of restaurants (total & franchise mix)
    fig_num_rest_mix, ax_num_rest_mix = \
        plt.subplots(1, 2, figsize=(18, 18))  
    
    gdf_emd_jj_frchs.plot(ax=ax_num_rest_mix[0], alpha=0.5,
                          edgecolor='k', linewidth=2, column='mean_rest',
                          cmap=c_cmap, legend=False, norm=c_norm_tot_r)
    gdf_emd_jj_frchs[gdf_emd_jj_frchs['mean_rest'].isnull()]\
    .plot(ax=ax_num_rest_mix[0], alpha=0.5,
                          edgecolor='k', linewidth=2,
                          color='lightgrey', legend=False, norm=c_norm_tot_r,
                          hatch='//')    
    
    ax_num_rest_mix[0].axis('off')
    ax_num_rest_mix[0].set_title('가맹음식점 기준')
    
    gdf_emd_jj.plot(ax=ax_num_rest_mix[1], alpha=0.5,
                    edgecolor='k', linewidth=2, column='mean_rest',
                    cmap=c_cmap, legend=False, norm=c_norm_tot_r)    
    
    ax_num_rest_mix[1].axis('off')
    ax_num_rest_mix[1].set_title('전체음식점 기준')  
    gdf_emd_jj[gdf_emd_jj['mean_rest'].isnull()]\
    .plot(ax=ax_num_rest_mix[1], alpha=0.5,
          edgecolor='k', linewidth=2,
          color='lightgrey', legend=False, norm=c_norm_tot_r,
          hatch='//')
    
    ax_num_rest_mix_cbar = \
        fig_num_rest_mix.colorbar(c_cbar_tot_r, ax=ax_num_rest_mix[1],
                                  fraction=0.05)
    ax_num_rest_mix_cbar.set_label(label='복지카드 사용 편의 지수',
                                     size=25 ,weight='bold')
        
    save_fig(dir_nm_save_fig, fig_num_rest_mix, 
             '활동반경을 고려한 접근가능 가맹음식점 지수 분포(가맹, 전체)',
             close_fig=False)         
    
    
        
        
    
    
    
    
    
    

