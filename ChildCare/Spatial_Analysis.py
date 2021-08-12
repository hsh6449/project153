# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 09:51:59 2021

script for spatial analysis (Spatial Autocorrelation and Hotspot Analysis)

author : Daegun Kim
"""

#%% Import packages
import esda
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
import libpysal as lps
import numpy as np
import matplotlib.pyplot as plt
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
def save_fig(save_dir, figure, title=None, close_fig=True):
    dir_org = os.getcwd()
    os.chdir(save_dir)
    
    figure.savefig('{}.{}'.format(title, fig_file_ext),
                   dpi=high_dpi)
    if close_fig:
        plt.close(figure)
    
    os.chdir(dir_org)
   
    
def get_all_attr(price):
    gdf_base1, gdf_base2 = load_map_rest_join(pri_thld=price)
    gdf_set = GdfCompare(gdf_base1, gdf_base2)
    gdf_set.run_local_g()
    gdf_set.run_stats_diff()
    gdf_set.run_global_moran()
    
    return gdf_set
    
#%% Custom classes
class GdfCompare:
    def __init__(self, gdf1, gdf2, min_th_dist=100):
        self.gdf_before = gdf1
        self.gdf_after = gdf2
        self.alternative = 'less'
        self.t_test = False
        self.wrst = False
        self.min_thld_distance = min_th_dist
        
    def run_stats_diff(self):
        sw_jj_frchs = shapiro(self.gdf_before.mean_rest)
        sw_jj = shapiro(self.gdf_after.mean_rest)
        
        if sw_jj_frchs.pvalue < 0.05 and sw_jj.pvalue < 0.05:
            t_test = ttest_rel(self.gdf_before.mean_rest,
                               self.gdf_after.mean_rest,
                               alternative=self.alternative)
            # H0 : Mean values are identical
            # H1 : Mean of a is less than the mean of b
            print('Paired T-Test was done')
            self.t_test = True
            self.t_test_pval = t_test.pvalue
        else:
            wrst = wilcoxon(self.gdf_before.mean_rest, 
                            self.gdf_after.mean_rest,
                            alternative=self.alternative,
                            zero_method='wilcox')
            # H0 : Two related paired samples come from the same distribution
            # H1 : Median of a is less than the median of b
            print('Wilcoxon Signed Rank Test was done')
            self.wrst = True
            self.wrst_pval = wrst.pvalue
            
            
    def run_global_moran(self):
        wq_before =  lps.weights.Queen.from_dataframe(self.gdf_before)
        wq_before.transform = 'r'

        wq_after =  lps.weights.Queen.from_dataframe(self.gdf_after)
        wq_after.transform = 'r'

        y_before = self.gdf_before.mean_rest
        
        y_after = self.gdf_after.mean_rest
        
        mi_before = esda.Moran(y_before, wq_before)
        mi_after = esda.Moran(y_after, wq_after)
        
        self.glo_m_before = [mi_before.I, mi_before.p_sim]
        self.glo_m_after = [mi_after.I, mi_after.p_sim]
    
    
    def run_local_g(self):
        # Get Getis Ord Local
        cent_jj = self.gdf_before.geometry.centroid
        xys = pd.DataFrame({'X': cent_jj.x, 'Y': cent_jj.y})
        min_wt_jj_thld = libpysal.weights.util.min_threshold_distance(xys)\
            *self.min_thld_distance
        wt_jj = libpysal.weights.DistanceBand(xys, threshold=min_wt_jj_thld)
        
        lg_gdf_before = esda.getisord.G_Local(self.gdf_before.mean_rest, 
                                            wt_jj, transform='r')
        self.lg_gdf_before = lg_gdf_before
        
        lg_gdf_after = esda.getisord.G_Local(self.gdf_after.mean_rest,
                                      wt_jj, transform='r')
        self.lg_gdf_after = lg_gdf_after
        
        self.gdf_before['lg_p_sim'] = lg_jj_frchs.p_sim
        self.gdf_before['lg_Zs'] = lg_jj_frchs.Zs
        
        self.gdf_after['lg_p_sim'] = lg_jj.p_sim
        self.gdf_after['lg_Zs'] = lg_jj.Zs
        
        # Classify the hotspot classes 
        self.gdf_before['hotspot_class'] = np.nan
        sig_gdf_before = self.gdf_before.lg_p_sim < 0.05
        
        self.gdf_before.loc[sig_gdf_before==False, 'hotspot_class'] = 0
        self.gdf_before.loc[(sig_gdf_before==True) & 
                            (self.gdf_before.lg_Zs > 0), 
                            'hotspot_class'] = 1
        self.gdf_before.loc[(sig_gdf_before==True) & 
                            (self.gdf_before.lg_Zs < 0), 
                            'hotspot_class'] = -1
        
        sig_gdf_after = self.gdf_after.lg_p_sim < 0.05
        
        self.gdf_after.loc[sig_gdf_after==False, 'hotspot_class'] = 0
        self.gdf_after.loc[(sig_jj==True) & 
                           (self.gdf_after.lg_Zs > 0), 
                           'hotspot_class'] = 1
        self.gdf_after.loc[(sig_jj==True) & 
                           (self.gdf_after.lg_Zs < 0), 
                           'hotspot_class'] = -1
            
        
    def plot_compare(self):
        fig, ax = plt.subplots(1, 2, figsize=(18, 9))
        
        sig_gdf_before = self.gdf_before.lg_p_sim < 0.05
        ns_jj_frchs = self.gdf_before[sig_gdf_before==False].plot(
            ax=ax[0], color='lightgrey', edgecolor='k', linewidth=0.1)
        hh_jj_frchs = self.gdf_before[(sig_gdf_before==True) & 
                                    (self.gdf_before.lg_Zs > 0)].plot(
            ax=ax[0], color='red', edgecolor='k', linewidth=0.1)
        ll_jj_frchs = gdf_emd_jj_frchs[(sig_gdf_before==True) & 
                                       (self.gdf_before.lg_Zs < 0)].plot(
           ax=ax[0], color='blue', edgecolor='k', linewidth=0.1)                                        
        ax[0].axis(False)
        
        # self.gdf_before[self.gdf_before.hotspot_class==]
        
        sig_gdf_after = self.gdf_after.lg_p_sim < 0.05
        ns_jj = self.gdf_after[sig_jj==False].plot(
            ax=ax[1], color='lightgrey', edgecolor='k', linewidth=0.1)
        hh_jj = self.gdf_after[(sig_jj==True) & 
                               (self.gdf_after.lg_Zs > 0)].plot(
            ax=ax[1], color='red', edgecolor='k', linewidth=0.1)
        ll_jj = self.gdf_after[(sig_jj==True) & 
                               (self.gdf_after.lg_Zs < 0)].plot(
            ax=ax[1], color='blue', edgecolor='k', linewidth=0.1)
        ax[1].axis(False)
        
        return fig
    
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
    import libpysal
    import pickle
    
    from tqdm import tqdm
    from matplotlib import pyplot as plt
    from shapely.geometry import Polygon, LineString, Point
    from scipy.stats import shapiro, ttest_rel, wilcoxon
    
    from eda import load_map_rest_join

    tot_run_time_start = time.time()
    
    #%% Make dir for saving figures
    dir_nm_save_fig = 'Spatial_Analysis_plot'
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

    #%% Load JeonJu emd geodataframe which is merged with restaurants
    gdf_emd_jj_frchs, gdf_emd_jj = load_map_rest_join()

    #%% Getis Ord Local
    cent_jj = gdf_emd_jj.geometry.centroid
    xys = pd.DataFrame({'X': cent_jj.x, 'Y': cent_jj.y})
    min_wt_jj_thld = libpysal.weights.util.min_threshold_distance(xys)*1.1
    wt_jj = libpysal.weights.DistanceBand(xys, threshold=min_wt_jj_thld)
    
    #%% Calculate local G statistic
    lg_jj_frchs = esda.getisord.G_Local(gdf_emd_jj_frchs.mean_rest, 
                                        wt_jj, transform='r')
    lg_jj = esda.getisord.G_Local(gdf_emd_jj.mean_rest,
                                  wt_jj, transform='r')
    
    gdf_emd_jj_frchs['lg_p_sim'] = lg_jj_frchs.p_sim
    gdf_emd_jj_frchs['lg_Zs'] = lg_jj_frchs.Zs
    
    gdf_emd_jj['lg_p_sim'] = lg_jj.p_sim
    gdf_emd_jj['lg_Zs'] = lg_jj.Zs

    #%% Plot - 
    fig, ax = plt.subplots(1, 2, figsize=(18, 9))
    
    sig_jj_frchs = gdf_emd_jj_frchs.lg_p_sim < 0.05
    
    ns_jj_frchs = gdf_emd_jj_frchs[sig_jj_frchs==False].plot(
        ax=ax[0], color='lightgrey', edgecolor='k', linewidth=0.1)
    
    hh_jj_frchs = gdf_emd_jj_frchs[(sig_jj_frchs==True) & 
                                (gdf_emd_jj_frchs.lg_Zs > 0)].plot(
        ax=ax[0], color='red', edgecolor='k', linewidth=0.1)
 
    ll_jj_frchs = gdf_emd_jj_frchs[(sig_jj_frchs==True) & 
                                (gdf_emd_jj_frchs.lg_Zs < 0)].plot(
        ax=ax[0], color='blue', edgecolor='k', linewidth=0.1)
                                    
    ax[0].axis(False)
    
    sig_jj = gdf_emd_jj.lg_p_sim < 0.05
    
    ns_jj = gdf_emd_jj[sig_jj==False].plot(
        ax=ax[1], color='lightgrey', edgecolor='k', linewidth=0.1)
    
    hh_jj = gdf_emd_jj[(sig_jj==True) & 
                                (gdf_emd_jj.lg_Zs > 0)].plot(
        ax=ax[1], color='red', edgecolor='k', linewidth=0.1)
 
    ll_jj = gdf_emd_jj[(sig_jj==True) & 
                                (gdf_emd_jj.lg_Zs < 0)].plot(
        ax=ax[1], color='blue', edgecolor='k', linewidth=0.1)
                                    
    ax[1].axis(False)
    
    # ========================================================================    
    #%% Paired T-test of Wilcoxon signed rank test for mean restaurants col
    # ========================================================================
    
    # Shapiro - Wilk test for normality
    sw_jj_frchs = shapiro(gdf_emd_jj_frchs.mean_rest)
    sw_jj = shapiro(gdf_emd_jj.mean_rest)
    
    print('Franchise Shapiro - Wilk Test p: {}\n'.format(sw_jj_frchs.pvalue))
    print('All restaurant Shapiro - Wilk Test p: {}\n'.format(sw_jj.pvalue))
    
    #%% Paired T - test or Wilcoxon Signed rank test
    if sw_jj_frchs.pvalue < 0.05 and sw_jj.pvalue < 0.05:
        t_test = ttest_rel(gdf_emd_jj_frchs.mean_rest, gdf_emd_jj.mean_rest,
                           alternative='less')
        # H0 : Mean values are identical
        # H1 : Mean of a is less than the mean of b
        print('Paired T-Test was done')
    else:
        wrst = wilcoxon(gdf_emd_jj_frchs.mean_rest, gdf_emd_jj.mean_rest,
                        alternative='less', zero_method='wilcox')
        # H0 : Two related paired samples come from the same distribution
        # H1 : Median of a is less than the median of b
        print('Wilcoxon Signed Rank Test was done')
        
    # ========================================================================
    #%% Sample Test
    # ========================================================================
    gdf_base1, gdf_base2 = load_map_rest_join(pri_thld=6000)
    baseline = GdfCompare(gdf_base1, gdf_base2)
    baseline.run_local_g()
    baseline.run_stats_diff()
    baseline.run_global_moran()
    baseline.plot_compare()
    
    print(baseline.glo_m_before, baseline.glo_m_after)

    for price in tqdm(np.arange(6000, 7000, 100),
                      desc='price variation test'):
        pass
        
    #%% Export by pickle
    gdf_6000 = get_all_attr(price=6000)
    gdf_7000 = get_all_attr(price=7000)
    with open ('gdf_6000_won', 'wb') as f:
        pickle.dump(gdf_6000, f)
    with open ('gdf_7000_won', 'wb') as f:
        pickle.dump(gdf_7000, f)
        
    import pickle
    with open ('gdf_6000_won', 'rb') as f:
        a = pickle.load(f)
    with open ('gdf_7000_won', 'rb') as f:
        b = pickle.load(f)

    
    