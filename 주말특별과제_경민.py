#!/usr/bin/env python
# coding: utf-8

# # Mission1

# ## 전주시 지도 GeoPandas로 시각화하기

# In[1]:


import geopandas as gpd
import matplotlib.pyplot as plt


# In[2]:


df = gpd.read_file('C:\\Users\\user\\Desktop\\공공빅데이터 청년인턴십\\개인 학습\\전북 전주시\\LSMD_CONT_LDREG_45110.shp', encoding='euckr')


# In[3]:


df.head()


# In[4]:


ax = df.plot(color='white', edgecolor='k', figsize=(15,15))
ax.set_title('Jeonjusi', fontsize = 40)
ax.set_axis_off()

plt.plot()


# #### cf. 전주시 shp 데이터 기준으로 출력

# In[ ]:





# In[5]:


df2 = gpd.read_file('C:\\Users\\user\\Desktop\\공공빅데이터 청년인턴십\\개인 학습\\전북 전주시\\hang2.shp', encoding='utf-8')


# In[6]:


df2.head()


# In[7]:


#대근 킴 한글폰트 설정 솔루션_그래프 그릴 때 깨짐현상 해결
from matplotlib import font_manager

for font in font_manager.fontManager.ttflist:
    if 'Malgun' in font.name:
        print(font.name, font.fname)
plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)        
        
        
        
bx = df2.plot(column = 'ADM_DR_NM', legend = True, categorical=True, figsize=(10, 10))
bx.set_title('Jeonju', fontsize=40)
bx.set_axis_off()


# #### cf. 읍면동 shp 데이터로 출력

# In[ ]:





# # Mission2

# ## 전국 음식점 데이터 업종(한식 중식 양식 등) 기준으로 나눠서 pie chart로 구현하기

# ### cf. 전국 음식점 파일 read가 안되서 ㅠ 전주 음식점 데이터로 대체합니다..

# In[8]:


import pandas as pd


# In[9]:


a = pd.read_csv('C:\\Users\\user\\Desktop\\데이터\\전라북도 전주시_음식점 정보_20201229.csv', encoding = 'utf-8')


# In[10]:


a.columns


# In[11]:


jj_res = a[['식당ID','식당명','업종(메뉴)정보','식당 대표전화번호','도로명주소','지번주소','식당위도','식당경도','대표메뉴']]


# In[12]:


jj_ctg = jj_res['업종(메뉴)정보'].value_counts()


# In[13]:


'''
[Daeguen K's code support]

jj_ctg = jj_res['업종(메뉴)정보'].value_counts()
res_ctg = jj_ctg.index

not_ctg = ['생선회', '김밥', '맥주', '족발', '피자', '차', '국밥']
only_ctg = res_ctg[~res_ctg.isin(not_ctg)]
only_ctg
'''


# In[14]:


res_ctg = jj_ctg.index
res_ctg


# In[79]:


not_for_chil = ['카페','술집','차','맥주','케이크전문','도너츠','아이스크림','와인','과일주스']
only_ctg = res_ctg[~res_ctg.isin(not_for_chil)]
only_ctg


# ## **막히는 부분 

# In[80]:


kfood = ['한식','김밥','국수','국밥','찌개,전골','족발','돼지고기','죽','닭갈비','감자탕','순대국','곱창,막창,양','한정식','추어탕','백반','곰탕','쌈밥','낙지','한식뷔페','주꾸미','해장국','보쌈','설렁탕','생선구이','찜닭','아귀찜','전','해물탕','매운탕','백숙','사철,영양탕','보리밥','두부요리집','삼계탕']
bfood = ['분식','칼국수','만두','냉면','우동','라멘','오뎅']
mfood = ['치킨','육류','족발','돼지고기','소고기구이','양꼬치','양갈비']
wfood = ['피자','햄버거','패밀리레스토랑','파스타','브런치','와플','이탈리아음식']
#대강 분류를 해보았다.
#어려운점_1: 정확하게 음식을 분류하기에 기준이 모호(기술적<주관적 요소)
#어려운점_2: 한식(kfood)으로 설정된 only_k_f 리스트의 요소들을 모두 합치고 싶은데, 인덱스 해제하는 법을 잘 모르겠다.
# ㄴ 만약 인덱스 해제 후 음식 종류별로 count를 합산할 수만 있다면, 원 그래프는 그릴 수 있을 것 같다.

only_k_f = only_ctg[only_ctg.isin(kfood)]
only_b_f = only_ctg[only_ctg.isin(bfood)]
only_m_f = only_ctg[only_ctg.isin(mfood)]
only_w_f = only_ctg[only_ctg.isin(wfood)]


# In[109]:


kf_df=jj_res[jj_res['업종(메뉴)정보'].isin(only_k_f)]
bf_df=jj_res[jj_res['업종(메뉴)정보'].isin(only_b_f)]
mf_df=jj_res[jj_res['업종(메뉴)정보'].isin(only_m_f)]
wf_df=jj_res[jj_res['업종(메뉴)정보'].isin(only_w_f)]


# In[110]:


kf_series=kf_df['업종(메뉴)정보'].value_counts()
bf_series=bf_df['업종(메뉴)정보'].value_counts()
mf_series=mf_df['업종(메뉴)정보'].value_counts()
wf_series=wf_df['업종(메뉴)정보'].value_counts()

kf_num = kf_series.sum()
bf_num = bf_series.sum()
mf_num = mf_series.sum()
wf_num = wf_series.sum()


# In[ ]:





# ## Pie 차트 그리기

# In[133]:


labels = ['kfood','bfood','m_food','w_food']
ratio = [kf_num, bf_num, mf_num, wf_num]
wedgeprops = {'width': 0.7}

plt.pie(ratio, labels=labels, autopct='%0.1d%%', wedgeprops=wedgeprops)


# #### 완벽하게 분류할 수 없어서 가능한 곳 까지

# In[ ]:

#수정 1
#수정 2
#수정 3
#경민이형 저 담배피고 올게여
#sunny 교육 중

#sanha
tset
test test test
xptmxm
