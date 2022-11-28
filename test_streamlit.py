# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 18:56:10 2022

@author: max_focus
"""

import pandas as pd 
import numpy as np 
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime
import plotly
import pickle
from scipy import stats
import seaborn as sns 

def style_negative(v, props=''):
    """ Style negative values in dataframe"""
    try: 
        return props if abs(v) < 3 else None
    except:
        pass
    
def style_positive(v, props=''):
    """Style positive values in dataframe"""
    try: 
        return props if abs(v) >= 3 else None
    except:
        pass    

def find_max(v, props=''):
    """Style positive values in dataframe"""
    try: 
        return props if v == max(v) else None
    except:
        pass   

def find_min(v, props=''):
    """Style positive values in dataframe"""
    try: 
        return props if v == min(v) else None
    except:
        pass 
    
@st.cache
def load_data():
    strategy_sharp = pd.read_excel('结果汇总-业绩-夏普比率.xlsx', sheet_name = None)
    all_strategy_sharp = pd.DataFrame()
    for key in strategy_sharp.keys():
        df_temp =  strategy_sharp[key].copy()
        df_temp = df_temp.set_index(['Unnamed: 0'])
        df_temp.columns = ['信用','利率','增长','海外','美元','通胀'
                           , 'PPI预期差', '情绪', '信用','利率','增长','海外','美元','通胀'
                           , '大宗利润', '历史波动', '股债波动']
        df_temp = df_temp.loc[['上行trend_bias','下行trend_bias','高位level_norm','低位level_norm',
                           '高位level_vol','低位level_vol'],:].iloc[:,6:]
        df_temp.index = ['上行','下行','高位','低位', '高波动','低波动']
        df_temp['策略名称'] = key
        all_strategy_sharp = pd.concat([all_strategy_sharp,df_temp],axis=0)
        
        df_process = all_strategy_sharp.reset_index() # stack()
        df_process['类别'] = df_process['index']+'_'+df_process['策略名称']
        df_process = df_process.drop(columns = ['index','策略名称'])
        df_process = df_process.set_index(['类别'])
        df_process = df_process.stack()
        df_process = df_process.reset_index()
        df_process.columns = ['类别','因子','sharp_ratio']
        df_process['场景'] = df_process['类别'].apply(lambda x:x.split('_')[0])
        df_process['策略'] = df_process['类别'].apply(lambda x:x.split('_')[-1])
        df_process['facet_row'] = np.nan
        df_process['color'] = np.nan
        df_process.loc[df_process['场景'].isin(['上行','下行']),'facet_row'] = '趋势'
        df_process.loc[df_process['场景'].isin(['低位','高位']),'facet_row'] = '水平'
        df_process.loc[df_process['场景'].isin(['高波动','低波动']),'facet_row'] = '波动水平'
        
        df_process.loc[df_process['场景'].isin(['上行','高位','高波动']),'color'] = 'up'
        df_process.loc[df_process['场景'].isin(['下行','低位','低波动']),'color'] = 'down'
    return df_process

@st.cache
def load_data1():
    names = ['high_all_mean','low_all_mean','high_all_std','low_all_std']
    dict2 = {}
    for i in names:
        f_read = open('{}.pkl'.format(i), 'rb')
        dict2[i] = pickle.load(f_read)
        f_read.close()
        
    [high_all_mean,low_all_mean,high_all_std,low_all_std] =  \
                        dict2['high_all_mean'],dict2['low_all_mean'],dict2['high_all_std'],dict2['low_all_std']
                        
    base_mean = pd.read_pickle('base_mean')
    base_std = pd.read_pickle('base_std')
    
    return base_mean,base_std,high_all_mean,low_all_mean,high_all_std,low_all_std


# 区间长度
@st.cache
def get_qujian():
    df_temp = pd.read_excel('结果汇总-业绩-区间长度.xlsx',sheet_name='综合策略',index_col=0)
    df_temp.columns = ['信用','利率','增长','海外','美元','通胀'
                       , 'PPI预期差', '情绪', '信用','利率','增长','海外','美元','通胀'
                       , '大宗利润', '历史波动', '股债波动']
    df_temp = df_temp.loc[['上行trend_bias','下行trend_bias','高位level_norm','低位level_norm',
                       '高位level_vol','低位level_vol'],:].iloc[:,6:]
    df_temp.index = ['上行','下行','高位','低位', '高波动','低波动']
    df_temp = df_temp.stack().reset_index()
    df_temp.columns = ['场景','因子','区间长度']
    df_qujian = df_temp.copy()
    return df_qujian

def ttest(base_mean,high_all_mean,low_all_mean,method='mean'):
    t_res1 = pd.DataFrame(index = high_all_mean.keys(),columns = base_mean.columns)
    
    t_res2 = pd.DataFrame(index = high_all_mean.keys(),columns = base_mean.columns)
    
    t_res3 = pd.DataFrame(index = high_all_mean.keys(),columns = base_mean.columns)
    base_mean = base_mean.astype('float64')
    changjing = list(set(high_all_mean.keys()))
    for changjing_select in changjing:
        up_mean = high_all_mean[changjing_select]
        down_mean = low_all_mean[changjing_select]
        up_mean = up_mean.astype('float64')
        down_mean = down_mean.astype('float64')
        for c in base_mean.columns:
            x0 = base_mean[c]
            x1 = up_mean[c]
            x2 = down_mean[c]
            t_res1.loc[changjing_select,c] = stats.ttest_ind(x0,x1, equal_var = False)[0]
            t_res2.loc[changjing_select,c] = stats.ttest_ind(x0,x2, equal_var = False)[0]
            t_res3.loc[changjing_select,c] = stats.ttest_ind(x2,x1, equal_var = False)[0]
    # t_res1[t_res1.abs()<3] = 0
    t_res1[t_res1.abs()>10] = 10
    # t_res2[t_res2.abs()<3] = 0
    t_res2[t_res2.abs()>10] = 10
    # t_res3[t_res3.abs()<3] = 0
    t_res3[t_res3.abs()>10] = 10
    return t_res1,t_res2,t_res3


#create dataframes from the function 
df_process = load_data()
base_mean,base_std,high_all_mean,low_all_mean,high_all_std,low_all_std = load_data1()
df_qujian = get_qujian()


# ###############################################################################
# #Start building Streamlit App
# ###############################################################################




add_sidebar = st.sidebar.selectbox('分析视角', ['统计检验-bootstrap_ttest','整体_按照因子分类','单因子分析','单场景分析-策略对比'
                                            ,'单场景分析-因子对比']) #,'整体_按照场景分类'
###########################################
if add_sidebar == '统计检验-bootstrap_ttest':
    st.title('CTA策略情景分析')
    st.text_area('概要','1. 此网页展示主要包括两部分，统计检验-bootstrap_ttest和历史情景分析。\n\
2. 统计检验是为了验证策略在经济学假定的情境下策略的历史业绩收益率是有本质的不同，\
还是仅仅由于偶尔因素，报告结合实际背景，对时间序列采用bootstrap法 模拟N次历史情景，\
并通过箱线图和T统计量进行检验。\n\
                3. 历史情景分析主要从两个维度出发，首先是对单个策略的分析，研究单个策略在不同场景下的表现差异,\
                    即整体_按照因子分类。其次，是对策略之间进行对比，研究同一情境下不同策略的表现；\
                    具体分为，单因子分析——在每一个场景，不同策略的实际sharp比率差异；\
                    单场景分析-策略对比，特定策略在不同因子下行（上行）时，对哪个因子更敏感；\
                    单场景分析-因子对比，在特定因子下行（上行）时，所有策略中哪个策略表现最好')
    
    st.text_area('第一部分主要结论如下：',
                 '1. 2016年至今每个场景的总时间约500个交易日左右\
                \n2. 每个场景持续的平均时间约20个交易日左右\n\
                3. 基于以上参数bootstrap采样统计发现，无论是每类场景的平均收益率还是夏普比率，各因子的大多数情境下的分段效果显著\n\
                4. 本部分后面呈现了模拟1000次场景的业绩表现的箱线图 和 非参检验T统计量。通过箱线图对比上下行场景种策略业绩差异，\n\
                    策略对因子上下行敏感的因子主要包括 PPI预期差、大宗利润、通胀\n\
                4.1 期限截面策略，对PPI预期差和大宗利润的波动率与分位数高低敏感\n\
                4.2 综合动量策略，对PPI预期差、通胀波动率、利率趋势、大宗利润水平、历史波动率的波动率 比较敏感\n\
                4.3 期限截面LS策略，对历史波动率因子、大宗利润水平、PPI预期差等比较敏感\n\
                4.4 截面动量NT，对PPI预期差、增长因子比较敏感\n\
                4.5 短期动量时序，对PPI预期差趋势、通胀波动率水平、大宗利润水平、利率趋势比较敏感\n\
                4.6 截面动量LS，对大宗利润水平、PPI预期差、通胀波动、历史波动的波动\n\
                4.7 期限时序策略，对PPI预期差波动率、大宗利润水平敏感\n\
                4.8 截面库存LS，对通胀波动率，PPI预期差水平\n\
                4.9 截面库存NT，对通胀因子、PPI预期差、大宗利润水平敏感\n\
                4.10 库存时序，对PPI预期差、大宗利润水平敏感\n\
                4.11 长期动量时序，对利率的波动率、PPI预期差敏感。',height=150)
    
    st.title('1 策略概览')
    st.subheader('1.1 净值表现')
    
    strategy_navs = pd.read_excel('data/原始数据.xlsx', sheet_name='净值数据2',index_col=0)
    strategys = list(set(df_process['策略']))
    strategy_selects = st.multiselect('筛选几个你想对比的策略:', strategys,strategys)
    # 策略的净值曲线
    strategy_nav_s = strategy_navs[strategy_selects].reset_index().melt(id_vars='DT')
    fig_line = px.line(strategy_nav_s, x='DT', y="value",color='variable')
    st.plotly_chart(fig_line)
    
    ##########################################
    ##########################################
    # 得到策略的sharp
    st.subheader('1.1 总览：策略历史场景下sharp比率对比')
    df_sharp1 = df_process[df_process['策略'].isin(strategy_selects)]
    # df_sharp2 = df_process[df_process['策略'] == strategy_selects[1]]
    fig = px.bar(df_sharp1,
                 x="因子",
          y="sharp_ratio",
          color="策略",
          barmode="group" , # ['stack', 'group', 'overlay', 'relative']
          facet_row = '场景'
        )
    st.plotly_chart(fig)
    
    # 场景时序长度
    st.title("2 每种场景下的区间总长度")
    st.text_area('说明','计算2016年至今每种场景发生的总长度（交易日），并且每次发生是能够平均持续多久的时间（交易日）')
    fig = px.bar(df_qujian,   # 带绘图数据 
                 x="因子",  # x轴
                 y="区间长度",   # y轴
                 color="场景",  # 颜色设置
                 barmode="group",  # 柱状图4种模式之一
                )
    st.plotly_chart(fig)
    
    df_up_range =  pd.read_excel('结果汇总-业绩-区间长度.xlsx',sheet_name='上行平均长度',index_col=0)
    df_down_range = pd.read_excel('结果汇总-业绩-区间长度.xlsx',sheet_name='下行平均长度',index_col=0)
    st.subheader("2.1 因子连续上行区间的平均长度")
    # st.dataframe(df_up_range)
    st.dataframe(df_up_range.style.background_gradient(cmap='RdYlGn_r').set_precision(2).highlight_null('yellow').highlight_min(axis=0, color='green').highlight_max(axis=0, color='red'))
    
    st.subheader("2.2 因子连续下行区间的平均长度")
    # st.dataframe(df_down_range)
    st.dataframe(df_down_range.style.background_gradient(cmap='RdYlGn_r').set_precision(2).highlight_null('yellow').highlight_min(axis=0, color='green').highlight_max(axis=0, color='red'))
    
    
    #uploaded_file = st.file_uploader("上传一张图片", type="jpg")
    #st.image(opencv_image, channels="BGR")
    #     Dimitris N. Politis & Joseph P. Romano (1994) The Stationary Bootstrap, Journal of the American Statistical 
    #         Association, 89:428, 1303-1313, DOI: 10.1080/01621459.1994.10476870  
    st.title('3 bootstrap 采样')
    st.text_area('说明','为了验证特定场景下，策略表现确实不同于所有场景在交替发生的正常经济环境中，而不是由于偶然因素或者随即发生的；报告对比特定场景下策略的收益率表现，通过两独立样本非参数检验两种场景下的收益率分布是否存在差异。考虑到样本区间数据有限，通过自助法采样来对比，每类场景的区间总长度在500天上下，每个场景连续出现的天数在20天上下；\
             考虑到时间序列数据存在时序上的自相关性，并且为了避免时间上开始和结束两端的采样频率过低，报告采用stationary bootstrap方法（Dimitris N. Politis & Joseph P. Romano (1994)）。\
             在全样本中采样1000次，得到1000个正常环境下的区间平均收益，特定场景样本同理，得到了该场景稳定发生时1000次平均收益，将这些不同环境下得到的收益率序列对比，通过非参检验验证是否不同场景收益率不同')
    st.text('某一因子上行的场景下、和正常环境下的 策略收益率分布是否不同')
    st.text('注：统计量绝对值大于10则被赋值为10')
    method = st.selectbox('展示收益率还是夏普比率的t统计量',['收益率','夏普'])
    if method=='收益率':
        high_all_mean = high_all_mean
        base_mean = base_mean
        low_all_mean = low_all_mean
    if method=='夏普':
        high_all_mean = high_all_std
        base_mean = base_std
        low_all_mean = low_all_std
    st.subheader('3.1 历史场景模拟的统计检验')  
    st.subheader('3.1.1 历史场景模拟的统计检验——上行与正常')
    t_res1,t_res2,t_res3 =  ttest(base_mean,high_all_mean,low_all_mean)
    # st.dataframe(t_res1.style.hide().applymap(style_negative, props='color:green;')\
    #              .applymap(style_positive, props='color:red;').applymap(find_max,props='color:yellow;').applymap(find_min,props='color:yellow;'))
    
    st.dataframe(t_res1.style.background_gradient(cmap='RdYlGn_r').set_precision(2))
    
   
    def pro_t_tres(t_res1):
        t_res1_plot = t_res1.stack().reset_index()
        t_res1_plot.columns = ['场景综合','策略','t值']
        t_res1_plot['因子'] = t_res1_plot['场景综合'].apply(lambda x:x.split('-')[0])
        t_res1_plot['场景'] = t_res1_plot['场景综合'].apply(lambda x:x.split('-')[1])
        return t_res1_plot
    
    t_res1_plot = pro_t_tres(t_res1)
    
    
    st.write('某一因子上行的场景下、和正常环境下的 策略收益率分布是否不同')
    strategys = set(t_res1_plot['策略'])
    strategy_sel = st.selectbox('Pick a strategy:', strategys)
    df_filter = t_res1_plot[t_res1_plot['策略'] == strategy_sel]
    fig = px.bar(df_filter,   # 带绘图数据 
                 x="因子",  # x轴
                 y= 't值' ,   # y轴
                 color="场景",  # 颜色设置
                 barmode="group" # 柱状图4种模式之一
                            )

    ## 正常-上行
    st.plotly_chart(fig)
    st.subheader('3.1.2 历史场景模拟的统计检验——下行与正常')
    st.write('某一因子下行的场景下、和正常环境下的 策略收益率分布是否不同')
    # st.dataframe(t_res2.style.hide().applymap(style_negative, props='color:green;')\
    #            .applymap(style_positive, props='color:red;').applymap(find_max,props='color:yellow;').applymap(find_min,props='color:yellow;'))#.format(df_to_pct))
    st.dataframe(t_res2.style.background_gradient(cmap='RdYlGn_r').set_precision(2))
    
    t_res2_plot = pro_t_tres(t_res2)
    strategys2 = set(t_res2_plot['策略'])
    strategy_sel = st.selectbox('Pick a strategy2:', strategys2)
    df_filter = t_res2_plot[t_res2_plot['策略'] == strategy_sel]
    fig = px.bar(df_filter,   # 带绘图数据 
                 x="因子",  # x轴
                 y= 't值' ,   # y轴
                 color="场景",  # 颜色设置
                 barmode="group" # 柱状图4种模式之一
                 # facet_row="场景",  #  行
                 # facet_col="场景"  # 列
                 # category_orders={
                 #     "day": ["Thur", "Fri", "Sat", "Sun"],
                 #     "time": ["Lunch", "Dinner"]   # 分类顺序设置
                                   # }
                )
    ## 正常-下行
    st.plotly_chart(fig)
    
    st.subheader('3.1.3 历史场景模拟的统计检验——下行与上行')
    st.write('某一因子上行的场景下、和下行环境下的 策略收益率分布是否不同')
    st.dataframe(t_res3.style.background_gradient(cmap='RdYlGn_r').set_precision(2))
    
    t_res3_plot = pro_t_tres(t_res3)
    strategys3 = set(t_res3_plot['策略'])
    strategy_sel = st.selectbox('Pick a strategy3:', strategys3)
    df_filter = t_res3_plot[t_res3_plot['策略'] == strategy_sel]
    fig = px.bar(df_filter,   # 带绘图数据 
                 x="因子",  # x轴
                 y= 't值' ,   # y轴
                 color="场景",  # 颜色设置
                 barmode="group" # 柱状图4种模式之一
                 # facet_row="场景",  #  行
                 # facet_col="场景"  # 列
                 # category_orders={
                 #     "day": ["Thur", "Fri", "Sat", "Sun"],
                 #     "time": ["Lunch", "Dinner"]   # 分类顺序设置
                                   # }
                )
    ## 上行-下行
    st.plotly_chart(fig)
    
    ####################################
    #####################################
    st.title('3.2 模拟N次场景中 sharp比率和平均收益率的分布')
    st.write('三行图，第一行trend_bias代表上下行趋势，第二行level_norm代表因子分位数的高低，第三行代表波动率的高低')
    method = st.selectbox('选择展示收益率还是sharp:', ['收益率','夏普'])        
    strategy_select1 = st.selectbox('Pick one strategy:', strategys)
    base_ = base_mean[strategy_select1]
    moni_up = pd.DataFrame(columns = high_all_mean.keys())
    moni_down = pd.DataFrame(columns = high_all_mean.keys())
    for changjing in high_all_mean.keys():
        moni_up[changjing] = high_all_mean[changjing][strategy_select1]
        moni_down[changjing] = low_all_mean[changjing][strategy_select1]
    moni_up['base'] = base_
    moni_down['base'] = base_
    moni_up_m = moni_up.melt(value_name='up')
    moni_down_m = moni_down.melt(value_name='down')
    st.subheader('上行-正常')
    fig_box =px.box(
                    moni_up_m,
                    x="variable",   # 分组的数据
                    y="up",  # 箱体图的数值
                    color="variable"  # 颜色分组
                )
    st.plotly_chart(fig_box)
    
    st.subheader('下行-正常')
    fig_box1 =px.box(
                    moni_down_m,
                    x="variable",   # 分组的数据
                    y="down",  # 箱体图的数值
                    color="variable",  # 颜色分组
                    
                )
    st.plotly_chart(fig_box1)
    
    
    st.subheader('上行-下行')
    moni_up_m['sign'] = 'up'
    moni_up_m['factor'] = moni_up_m['variable'].apply(lambda x:x.split('-')[0])
    moni_up_m['changjing'] = moni_up_m['variable'].apply(lambda x:x.split('-')[-1])
    moni_up_m = moni_up_m.rename(columns={'up':'value'})
    
    moni_down_m['sign'] = 'down'
    moni_down_m['factor'] = moni_down_m['variable'].apply(lambda x:x.split('-')[0])
    moni_down_m['changjing'] = moni_down_m['variable'].apply(lambda x:x.split('-')[-1])
    moni_down_m = moni_down_m.rename(columns={'down':'value'})
    moni_all_ = pd.concat([moni_up_m,moni_down_m])
    moni_all_ = moni_all_[moni_all_['variable']!='base']
    # for changjing_i in set(moni_up_m['changjing']):
    #     moni_all_need = moni_all_[moni_all_['changjing']==changjing_i]
    fig_box2 =px.box(
                    moni_all_,
                    x="factor",   # 分组的数据
                    y="value",  # 箱体图的数值
                    color="sign",# 颜色分组
                    facet_row='changjing'
                )
    
    st.plotly_chart(fig_box2)

###########################################
if add_sidebar == '整体_按照因子分类':
    st.header("整体_按照因子分类")
    st.text('针对每一个策略，从因子出发，研究当前策略在每个因子上行下行时的实际sharp比率')
    
    for strategy in set(df_process['策略']):
        st.subheader(strategy)
        if strategy == '截面动量NT':
            st.text('''
                1. 策略sharp_ratio与趋势的关系
                   1. 正相关：股债波动，即股债波动趋势上行，sharp越高
                   2. 负相关：利率、增长
                2. 策略sharp_ratio与水平
                   1. 正相关：海外，即海外因子（中债收益率-美债收益率）分位数越高，sharp比率越好
                   2. 负相关：利率、历史波动
                3. 策略sharp_ratio与波动水平
                   1. 正相关：历史波动率
                   2. 负相关：增长、PPI预期差
                   ''')
        if strategy == '截面动量LS':
            st.text('''
               1. 策略sharp_ratio与趋势
                   1. 正相关：-
                   2. 负相关：情绪、增长、历史波动率
                2. 策略sharp_ratio与水平
                   1. 正相关：信用
                   2. 负相关：海外、通胀、历史波动
                3. 策略sharp_ratio与波动水平
                   1. 正相关：增长、利率
                   2. 负相关：PPI预期差、情绪、通胀、历史波动
                  ''')
        if strategy == '长期动量时序':
            st.text('''
            1. 趋势
               1. 正相关：-
               2. 负相关：-
            2. 水平
               1. 正相关：PPI预期差、增长
               2. 负相关：情绪、利率、美元
            3. 波动水平
               1. 正相关：利率、PPI预期差、股债波动
               2. 负相关：大宗利润、历史波动
                   ''')
        if strategy == '短期动量时序':
            st.text('''
            1. 趋势
               1. 正相关：信用
               2. 负相关：利率、增长、海外
            2. 水平
               1. 正相关：-
               2. 负相关：通胀
            3. 波动水平
               1. 正相关：-
               2. 负相关：PPI预期差、通胀、股债波动
                   ''')
        if strategy == '综合动量':
           st.text('''
        1. 趋势
           1. 正相关：-
           2. 负相关：利率、增长、海外
        2. 水平
           1. 正相关：-
           2. 负相关：海外、通胀
        3. 波动水平
           1. 正相关：增长、利率
           2. 负相关：PPI预期差、通胀、历史波动
                  ''')
                  
        if strategy == '截面库存NT':
            st.text('''
        1. 趋势
           1. 正相关：-
           2. 负相关：PPI预期差、通胀
        2. 水平
           1. 正相关：PPI预期差
           2. 负相关：通胀、历史波动
        3. 波动水平
           1. 正相关：-
           2. 负相关：情绪、通胀、利率
                   ''')
       
        if strategy == '截面库存LS':
            st.text('''
        1. 趋势
           1. 正相关：美元
           2. 负相关：PPI预期差、通胀
        2. 水平
           1. 正相关：PPI预期差
           2. 负相关：信用
        3. 波动水平
           1. 正相关：-
           2. 负相关：通胀
                   ''')
       
        if strategy == '综合库存':
            st.text('''
        1. 趋势
           1. 正相关：大宗利润
           2. 负相关：PPI预期差、通胀
        2. 水平
           1. 正相关：PPI预期差
           2. 负相关：通胀
        3. 波动水平
           1. 正相关：增长、PPI预期差
           2. 负相关：通胀
                   ''')
                   
        if strategy == '期限截面LS':
            st.text('''
        1. 趋势
           1. 正相关：PPI预期差、情绪、海外、股债波动
           2. 负相关：历史波动率
        2. 水平
           1. 正相关：增长
           2. 负相关：利率、通胀
        3. 波动水平
           1. 正相关：增长、PPI预期差、通胀、历史波动
           2. 负相关：情绪
                   ''')
                   
        if strategy == '期限截面NT':
            st.text('''
        1. 趋势
           1. 正相关：利率
           2. 负相关：PPI预期差、通胀
        2. 水平
           1. 正相关：PPI预期差
           2. 负相关：通胀、历史波动率
        3. 波动水平
           1. 正相关：-
           2. 负相关：情绪、利率、通胀
                   ''')       
        
        if strategy == '期限时序':
            st.text('''
        1. 趋势
           1. 正相关：PPI预期差、情绪
           2. 负相关：美元、历史波动
        2. 水平
           1. 正相关：-
           2. 负相关：通胀、情绪
        3. 波动水平
           1. 正相关：增长、信用、美元
           2. 负相关：情绪
                   ''')    
                   
        if strategy == '综合期限':
            st.text('''
        1. 趋势
           1. 正相关：PPI预期差、情绪、海外、股债波动
           2. 负相关：历史波动
        2. 水平
           1. 正相关：-
           2. 负相关：通胀、情绪
        3. 波动水平
           1. 正相关：美元
           2. 负相关：情绪
                   ''')           
                   
        df_filter = df_process[df_process['策略']==strategy]
        fig = px.bar(df_filter,   # 带绘图数据 
                     x="因子",  # x轴
                     y="sharp_ratio",   # y轴
                     color="color",  # 颜色设置
                     barmode="group",  # 柱状图4种模式之一
                     facet_row="facet_row",  #  行
                     # category_orders={
                     #     "day": ["Thur", "Fri", "Sat", "Sun"],
                     #     "time": ["Lunch", "Dinner"]   # 分类顺序设置
                                       # }
                    )
        fig.update_layout(title_text="{}".format(strategy))
        st.plotly_chart(fig)
        
        
if add_sidebar == '单因子分析':
    st.title("单因子分析")
    st.subheader('在每一个场景，不同策略的实际sharp比率差异，找到显著表现较好的策略')
    st.text_area('动量类策略分析','长期动量和短期动量时序具有明显负相关关系,如以3个月波动率衡量的 高波动时期长期动量表现更好，而低波动时期短期动量表现更好，这时因为高波动往往对应着趋势，敏感的短期时序动量在高波动时期无法把握长期趋势，短期内又容易产生较多错误信号；截面动量LS，截面动量NT表现出明显的负相关关系，是因为投资产品的商品池不同')
    st.text_area('期限类策略分析','首先期限时序类策略表现显著优于期限截面类策略，其次，期限截面LS和期限截面NT具有一定负相关关系,如以3个月波动率衡量的 高波动时期期限截面LS表现更好，而低波动时期期限截面NT表现更好')
    st.text_area('库存类策略分析','首先库存时序类策略表现显著优于库存截面类策略')
    
    factor = list(set(df_process['因子']))
    for factor_select in factor:
        st.subheader(factor_select)
        df_filtered = df_process[df_process['因子'] == factor_select]
      
        fig = px.bar(df_filtered,
             x="场景",
             y="sharp_ratio",
             color="策略",
             barmode="group",  # ['stack', 'group', 'overlay', 'relative']
            width=1000
            )
        st.plotly_chart(fig)
        
        if factor_select == '情绪':
            st.text_area('表现较好的策略:',
                    '''1. 上行：期限时序、库存时序、期限截面LS\n\
                        2. 下行：库存时序\n\
                        3. 高位：库存时序\n\
                        4. 低位：期限时序\n\
                        5. 高波动：短期动量时序、长期动量时序\n\
                        6. 低波动：期限时序'''
                        )
        if factor_select == '海外':
            st.text_area('表现较好的策略:',
                    '''1. 上行：期限时序、库存时序
                        2. 下行：期限时序、短期动量
                        3. 高位：期限截面NT
                        4. 低位：期限时序
                        5. 高波动：短期动量时序
                        6. 低波动：库存时序'''
                        )
        if factor_select == '股债波动':
            st.text_area('表现较好的策略:',
                    '''1. 上行：期限时序、期限截面LS、期限截面NT
                        2. 下行：期限时序、库存时序
                        3. 高位：期限时序、短期动量时序
                        4. 低位：期限时序、长期动量时序、库存时序
                        5. 高波动：期限时序、库存时序
                        6. 低波动：期限时序'''
                        )
        if factor_select == '历史波动':
            st.text_area('表现较好的策略:',
                    '''1. 上行：库存时序
2. 下行：期限时序
3. 高位：期限时序、库存时序
4. 低位：库存时序
5. 高波动：期限时序
6. 低波动：期限时序'''
                        )
        if factor_select == '信用':
            st.text_area('表现较好的策略:',
                    '''1. 上行：库存时序
2. 下行：期限时序、库存时序
3. 高位：库存时序、截面库存LS
4. 低位：期限时序、库存时序、截面库存LS
5. 高波动：期限时序、长期动量时序
6. 低波动：库存时序、期限截面NT'''
                        )
        if factor_select == 'PPI预期差':
            st.text_area('表现较好的策略:',
                    '''1. 上行：期限时序、期限截面LS
2. 下行：库存时序、截面库存NT
3. 高位：库存时序、截面库存LS
4. 低位：期限时序
5. 高波动：期限时序、长期动量时序
6. 低波动：短期动量时序、截面动量NT、期限时序'''
                )
        if factor_select == '利率':
            st.text_area('表现较好的策略:',
                    '''1. 上行：期限时序、库存时序
2. 下行：期限时序
3. 高位：期限时序、期限截面NT
4. 低位：期限时序、期限截面LS
5. 高波动：长期动量时序
6. 低波动：短期动量时序、截面库存NT、库存时序'''
                        )
        if factor_select == '美元':
            st.text_area('表现较好的策略:',
                    '''1. 上行：库存时序
2. 下行：期限时序
3. 高位：期限截面
4. 低位：期限时序、库存时序、期限截面LS、长期动量时序
5. 高波动：期限时序、库存时序
6. 低波动：期限截面NT'''
                        )
        if factor_select == '增长':
            st.text_area('表现较好的策略:',
                    '''1. 上行：期限时序
2. 下行：期限截面NT、库存时序
3. 高位：期限时序、期限截面LS、长期动量时序
4. 低位：短期动量时序
5. 高波动：期限时序、库存时序
6. 低波动：期限截面NT、库存时序'''
                        )
        if factor_select == '大宗利润':
            st.text_area('表现较好的策略:',
                    '''1. 上行：期限时序、库存时序
2. 下行：
3. 高位：期限时序、库存时序
4. 低位：期限时序、期限截面LS
5. 高波动：期限时序
6. 低波动：期限截面NT'''
                        )
        if factor_select == '通胀':
            st.text_area('表现较好的策略:',
                    '''1. 上行：期限时序
2. 下行：库存时序
3. 高位：期限时序、库存时序
4. 低位：期限时序、库存时序、期限截面LS
5. 高波动：期限时序
6. 低波动：短期动量时序T'''
                        )
            
if add_sidebar == '单场景分析-策略对比':
    st.write("单场景分析-策略对比")
    st.text('特定策略在不同因子下行（上行）时，对哪个因子更敏感')
    changjing = list(set(df_process['场景']))
    changjing_select = st.selectbox('Pick a Video:', changjing)
    df_filtered = df_process[df_process['场景'] == changjing_select]
        
    fig = px.bar(df_filtered[df_filtered['因子'].isin(['信用','利率','增长','海外','美元','通胀'])],
         x="策略",
         y="sharp_ratio",
         color="因子",
         barmode="group",  # ['stack', 'group', 'overlay', 'relative']
        )
    st.plotly_chart(fig)
    
    fig = px.bar(df_filtered[df_filtered['因子'].isin(['PPI预期差', '情绪','大宗利润', '历史波动', '股债波动'])],
         x="策略",
         y="sharp_ratio",
         color="因子",
         barmode="group" # ['stack', 'group', 'overlay', 'relative']
         # facet_row="因子"
        )
    st.plotly_chart(fig)
    
    
######################################
if add_sidebar == '单场景分析-因子对比':
    st.write("单场景分析-因子对比")
    st.text('所有策略在特定因子下行（上行）时，哪个策略表现最好')
    changjing1 = list(set(df_process['场景']))
    changjing_select1 = st.selectbox('Pick a Video:', changjing1)
    df_filtered = df_process[df_process['场景'] == changjing_select1]
    
    fig = px.bar(df_filtered[df_filtered['因子'].isin(['信用','利率','增长','海外','美元','通胀'])],
          x="因子",
          y="sharp_ratio",
          color="策略",
          barmode="group"  # ['stack', 'group', 'overlay', 'relative']
        )
    st.plotly_chart(fig)
    
    fig = px.bar(df_filtered[df_filtered['因子'].isin(['PPI预期差', '情绪','大宗利润', '历史波动', '股债波动'])],
         x="因子",
         y="sharp_ratio",
         color="策略",
         barmode="group" # ['stack', 'group', 'overlay', 'relative']
         # facet_row="因子"
        )
    st.plotly_chart(fig)

    
# if add_sidebar == '整体_按照场景分类':
    
#     for strategy in set(df_process['策略']):
#         st.subheader(strategy)
#         df_filter = df_process[df_process['策略']==strategy]
#         fig = px.bar(df_filter,   # 带绘图数据 
#                      x="因子",  # x轴
#                      y="sharp_ratio",   # y轴
#                      color="因子",  # 颜色设置
#                      barmode="group",  # 柱状图4种模式之一
#                      facet_row="策略",  #  行
#                      facet_col="场景"  # 列
#                      # category_orders={
#                      #     "day": ["Thur", "Fri", "Sat", "Sun"],
#                      #     "time": ["Lunch", "Dinner"]   # 分类顺序设置
#                                        # }
#                     )
#         st.plotly_chart(fig)   
    
# # if add_sidebar == '单个策略分析_bootstrap':
    
   