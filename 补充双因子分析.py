# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 23:58:48 2022

@author: max_focus
"""

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st


add_sidebar = st.sidebar.selectbox('双因素分析-大宗利润分层的效果最好', ['双高双低','高低结合'])

if add_sidebar=='双高双低':
    st.title('双因素分析')
    st.write()
    st.text_area('分析因子高位高波动和低位低波动时，策略表现：',
 '1. 当大宗利润低位低波动时，期限时序策略年化sharp比率3.34,综合期限策略sharp比率1.59,短期时序动量 截面动量LSsharp比率1.1左右。这时因为大宗利润低位地波动时，经济表现为在底部，平静等待反转时机。这时波动率较低，对信号敏感短期动量类策略的操作准确度就会相对较高；对于期限时序类策略，其买入近月合约基差大的品种，其基差大的原因是因为该品种对应的基本面反弹，从而商品期限结构有contango逐渐变的平坦，转为back结构，这说明基本面改善的品种，其基差会随着基本面改善而扩大，从而未来近月合约的收益率更高。\
\n2. 当宏观信用因子高位高波动时，综合期限、期限时序、期限截面LS三个策略的sharp比率达到0.6左右。较高的当前利率意味着未来投资活动的收缩，流动性收缩，利率高波动说明市场对当前利率预期产生分期，从而近月合约的基差波动较大，期限时序策略能够获利；而从期限结构看，无论当前是back结构 还是 contango结构，高低展期收益品种对冲后能够锁定一个固定的展期收益率，而市场的高波动率下，期限结构形状变动较大，从而提供了更多的套利时机，使得期限截面策略能够锁定更多的无风险套利机会。\
\n3. 其他各类因子高位高波动或者低位低波动时期，大多数策略表现均较差，夏普比率不足0.5；',height=250)
    files = os.listdir('data/two_factor')
    all_data_ret = pd.read_excel('data/原始数据.xlsx',sheet_name='收益率数据',index_col=0)
    
    for file in files:
        factor = pd.read_csv('data/two_factor/'+file,index_col=0).dropna()
        for c in factor.columns:
            factor_i = factor[c]
            hv = factor_i[factor_i==factor_i.max()].dropna().index
            lv = factor_i[factor_i==factor_i.min()].dropna().index
            
            hv_sharp_data = all_data_ret.reindex(index = hv)
            lv_sharp_data = all_data_ret.reindex(index = lv)
            
            hv_data = all_data_ret.reindex(index = hv).melt(var_name='策略',value_name=c)
            hv_data['sign'] = '高位高波动'
            lv_data = all_data_ret.reindex(index = lv).melt(var_name='策略',value_name=c)
            lv_data['sign'] = '低位低波动'
            data_plot = pd.concat([hv_data,lv_data])
            
            st.subheader(file.split('.')[0]+c)
            fig_box =px.box(
                            data_plot,
                            x="策略",   # 分组的数据
                            y=c,  # 箱体图的数值
                            color="sign"  # 颜色分组
                        )
            st.write(file.split('.')[0]+c+'-夏普比率')
            sharp = pd.concat([(np.power((1+hv_sharp_data).cumprod().iloc[-1],250/len(hv_sharp_data))-1)/np.sqrt(np.std(hv_sharp_data)*250),
            (np.power((1+lv_sharp_data).cumprod().iloc[-1],250/len(lv_sharp_data))-1)/np.sqrt(np.std(lv_sharp_data)*250)],axis=1)
            sharp.columns = ['高位高波动','低位低波动']
            st.dataframe(sharp.style.background_gradient(cmap='RdYlGn_r').set_precision(2).highlight_null('yellow').highlight_min(axis=0, color='green').highlight_max(axis=0, color='red'))
            
            st.plotly_chart(fig_box)
        
if add_sidebar=='高低结合':
    st.title('双因素分析')
    st.text_area('分析各类因子高位低波动或低位高波动时，策略表现：',
'1. 当大宗利润低位高波动时，期限截面NT策略年化sharp比率6.13,综合期限策略sharp比率1.85,期限时序策略、库存时序策略的sharp比率1.5左右。低位高波动，基本面反弹，基差动量效应更强，高波动带来期限截面更多锁定套利收益的机会。库存上来看，低利润时期市场上隐性库存较少，库存指标有效性更高。\
\n2. 其他各类因子高位高波动或者低位低波动时期，大多数策略表现均较差，夏普比率不足0.5；',height=200)
   
    files = os.listdir('data/two_factor_gaodi')
    all_data_ret = pd.read_excel('data/原始数据.xlsx',sheet_name='收益率数据',index_col=0)
    
    for file in files:
        factor = pd.read_csv('data/two_factor_gaodi/'+file,index_col=0).dropna()
        for c in factor.columns:
            factor_i = factor[c]
            hv = factor_i[factor_i==factor_i.max()].dropna().index
            lv = factor_i[factor_i==factor_i.min()].dropna().index
            
            hv_sharp_data = all_data_ret.reindex(index = hv).dropna(how='all')
            lv_sharp_data = all_data_ret.reindex(index = lv).dropna(how='all')
            
            hv_data = all_data_ret.reindex(index = hv).melt(var_name='策略',value_name=c)
            hv_data['sign'] = '低位高波动'
            lv_data = all_data_ret.reindex(index = lv).melt(var_name='策略',value_name=c)
            lv_data['sign'] = '高位低波动'
            data_plot = pd.concat([hv_data,lv_data])
            
            st.subheader(file.split('.')[0]+c)
            fig_box =px.box(
                            data_plot,
                            x="策略",   # 分组的数据
                            y=c,  # 箱体图的数值
                            color="sign"  # 颜色分组
                        )
            st.write(file.split('.')[0]+c+'-夏普比率')
            sharp = pd.concat([(np.power((1+hv_sharp_data).cumprod().iloc[-1],250/len(hv_sharp_data))-1)/np.sqrt(np.std(hv_sharp_data)*250),
            (np.power((1+lv_sharp_data).cumprod().iloc[-1],250/len(lv_sharp_data))-1)/np.sqrt(np.std(lv_sharp_data)*250)],axis=1)
            sharp.columns = ['低位高波动','高位低波动']
            st.dataframe(sharp.style.background_gradient(cmap='RdYlGn_r').set_precision(2).highlight_null('yellow').highlight_min(axis=0, color='green').highlight_max(axis=0, color='red'))
            
            st.plotly_chart(fig_box)
        
