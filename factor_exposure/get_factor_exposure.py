import numpy as np
import pandas as pd
import statsmodels.api as st
from datetime import datetime
from datetime import timedelta

import rqdatac

rqdatac.init('rice', 'rice', ('192.168.10.64', 16030))


def get_shenwan_industry_exposure(stock_list, date):

    industry_classification = rqdatac.shenwan_instrument_industry(stock_list, date)

    if date > '2014-01-01':

        shenwan_industry_name = ['农林牧渔', '采掘', '化工', '钢铁', '有色金属', '电子', '家用电器', '食品饮料', '纺织服装', '轻工制造',\
                                 '医药生物', '公用事业', '交通运输', '房地产', '商业贸易', '休闲服务','综合', '建筑材料',  '建筑装饰', '电气设备',\
                                 '国防军工', '计算机', '传媒', '通信', '银行', '非银金融', '汽车', '机械设备']
    else:

        shenwan_industry_name = ['金融服务', '房地产', '医药生物', '有色金属', '餐饮旅游', '综合', '建筑建材', '家用电器',
                                 '交运设备', '食品饮料', '电子', '信息设备', '交通运输', '轻工制造', '公用事业', '机械设备',
                                 '纺织服装', '农林牧渔', '商业贸易', '化工', '信息服务', '采掘', '黑色金属']

    industry_exposure_df = pd.DataFrame(0, index = industry_classification.index, columns = shenwan_industry_name)

    for industry in shenwan_industry_name:

        industry_exposure_df.loc[industry_classification[industry_classification['index_name'] == industry].index, industry] = 1

    return industry_exposure_df


def get_factor_exposure(stock_list,date):

    industry_exposure = get_shenwan_industry_exposure(stock_list, date)

    style_exposure = rqdatac.get_style_factor_exposure(stock_list, date, date, factors = 'all')

    style_exposure.index = style_exposure.index.droplevel('date')

    factor_exposure = pd.concat([style_exposure,industry_exposure],axis=1)

    factor_exposure['comovement'] = 1

    return factor_exposure


date = '2018-07-12'
stock_list = rqdatac.all_instruments(type='CS',country='cn',date=date).order_book_id.tolist()

factor_exposure = get_factor_exposure(stock_list,date)

style_exposure = rqdatac.get_style_factor_exposure(stock_list, date, date, factors=['beta'])
style_exposure.index = style_exposure.index.droplevel('date')

liquidity_exposure = rqdatac.get_factor_exposure(stock_list,date,date,factors=['beta'])
liquidity_exposure.index = liquidity_exposure.index.droplevel('date')

data = pd.concat([factor_exposure['beta'],style_exposure,liquidity_exposure],axis=1)

data.columns=['源代码计算beta暴露度结果','get_style_factor_exposure API','get_factor_exposure API']



