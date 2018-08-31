import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import pickle

import rqdatac
rqdatac.init('rice','rice',('192.168.10.64',16030))


dailyParameters = {'factor_return_length': 252*3,
                    'volatility_half_life': 42,
                    'NeweyWest_volatility_lags': np.nan,
                    'correlation_half_life': 200,
                    'NeweyWest_correlation_lags': np.nan,
                    'VRA_half_life': 4}

shortTermParameters = {'factor_return_length': 1260,
                       'volatility_half_life': 84,
                       'NeweyWest_volatility_lags': 5,
                       'correlation_half_life': 504,
                       'NeweyWest_correlation_lags': 2,
                       'VRA_half_life': 42}

longTermParameters = {'factor_return_length': 1260,
                      'volatility_half_life': 252,
                      'NeweyWest_volatility_lags': 5,
                      'correlation_half_life': 504,
                      'NeweyWest_correlation_lags': 2,
                      'VRA_half_life': 168}

'''
def get_multiperiod_factor_returns(latest_trading_date, parameters):

    industry_factors = ['CNE5S_ENERGY', 'CNE5S_CHEM', 'CNE5S_CONMAT', 'CNE5S_MTLMIN', 'CNE5S_MATERIAL', 'CNE5S_AERODEF',
                        'CNE5S_BLDPROD', 'CNE5S_CNSTENG', 'CNE5S_ELECEQP', 'CNE5S_INDCONG', 'CNE5S_MACH',
                        'CNE5S_TRDDIST',
                        'CNE5S_COMSERV', 'CNE5S_AIRLINE', 'CNE5S_MARINE', 'CNE5S_RDRLTRAN', 'CNE5S_AUTO',
                        'CNE5S_HOUSEDUR',
                        'CNE5S_LEISLUX', 'CNE5S_CONSSERV', 'CNE5S_MEDIA', 'CNE5S_RETAIL', 'CNE5S_PERSPRD', 'CNE5S_BEV',
                        'CNE5S_FOODPROD', 'CNE5S_HEALTH', 'CNE5S_BANKS', 'CNE5S_DVFININS', 'CNE5S_REALEST',
                        'CNE5S_SOFTWARE',
                        'CNE5S_HDWRSEMI', 'CNE5S_UTILITIE']

    style_factors = ['CNE5S_BETA', 'CNE5S_MOMENTUM', 'CNE5S_SIZE', 'CNE5S_EARNYILD', 'CNE5S_RESVOL', 'CNE5S_GROWTH',
                     'CNE5S_BTOP', 'CNE5S_LEVERAGE', 'CNE5S_LIQUIDTY', 'CNE5S_SIZENL']

    country_factor = ['CNE5S_COUNTRY']

    all_factors = industry_factors + style_factors + country_factor

    # 取出多期的收益率，在 Newey West 中计算当期因子收益和滞后因子收益的经验协方差

    end_dates = rqdatac.get_trading_dates(latest_trading_date - timedelta(days=30), latest_trading_date, country='cn')[-parameters.get('NeweyWest_volatility_lags'):]
    start_dates = rqdatac.get_trading_dates(latest_trading_date - timedelta(days=6000), latest_trading_date, country='cn')[-(parameters.get('factor_return_length') + parameters.get('NeweyWest_volatility_lags')):-parameters.get('factor_return_length')]

    # 以百分比为单位，所以乘以 100

    daily_factor_return = rqdatac.barra.get_factor_return(start_dates[0], end_dates[-1], all_factors) * 100
    multiperiod_factor_returns = {}

    for i in range(1, parameters.get('NeweyWest_volatility_lags') + 1):

        multiperiod_factor_returns['lag_' + str(i)] = daily_factor_return[-(parameters.get('factor_return_length') + i): -i]

    # 返回当期的因子收益序列，以及滞后N期的因子收益序列

    multiperiod_factor_returns['current'] = daily_factor_return[-parameters.get('factor_return_length'):]

    return multiperiod_factor_returns
'''


def get_multiperiod_factor_returns(latest_trading_date, parameters):

    if str(latest_trading_date) >= '2014-01-01':

        industry_factors = ['农林牧渔', '采掘', '化工', '钢铁', '有色金属', '电子', '家用电器', '食品饮料', '纺织服装', '轻工制造',\
                            '医药生物', '公用事业', '交通运输', '房地产', '商业贸易', '休闲服务','综合', '建筑材料',  '建筑装饰', '电气设备',\
                            '国防军工', '计算机', '传媒', '通信', '银行', '非银金融', '汽车', '机械设备']
    else:

        industry_factors = ['金融服务', '房地产', '医药生物', '有色金属', '餐饮旅游', '综合', '建筑建材', '家用电器',
                            '交运设备', '食品饮料', '电子', '信息设备', '交通运输', '轻工制造', '公用事业', '机械设备',
                            '纺织服装', '农林牧渔', '商业贸易', '化工', '信息服务', '采掘', '黑色金属']

    style_factors = ['beta', 'momentum', 'earnings_yield', 'residual_volatility', 'growth', 'book_to_price',
                     'leverage', 'liquidity']

    country_factor = ['comovement']

    all_factors = industry_factors + style_factors + country_factor

    # 取出多期的收益率，在 Newey West 中计算当期因子收益和滞后因子收益的经验协方差

    end_dates = rqdatac.get_trading_dates(latest_trading_date - timedelta(days=30), latest_trading_date, country='cn')[-parameters.get('NeweyWest_volatility_lags'):]
    start_dates = rqdatac.get_trading_dates(latest_trading_date - timedelta(days=6000), latest_trading_date, country='cn')[-(parameters.get('factor_return_length') + parameters.get('NeweyWest_volatility_lags')):-parameters.get('factor_return_length')]

    # 以百分比为单位，所以乘以 100

    daily_factor_return = rqdatac.get_factor_return(start_dates[0], end_dates[-1], all_factors) * 100
    multiperiod_factor_returns = {}

    for i in range(1, parameters.get('NeweyWest_volatility_lags') + 1):

        multiperiod_factor_returns['lag_' + str(i)] = daily_factor_return[-(parameters.get('factor_return_length') + i): -i]

    # 返回当期的因子收益序列，以及滞后N期的因子收益序列

    multiperiod_factor_returns['current'] = daily_factor_return[-parameters.get('factor_return_length'):]

    return multiperiod_factor_returns


def get_exponential_weight(half_life, length):

    # 生成权重后，需要对数组进行倒序（[::-1]）
    exp_weight = np.cumprod(np.repeat(1 / np.exp(np.log(2) / half_life), length))[::-1]

    # 返回归一化权重
    return exp_weight / exp_weight.sum()


def get_NeweyWest_weight(multiperiod_factor_returns, lags):

    lagging_terms = list(sorted(multiperiod_factor_returns.keys()))
    lagging_terms.remove('current')
    lagging_terms = lagging_terms[:lags]

    _lagging_coef_calc = lambda x: (1 - x / (lags + 1))
    lagging_weight = pd.Series(_lagging_coef_calc(np.arange(1, lags + 1)), index=lagging_terms)

    return lagging_weight


def volatility_NeweyWest_adjustment(multiperiod_factor_returns, parameters):

    demeaned_current_factor_return = multiperiod_factor_returns['current'].iloc[-parameters['factor_return_length']::] - multiperiod_factor_returns['current'].iloc[-parameters['factor_return_length']::].mean()

    exp_weight = get_exponential_weight(parameters['volatility_half_life'], parameters['factor_return_length'])
    # 先计算当期因子收益率加权经验方差，再加入自协方差

    NeweyWest_adjusted_variance = demeaned_current_factor_return.multiply(demeaned_current_factor_return).T.dot(exp_weight) * 252

    if parameters['NeweyWest_volatility_lags'] is not np.nan:

        lagging_weight = get_NeweyWest_weight(multiperiod_factor_returns, parameters['NeweyWest_volatility_lags'])

        for item in lagging_weight.index.tolist():

            demeaned_lagging_factor_return = multiperiod_factor_returns[item] - multiperiod_factor_returns[item].mean()

            demeaned_lagging_factor_return.index = demeaned_current_factor_return.index

            autocovariance = demeaned_current_factor_return.multiply(demeaned_lagging_factor_return).T.dot(exp_weight) * 252

            NeweyWest_adjusted_variance = NeweyWest_adjusted_variance + lagging_weight[item] * autocovariance

    NeweyWest_adjusted_volatility = pd.Series(np.sqrt(NeweyWest_adjusted_variance), index=NeweyWest_adjusted_variance.index)

    return NeweyWest_adjusted_volatility


def correlation_NeweyWest_adjustment(multiperiod_factor_returns, parameters):

    demeaned_current_factor_return = multiperiod_factor_returns['current'].iloc[-parameters['factor_return_length']::] - multiperiod_factor_returns['current'].iloc[-parameters['factor_return_length']::].mean()

    exp_weight = get_exponential_weight(parameters['correlation_half_life'], parameters['factor_return_length'])

    # 先计算指数加权经验协方差，并作年化处理

    NeweyWest_adjusted_covariance = demeaned_current_factor_return.T.multiply(exp_weight).dot(demeaned_current_factor_return) * 252

    # 先对当期因子收益率进行加权处理，再加入自协方差

    weighted_current_factor_return = demeaned_current_factor_return.T.multiply(exp_weight)

    if parameters['NeweyWest_volatility_lags'] is not np.nan:

        lagging_weight = get_NeweyWest_weight(multiperiod_factor_returns, parameters['NeweyWest_correlation_lags'])

        for item in lagging_weight.index.tolist():

            demeaned_lagging_factor_return = multiperiod_factor_returns[item] - multiperiod_factor_returns[item].mean()

            demeaned_lagging_factor_return.index = demeaned_current_factor_return.index

            autocovariance = weighted_current_factor_return.dot(demeaned_lagging_factor_return) * 252

            # 对于因子收益率 f(k) 和 f(l)，滞后N期的自相关性包括 correlation{f(k,t), f(l,t-N)} 以及 correlation{f(k,t-N), f(l,t)} 两种情况，所以需要进行转置。

            NeweyWest_adjusted_covariance = NeweyWest_adjusted_covariance + lagging_weight[item] * (autocovariance + autocovariance.T)

    factor_volatility = pd.Series(np.sqrt(np.diag(NeweyWest_adjusted_covariance)), index=NeweyWest_adjusted_covariance.index)

    # 计算因子波动率的外积和相关系数矩阵

    volatility_outerproduct = pd.DataFrame(np.outer(factor_volatility, factor_volatility), index = factor_volatility.index, columns = factor_volatility.index)
    NeweyWest_adjusted_correlation = NeweyWest_adjusted_covariance / volatility_outerproduct

    return NeweyWest_adjusted_correlation


def eigenfactor_risk_adjustment(factor_covariance):

    eigenvalue, eigenvector = np.linalg.eig(factor_covariance)

    # 验证是否能从分解项得到原协方差矩阵
    #test = eigenvector.dot(np.diag(eigenvalue.reshape(1, len(factor_covariance.index))[0])).dot(eigenvector.T)
    #test_matrix = pd.DataFrame(data=test,index=factor_covariance.index,columns=factor_covariance.columns)
    #pd.concat([factor_covariance['CNE5S_SIZE'], test_matrix['CNE5S_SIZE']],axis=1)

    sampling_size = 1000
    simulated_volatility_bias = pd.DataFrame(data=np.nan, index=eigenvalue, columns=np.arange(1, sampling_size + 1, 1))

    for sample in np.arange(1, sampling_size + 1, 1):

        # 生成服从 N(0,1) 正态分布的随机变量矩阵（252乘43），再乘以各因子的收益波动率。
        sampling = np.multiply(np.random.normal(0, 1, size=(252, len(factor_covariance.index.tolist()))), np.sqrt(eigenvalue))
        simulated_factor_returns = pd.DataFrame(eigenvector.dot(sampling.T).T, index=np.arange(1, 253, 1), columns=factor_covariance.index)

        # 观察模拟得到的协方差矩阵和原协方差矩阵差异
        #simulated_factor_covariance = pd.DataFrame(index=NeweyWest_adjusted_covariance.index, columns=NeweyWest_adjusted_covariance.columns, data=simulated_factor_returns.cov().values)
        #pd.concat([factor_covariance['CNE5S_SIZE'], simulated_factor_covariance['CNE5S_SIZE']], axis=1)
        #pd.concat([factor_covariance['CNE5S_SIZENL'], simulated_factor_covariance['CNE5S_SIZENL']], axis=1)

        simulated_eigenvalue, simulated_eigenvector = np.linalg.eig(simulated_factor_returns.cov().values)
        simulated_eigenvfactor_variance = simulated_eigenvector.T.dot(factor_covariance).dot(simulated_eigenvector)

        # 观察特征因子的原始方差和模拟方差
        #np.stack([eigenvalue, np.diag(simulated_eigenvfactor_variance)])

        #previous_simulated_volatility_bias = simulated_volatility_bias.mean(axis=1).copy()

        simulated_volatility_bias[sample] = np.diag(simulated_eigenvfactor_variance)/eigenvalue

        # 观察特征因子的偏差如何随特征值变化
        #simulated_volatility_bias[sample].sort_index()

        # 观察偏差统计量随样本量增加的收敛情况
        #print('convergence', sample, np.linalg.norm(simulated_volatility_bias.mean(axis=1) - previous_simulated_volatility_bias))

    # 剔除特征值最小的15个特征因子，再进行偏差项的抛物线拟合

    parabolic_coef = np.polyfit(simulated_volatility_bias.mean(axis=1).sort_index().iloc[15:].index, simulated_volatility_bias.mean(axis=1).sort_index().iloc[15:].values, 2)
    parabolic_fit = np.poly1d(parabolic_coef)
    fitted_volatility_bias = parabolic_fit(simulated_volatility_bias.mean(axis=1).sort_index().index)

    # 参照 USE4 进行经验校正
    scaled_volatility_bias = 1.4 * (fitted_volatility_bias - 1) + 1

    # 对原协方差矩阵进行调整
    adjusted_eigenvalue = (scaled_volatility_bias[::-1] ** 2) * eigenvalue
    adjusted_factor_covariance = eigenvector.dot(np.diag(adjusted_eigenvalue.reshape(1, len(factor_covariance.index))[0])).dot(eigenvector.T)
    adjusted_factor_covariance_df = pd.DataFrame(data=adjusted_factor_covariance, index=factor_covariance.index,columns=factor_covariance.columns)

    # 观察校正后的协方差矩阵和原协方差矩阵差异
    #pd.concat([adjusted_factor_covariance_df['CNE5S_SIZE'], factor_covariance['CNE5S_SIZE']], axis=1)
    #pd.concat([adjusted_factor_covariance_df['CNE5S_SIZENL'], factor_covariance['CNE5S_SIZENL']], axis=1)

    return adjusted_factor_covariance_df


def volatility_regime_adjustment(factor_covariance, factor_return, date, parameters):

    # 以权重半衰期的四倍作为样本数量
    start_date = pd.Timestamp(date) - relativedelta(months=np.ceil(4*parameters['VRA_half_life']/252 * (12 * 3/2)))
    end_date = rqdatac.get_previous_trading_date(date)
    trading_dates = rqdatac.get_trading_dates(start_date, end_date)[-4*parameters['VRA_half_life']:]

    forecast_factor_volatility = pd.DataFrame()

    # for date in trading_dates:
    #
    #     previous_factor_covariance = rqdatac.barra.get_factor_covariance(date)
    #     forecast_factor_volatility[date] = pd.Series(data=np.diag(previous_factor_covariance), index=factor_covariance.index).pow(0.5)
    for date in trading_dates:

        if np.isnan(parameters['NeweyWest_volatility_lags']):

            previous_factor_covariance = pickle.load(open('/Users/rice/Desktop/covariance/daliy_eigen.pkl', 'rb'))[date]

        elif parameters==shortTermParameters:
            previous_factor_covariance = pickle.load(open('/Users/rice/Desktop/covariance/short_eigen.pkl', 'rb'))[date]

        else:
            previous_factor_covariance = pickle.load(open('/Users/rice/Desktop/covariance/long_eigen.pkl', 'rb'))[date]

        forecast_factor_volatility[date] = pd.Series(data=np.diag(previous_factor_covariance), index=factor_covariance.index).pow(0.5)

    # 反年化处理，计算日波动率
    daily_factor_volatility = forecast_factor_volatility/np.sqrt(252)

    exp_weight = get_exponential_weight(parameters['VRA_half_life'], 4*parameters['VRA_half_life'])
    cross_sectional_bias = (factor_return.loc[trading_dates[0]:trading_dates[-1]]/daily_factor_volatility.T).pow(2).mean(axis = 1).pow(0.5)
    factor_volatility_multiplier = cross_sectional_bias.dot(exp_weight)
    VRA_adjusted_covariance = factor_covariance * (factor_volatility_multiplier**2)

    return VRA_adjusted_covariance


def get_factor_covariance(date, parameters):

    latest_trading_date = rqdatac.get_previous_trading_date((datetime.strptime(str(date), "%Y-%m-%d") + timedelta(days=1)))
    multiperiod_factor_returns = get_multiperiod_factor_returns(latest_trading_date, longTermParameters)

    NeweyWest_adjusted_volatility = volatility_NeweyWest_adjustment(multiperiod_factor_returns, parameters)
    NeweyWest_adjusted_correlation = correlation_NeweyWest_adjustment(multiperiod_factor_returns, parameters)

    volatility_outerproduct = pd.DataFrame(np.outer(NeweyWest_adjusted_volatility, NeweyWest_adjusted_volatility), index = NeweyWest_adjusted_volatility.index, columns =NeweyWest_adjusted_volatility.index)
    NeweyWest_adjusted_covariance = NeweyWest_adjusted_correlation * volatility_outerproduct

    eigenfactor_adjusted_covariance = eigenfactor_risk_adjustment(NeweyWest_adjusted_covariance)
    #VRA_adjusted_covariance = volatility_regime_adjustment(eigenfactor_adjusted_covariance, multiperiod_factor_returns['current'], date, parameters)
    return NeweyWest_adjusted_covariance, eigenfactor_adjusted_covariance
    #return NeweyWest_adjusted_covariance, eigenfactor_adjusted_covariance, VRA_adjusted_covariance


# 检查矩阵条件数
#np.linalg.cond(NeweyWest_adjusted_covariance)
#np.linalg.cond(eigenfactor_adjusted_covariance)
#np.linalg.cond(np.matrix(VRA_adjusted_covariance,dtype='float'))


start_date = '2014-01-01'
end_date = '2014-09-09'
trading_dates = rqdatac.get_trading_dates(start_date,end_date)

daily_NeweyWest_adjusted_covariance = {}
daily_eigenfactor_adjusted_covariance = {}

short_NeweyWest_adjusted_covariance = {}
short_eigenfactor_adjusted_covariance = {}

long_NeweyWest_adjusted_covariance = {}
long_eigenfactor_adjusted_covariance = {}

for date in trading_dates:

    daily_NeweyWest_adjusted_covariance[date], daily_eigenfactor_adjusted_covariance[date] = get_factor_covariance(date, dailyParameters)
    short_NeweyWest_adjusted_covariance[date], short_eigenfactor_adjusted_covariance[date] = get_factor_covariance(date, shortTermParameters)
    long_NeweyWest_adjusted_covariance[date], long_eigenfactor_adjusted_covariance[date] = get_factor_covariance(date, shortTermParameters)

    pickle.dump(daily_NeweyWest_adjusted_covariance[date], open('/Users/rice/Desktop/covariance/daliy_nw.pkl', 'wb'))
    pickle.dump(daily_eigenfactor_adjusted_covariance[date],open('/Users/rice/Desktop/covariance/daliy_eigen.pkl', 'wb'))
    pickle.dump(short_NeweyWest_adjusted_covariance[date],open('/Users/rice/Desktop/covariance/short_nw.pkl', 'wb'))
    pickle.dump(short_eigenfactor_adjusted_covariance[date],open('/Users/rice/Desktop/covariance/short_eigen.pkl', 'wb'))
    pickle.dump(long_NeweyWest_adjusted_covariance[date],open('/Users/rice/Desktop/covariance/long_nw.pkl', 'wb'))
    pickle.dump(long_eigenfactor_adjusted_covariance[date],open('/Users/rice/Desktop/covariance/long_eigen.pkl', 'wb'))


