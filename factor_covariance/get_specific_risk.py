import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta



import rqdatac
rqdatac.init('rice','rice',('192.168.10.64',16030))


dailyParameters = {'specific_return_length': 252,
                    'sepcific_volatility_half_life': 42,
                    'Newey_West_Auto_Correlation_Lags': 0,
                    'Newey_West_Auto_correlation_half_life': np.nan,
                    'Bayesian_Shrinkage_parameter': 0.1,
                    'volatilityRegimeAdjustment_half_life': 4}

shortTermParameters = {'specific_return_length': 252,
                       'sepcific_volatility_half_life': 84,
                       'Newey_West_Auto_Correlation_Lags': 5,
                       'Newey_West_Auto_correlation_half_life': 252,
                       'Bayesian_Shrinkage_parameter': 0.25,
                       'volatilityRegimeAdjustment_half_life': 42}

longTermParameters = {'specific_return_length': 252,
                       'sepcific_volatility_half_life': 252,
                       'Newey_West_Auto_Correlation_Lags': 5,
                       'Newey_West_Auto_correlation_half_life': 252,
                       'Bayesian_Shrinkage_parameter': 0.25,
                       'volatilityRegimeAdjustment_half_life': 168}


def get_multiperiod_stock_returns(stock_list, latest_trading_date, parameters):

    # 取出多期的收益率，在 Newey West 中计算当期因子收益和滞后因子收益的经验协方差

    end_dates = rqdatac.get_trading_dates(latest_trading_date - timedelta(days=30), latest_trading_date, country='cn')[-parameters.get('Newey_West_Auto_Correlation_Lags'):]

    start_dates = rqdatac.get_trading_dates(latest_trading_date - timedelta(days=400), latest_trading_date, country='cn')[-(parameters.get('specific_return_length') + parameters.get('Newey_West_Auto_Correlation_Lags')):-parameters.get('specific_return_length')]

    daily_specific_return = rqdatac.barra.get_specific_return(stock_list,start_dates[0],end_dates[-1])

    multiperiod_specific_return = {}

    for i in range(1, parameters.get('Newey_West_Auto_Correlation_Lags') + 1):

        multiperiod_specific_return['lag_' + str(i)] = daily_specific_return[-(parameters.get('specific_return_length') + i): -i]

    # 返回当期的因子收益序列，以及滞后N期的因子收益序列

    return daily_specific_return,daily_specific_return[-parameters.get('specific_return_length'):], multiperiod_specific_return


def get_exponential_weight(half_life, length):

    # 生成权重后，需要对数组进行倒序（[::-1]）

    return np.cumprod(np.repeat(1/np.exp(np.log(2)/half_life), length))[::-1]


def Newey_West_adjustment(daily_specific_return, multiperiod_specific_return, parameters):

    stock_list = daily_specific_return.columns.tolist()

    volatility_exp_weight = pd.Series(data=get_exponential_weight(parameters['sepcific_volatility_half_life'], parameters['specific_return_length']),index=daily_specific_return.index)

    correlation_exp_weight = pd.Series(data=get_exponential_weight(parameters['Newey_West_Auto_correlation_half_life'], parameters['specific_return_length']),index=daily_specific_return.index)

    demeaned_daily_specific_return = daily_specific_return - daily_specific_return.mean()

    exp_weighted_daily_specific_return = daily_specific_return.apply(lambda x: volatility_exp_weight * x /volatility_exp_weight.sum())

    correlation = pd.Series(index=stock_list,data=0)

    intermediate_corr = pd.Series(index=stock_list,data=0)

    estimated_var = volatility_exp_weight.dot(demeaned_daily_specific_return.pow(2)) / volatility_exp_weight.sum()

    #newey_west_first_component = correlation_exp_weight.dot(demeaned_daily_specific_return.pow(2)) / correlation_exp_weight.sum()

    for lag in range(1, 6):

        correlation_exp_weight = pd.Series(data=get_exponential_weight(parameters['Newey_West_Auto_correlation_half_life'],parameters['specific_return_length']), index=multiperiod_specific_return['lag_' + str(lag)].index)

        exp_weighted_lag_daily_specific_return = multiperiod_specific_return['lag_' + str(lag)][stock_list].apply(lambda x: correlation_exp_weight * x / correlation_exp_weight.sum())

        print(str(lag)+':',exp_weighted_daily_specific_return[stock].corr(exp_weighted_lag_daily_specific_return[stock]))
        print(str(lag)+':',(1 - lag / (1 + parameters.get('Newey_West_Auto_Correlation_Lags'))) *(exp_weighted_daily_specific_return[stock].corr(exp_weighted_lag_daily_specific_return[stock])))


        #demeaned_lag_specific_return = multiperiod_specific_return['lag_' + str(lag)][stock_list] - multiperiod_specific_return['lag_' + str(lag)][stock_list].mean()

        #print(str(lag)+':',correlation_exp_weight.dot(demeaned_daily_specific_return[stock].values * demeaned_lag_specific_return[stock].values) / correlation_exp_weight.sum())

        #exp_weighted_var.loc[stock] = correlation_exp_weight.dot(demeaned_daily_specific_return[stock].values * demeaned_lag_specific_return[stock].values) / correlation_exp_weight.sum()

        #print(str(lag)+':',(1 - lag / (1 + parameters.get('Newey_West_Auto_Correlation_Lags'))) * (exp_weighted_var.loc[stock] + exp_weighted_var.loc[stock]))

        for stock in stock_list:

            correlation.loc[stock] = exp_weighted_daily_specific_return[stock].corr(exp_weighted_lag_daily_specific_return[stock])

            #exp_weighted_var.loc[stock] = correlation_exp_weight.dot(demeaned_daily_specific_return[stock].values * demeaned_lag_specific_return[stock].values) / correlation_exp_weight.sum()

        intermediate_corr = intermediate_corr + (1 - lag / (1 + parameters.get('Newey_West_Auto_Correlation_Lags'))) * correlation

    #Newey_West_adjustment_var = newey_west_first_component + intermediate_var

    newey_west_adjuestment = np.sqrt(intermediate_corr * estimated_var * 252)

    return newey_west_adjuestment


def structural_risk_adjustment(daily_specific_return_with_lag,daily_specific_return,newey_west_adjuestment,parameters):

    # 结构化调整需要对数据缺失和有肥尾特征的股票进行回归

    effective_data_number = parameters['specific_return_length'] + parameters['Newey_West_Auto_Correlation_Lags'] - daily_specific_return_with_lag.isnull().sum()

    missing_value_test = (effective_data_number - (parameters['specific_return_length'] + parameters['Newey_West_Auto_Correlation_Lags']) / 3)/(2 * (parameters['specific_return_length'] + parameters['Newey_West_Auto_Correlation_Lags']) / 3)

    missing_value_test[missing_value_test < 0] = 0

    missing_value_test[missing_value_test > 1] = 1

    # 对没有数据缺失的股票，判断其是否存在肥尾特征，对有肥尾现象的股票进行模拟回归

    quarter_mean = pd.Series(data=(np.nanpercentile(daily_specific_return,75,axis=0) - np.nanpercentile(daily_specific_return,25,axis=0)) / 1.35, index = daily_specific_return.columns)

    # 计算调整系数，调整系数小于1的认为存在肥尾现象

    fat_tail_test = np.exp(1 - abs((daily_specific_return.std() - quarter_mean)/quarter_mean)).replace(np.nan,0)

    fat_tail_test[fat_tail_test > 1] = 1

    coordination_parameter = missing_value_test * fat_tail_test

    # 对于调整系数小于1和数据缺失的股票要进行结构化模型调整，通过对调整系数等于1的股票进行回归得到调整后的结果

    unadjusted_stocks = coordination_parameter[np.round(coordination_parameter,2) == 1].index.tolist()

    adjusted_stocks = list(set(coordination_parameter.index.tolist()) - set(unadjusted_stocks))

    Y = newey_west_adjuestment.loc[unadjusted_stocks]

    X = daily_specific_return[unadjusted_stocks]

    # 添加常数项
    X.loc['constant'] = 1

    # 最小二乘法得到的系数

    coef = pd.Series(data = np.linalg.inv(X.dot(X.T)).dot(X).dot(np.log(Y)),index=X.index)

    daily_specific_return.loc['constant'] = 1

    expect_structural_specific_risk = np.exp(coef.dot(daily_specific_return.replace(np.nan,0)))

    adjusted_specific_risk = coordination_parameter.loc[adjusted_stocks] * (newey_west_adjuestment.loc[adjusted_stocks].replace(np.nan,0)) + (1 - coordination_parameter.loc[adjusted_stocks]) * expect_structural_specific_risk.loc[adjusted_stocks]

    return pd.concat([newey_west_adjuestment.loc[unadjusted_stocks],adjusted_specific_risk])


def Bayesian_Shrinkage_adjustment(unadjusted_specific_risk,parameters,date):

    # 将经过newey-west和结构化模型调整的specific risk按照波动率分为10组，组内按照市值加权平均

    bayesian_adjustment = pd.Series()

    for i in range(1,11):

        sigma_sorted = unadjusted_specific_risk.where(unadjusted_specific_risk < np.percentile(unadjusted_specific_risk,i*10)).where(unadjusted_specific_risk >= np.percentile(unadjusted_specific_risk,(i-1)*10)).dropna()

        market_cap = rqdatac.get_factor(sigma_sorted.index.tolist(),factor='a_share_market_val',date=date)

        group_market_cap_mean = (market_cap * sigma_sorted / market_cap.sum()).sum()

        delta_group = np.sqrt((sigma_sorted - group_market_cap_mean).pow(2).sum() / len(sigma_sorted))

        shrinkage_intensity = parameters.get('Bayesian_Shrinkage_parameter') * np.abs(sigma_sorted - group_market_cap_mean) / (delta_group + parameters.get('Bayesian_Shrinkage_parameter') * np.abs(sigma_sorted - group_market_cap_mean))

        bayesian_adjustment_group = shrinkage_intensity * group_market_cap_mean + (1 - shrinkage_intensity) * sigma_sorted

        bayesian_adjustment = pd.concat([bayesian_adjustment,bayesian_adjustment_group],axis=0)

    return bayesian_adjustment


def volatility_regime_adjustment(Bayesian_Shrinkage_adjustment_risk,daily_specific_return,parameters):

    volatility_regime_exp_weight = get_exponential_weight(parameters['volatilityRegimeAdjustment_half_life'], parameters['factor_return_length'])

    empirical_factor_volitality = pd.Series(data=daily_specific_return.std(),index=daily_specific_return.columns)

    bias = pd.Series(index=daily_specific_return.index)

    for date in daily_specific_return.index.tolist():

        bias.loc[date] = np.square(daily_specific_return.loc[date]/empirical_factor_volitality).sum()/len(daily_specific_return.columns)

    lambda_f = np.sqrt(volatility_regime_exp_weight.dot(bias)/volatility_regime_exp_weight.sum())

    volatility_regime_adjustment_risk = lambda_f**(2) * Bayesian_Shrinkage_adjustment_risk

    return volatility_regime_adjustment_risk


#test

date = '2018-02-02'

unadjusted_specific_risk = pd.read_csv('/Users/rice/Desktop/与barra对比/covariance_data/unadjusted_specific_risk_20180202.csv',index_col=0)['UnadjSpecRisk%']

latest_trading_date = rqdatac.get_previous_trading_date((datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)))

previous_trading_date = rqdatac.get_previous_trading_date(latest_trading_date)

parameters = shortTermParameters

stock_list = rqdatac.all_instruments(type='CS', date=previous_trading_date)['order_book_id'].values.tolist()

#trading_volume = rqdatac.get_price(stock_list, start_date=latest_trading_date, end_date=latest_trading_date, frequency='1d', fields='volume',country='cn')

#stock_list = trading_volume.loc[latest_trading_date][trading_volume.loc[latest_trading_date].values > 0].index.tolist()

#threshold = str(latest_trading_date - timedelta(days=252))

#stock_list = [stock for stock in stock_list if rqdatac.instruments(stock).listed_date <= threshold]

daily_specific_return_with_lag,daily_specific_return, multiperiod_specific_return = get_multiperiod_stock_returns(stock_list, latest_trading_date, parameters)

newey_west_adjuestment = Newey_West_adjustment(daily_specific_return, multiperiod_specific_return, parameters)

first_step_adjusted_specific_risk = structural_risk_adjustment(daily_specific_return_with_lag,daily_specific_return,newey_west_adjuestment,parameters)

contrast_raw_data = pd.concat([unadjusted_specific_risk,daily_specific_return.std()*np.sqrt(252)],axis=1)

contrast_newey_data = pd.concat([unadjusted_specific_risk,newey_west_adjuestment],axis=1)

contrast_after_structural_data = pd.concat([unadjusted_specific_risk,first_step_adjusted_specific_risk],axis=1)

