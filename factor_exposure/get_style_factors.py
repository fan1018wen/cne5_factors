import sys

# sys.path.append("/Users/rice/Documents/cne5_factors/factor_exposure/")
sys.path.append("/Users/jjj728/git/cne5_factors/style_factor_exposure/")

from intermediate_variables import *
from operators import *
from atomic_descriptors import *
from get_stock_beta import *

import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
from sklearn import linear_model

import rqdatac

rqdatac.init("ricequant", "Ricequant123", ('rqdatad-pro.ricequant.com', 16004))


def get_market_portfolio_beta(stock_excess_return, market_portfolio_excess_return, market_cap_on_current_day):

    exp_weight = get_exponential_weight(half_life=63, length=252)

    weighted_stock_excess_return = stock_excess_return.T.multiply(exp_weight).T

    weighted_market_portfolio_excess_return = market_portfolio_excess_return.multiply(exp_weight).T

    weighted_market_portfolio_variance = weighted_market_portfolio_excess_return.var()

    beta = [weighted_market_portfolio_excess_return.cov(
        weighted_stock_excess_return[stock]) / weighted_market_portfolio_variance for stock in
            stock_excess_return.columns]

    market_portfolio_beta = pd.Series(beta, index=stock_excess_return.columns)

    market_portfolio_beta_exposure = winsorization_and_market_cap_weighed_standardization(market_portfolio_beta,
                                                                                          market_cap_on_current_day)

    # 细分因子 historical_sigma 的计算需要 beta 的原始值，所以同时返回原始暴露度和标准化暴露度

    return market_portfolio_beta, market_portfolio_beta_exposure


def get_momentum(stock_list, date, market_cap_on_current_day):

    trading_date_525_before = rqdatac.get_trading_dates(date - timedelta(days=1000), date, country='cn')[-525]

    trading_date_21_before = rqdatac.get_trading_dates(date - timedelta(days=40), date, country='cn')[-21]

    # 共需要 525 - 21 = 504 个交易日的收益率

    exp_weight = get_exponential_weight(half_life=126, length=504)

    # 提取股票价格数据，对于退市情况，考虑作股价向前填补（日收益率为0）

    daily_return = rqdatac.get_price(stock_list, trading_date_525_before, trading_date_21_before, frequency='1d',
                                     fields='close').fillna(method='ffill').pct_change()[1:]

    # 剔除收益率数据存在空值的股票

    inds = daily_return.isnull().sum()[daily_return.isnull().sum() > 0].index

    daily_return = daily_return.drop(daily_return[inds], axis=1)

    # 把复利无风险日收益率转为日收益率

    compounded_risk_free_return = rqdatac.get_yield_curve(start_date=trading_date_525_before, end_date=date, tenor='0S')

    risk_free_return = (((1 + compounded_risk_free_return) ** (1 / 365)) - 1).loc[daily_return.index]

    relative_strength = np.log(1 + daily_return).T.subtract(np.log(1 + risk_free_return.iloc[:, 0])).dot(exp_weight)

    processed_relative_strength = winsorization_and_market_cap_weighed_standardization(relative_strength, market_cap_on_current_day[relative_strength.index])

    return processed_relative_strength


def get_size(market_cap_on_current_day):

    processed_size = winsorization_and_market_cap_weighed_standardization(np.log(market_cap_on_current_day.replace(0, np.nan)), market_cap_on_current_day)

    return processed_size


def get_earnings_yield(latest_trading_date, market_cap_on_current_day, recent_report_type):

    earnings_to_price_ratio = get_earnings_to_price_ratio(latest_trading_date, recent_report_type, market_cap_on_current_day)

    cash_earnings_to_price_ratio = get_cash_earnings_to_price_ratio(latest_trading_date, recent_report_type, market_cap_on_current_day)

    earnings_yield = earnings_to_price_ratio

    return earnings_to_price_ratio, cash_earnings_to_price_ratio, earnings_yield


def get_residual_volatility(stock_list, latest_trading_date, stock_excess_return, market_portfolio_excess_return,market_cap_on_current_day, market_portfolio_beta_exposure, market_portfolio_beta):

    daily_standard_deviation_exposure = get_daily_standard_deviation(stock_excess_return, market_cap_on_current_day)

    cumulative_range_exposure = get_cumulative_range(stock_list, latest_trading_date, market_cap_on_current_day)

    historical_sigma_exposure = get_historical_sigma(stock_excess_return, market_portfolio_excess_return,market_portfolio_beta, market_portfolio_beta_exposure,market_cap_on_current_day)

    atomic_descriptors_df = pd.concat([daily_standard_deviation_exposure, cumulative_range_exposure, historical_sigma_exposure], axis=1)

    atomic_descriptors_df.columns = ['daily_standard_deviation', 'cumulative_range', 'historical_sigma']

    atom_descriptors_weight = pd.Series(data=[0.74, 0.16, 0.1],index=['daily_standard_deviation', 'cumulative_range', 'historical_sigma'])

    residual_volatility = atomic_descriptors_imputation_and_combination(atomic_descriptors_df, atom_descriptors_weight)

    orthogonalized_weighted_residual_volatility = orthogonalize(target_variable=residual_volatility,reference_variable=market_portfolio_beta_exposure,regression_weight=np.sqrt(market_cap_on_current_day) / (np.sqrt(market_cap_on_current_day).sum()))

    processed_residual_volatility_exposure = winsorization_and_market_cap_weighed_standardization(orthogonalized_weighted_residual_volatility, market_cap_on_current_day)

    return daily_standard_deviation_exposure, cumulative_range_exposure, historical_sigma_exposure, processed_residual_volatility_exposure


def get_growth(latest_trading_date, market_cap_on_current_day, recent_five_annual_shares, recent_report_type):

    sales_growth = get_sales_growth(latest_trading_date.strftime('%Y-%m-%d'), market_cap_on_current_day,recent_five_annual_shares, recent_report_type)

    earnings_growth = get_earnings_growth(latest_trading_date.strftime('%Y-%m-%d'), market_cap_on_current_day,recent_five_annual_shares, recent_report_type)

    atomic_descriptors_df = pd.concat([sales_growth, earnings_growth], axis=1)

    atomic_descriptors_df.columns = ['sales_growth', 'earnings_growth']

    atom_descriptors_weight = pd.Series(data=[0.47 / (0.47 + 0.24), 0.24 / (0.47 + 0.24)],index=['sales_growth', 'earnings_growth'])

    growth = atomic_descriptors_imputation_and_combination(atomic_descriptors_df, atom_descriptors_weight)

    processed_growth_exposure = winsorization_and_market_cap_weighed_standardization(growth, market_cap_on_current_day)

    return sales_growth, earnings_growth, processed_growth_exposure


# book-to-price = (股东权益合计-优先股)/市值

def get_book_to_price_ratio(market_cap_on_current_day, last_reported_preferred_stock, recent_report_type):

    last_reported_total_equity = get_last_reported_values(rqdatac.financials.balance_sheet.total_equity,recent_report_type)

    book_to_price_ratio = (last_reported_total_equity - last_reported_preferred_stock) / market_cap_on_current_day[last_reported_total_equity.index]

    processed_book_to_price_ratio = winsorization_and_market_cap_weighed_standardization(book_to_price_ratio,market_cap_on_current_day[last_reported_total_equity.index])

    return processed_book_to_price_ratio


def get_leverage(market_cap_on_current_day, last_reported_non_current_liabilities, last_reported_preferred_stock,recent_report_type):

    market_leverage = get_market_leverage(market_cap_on_current_day, last_reported_non_current_liabilities,last_reported_preferred_stock)

    debt_to_assets = get_debt_to_assets(market_cap_on_current_day, recent_report_type)

    book_leverage = get_book_leverage(market_cap_on_current_day, last_reported_non_current_liabilities,last_reported_preferred_stock, recent_report_type)

    atomic_descriptors_df = pd.concat([market_leverage, debt_to_assets, book_leverage], axis=1)

    atomic_descriptors_df.columns = ['market_leverage', 'debt_to_assets', 'book_leverage']

    atom_descriptors_weight = pd.Series(data=[0.38, 0.35, 0.27],index=['market_leverage', 'debt_to_assets', 'book_leverage'])

    leverage = atomic_descriptors_imputation_and_combination(atomic_descriptors_df, atom_descriptors_weight)

    processed_leverage_exposure = winsorization_and_market_cap_weighed_standardization(leverage,market_cap_on_current_day)

    return market_leverage, debt_to_assets, book_leverage, processed_leverage_exposure


def get_liquidity(stock_list, date, market_cap_on_current_day):

    trading_date_252_before = rqdatac.get_trading_dates(date - timedelta(days=500), date, country='cn')[-252]

    trading_volume = rqdatac.get_price(stock_list, trading_date_252_before, date, frequency='1d', fields='volume')

    inds = trading_volume.loc[date][trading_volume.loc[date].values == 0].index.tolist()

    stock_list = list(set(stock_list) - set(inds))

    outstanding_shares = rqdatac.get_shares(stock_list, trading_date_252_before, date, fields='total_a')

    daily_turnover_rate = trading_volume[stock_list].divide(outstanding_shares)

    # 对于对应时期内换手率为 0 的股票，其细分因子暴露度也设为0

    one_month_share_turnover = winsorization_and_market_cap_weighed_standardization(np.log(daily_turnover_rate.iloc[-21:].sum().replace(0, np.nan)), market_cap_on_current_day)

    three_months_share_turnover = winsorization_and_market_cap_weighed_standardization(np.log(daily_turnover_rate.iloc[-63:].sum().replace(0, np.nan) / 3), market_cap_on_current_day)

    twelve_months_share_turnover = winsorization_and_market_cap_weighed_standardization(np.log(daily_turnover_rate.iloc[-252:].sum().replace(0, np.nan) / 12), market_cap_on_current_day)

    atomic_descriptors_df = pd.concat([one_month_share_turnover, three_months_share_turnover, twelve_months_share_turnover], axis=1)

    atomic_descriptors_df.columns = ['one_month_share_turnover', 'three_months_share_turnover','twelve_months_share_turnover']

    atom_descriptors_weight = pd.Series(data=[0.35, 0.35, 0.3],index=['one_month_share_turnover', 'three_months_share_turnover','twelve_months_share_turnover'])

    liquidity = atomic_descriptors_imputation_and_combination(atomic_descriptors_df, atom_descriptors_weight)

    processed_liquidity = winsorization_and_market_cap_weighed_standardization(liquidity, market_cap_on_current_day)

    return one_month_share_turnover, three_months_share_turnover, twelve_months_share_turnover, processed_liquidity


def get_non_linear_size(size_exposure, market_cap_on_current_day):

    cubed_size = np.power(size_exposure, 3)

    #processed_cubed_size = winsorization_and_market_cap_weighed_standardization(cubed_size, market_cap_on_current_day)

    orthogonalized_cubed_size = orthogonalize(target_variable=cubed_size, reference_variable=size_exposure,regression_weight=np.sqrt(market_cap_on_current_day) / (np.sqrt(market_cap_on_current_day).sum()))

    processed_orthogonalized_cubed_size = winsorization_and_market_cap_weighed_standardization(orthogonalized_cubed_size, market_cap_on_current_day)

    return processed_orthogonalized_cubed_size


def get_style_factors(date):

    latest_trading_date = rqdatac.get_previous_trading_date(datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1))

    trading_date_252_before = rqdatac.get_trading_dates(latest_trading_date - timedelta(days=500), latest_trading_date, country='cn')[-252]

    stock_list = rqdatac.all_instruments(type='CS', date=latest_trading_date)['order_book_id'].values.tolist()

    ### 获取因子计算共用的行情数据和财务数据

    recent_report_type, annual_report_type, market_cap_on_current_day, \
    stock_excess_return, market_portfolio_excess_return, recent_five_annual_shares, \
    last_reported_non_current_liabilities, last_reported_preferred_stock = get_financial_and_market_data(stock_list,latest_trading_date,trading_date_252_before)

    # 风格因子计算

    size = get_size(market_cap_on_current_day)

    non_linear_size = get_non_linear_size(size, market_cap_on_current_day)

    market_portfolio_beta, market_portfolio_beta_exposure = get_market_portfolio_beta(stock_excess_return,market_portfolio_excess_return,market_cap_on_current_day)

    # 获取每只股票的 beta

    benchmark_list = ['000016.XSHG', '000300.XSHG', '000905.XSHG', '000906.XSHG', '000985.XSHG']

    stock_beta = pd.DataFrame()

    for benchmark in benchmark_list:
        stock_beta[benchmark] = get_stock_beta(stock_list, stock_excess_return, benchmark, latest_trading_date,market_cap_on_current_day)

    daily_standard_deviation, cumulative_range, historical_sigma, residual_volatility = get_residual_volatility(
        stock_list, latest_trading_date, stock_excess_return, market_portfolio_excess_return, market_cap_on_current_day,
        market_portfolio_beta_exposure, market_portfolio_beta)

    momentum = get_momentum(stock_list, latest_trading_date, market_cap_on_current_day)

    one_month_share_turnover, three_months_share_turnover, twelve_months_share_turnover, liquidity = get_liquidity(stock_list, latest_trading_date, market_cap_on_current_day)

    earnings_to_price_ratio, cash_earnings_to_price_ratio, earnings_yield = get_earnings_yield(latest_trading_date,market_cap_on_current_day,recent_report_type)

    book_to_price = get_book_to_price_ratio(market_cap_on_current_day, last_reported_preferred_stock,recent_report_type)

    market_leverage, debt_to_assets, book_leverage, leverage = get_leverage(market_cap_on_current_day,last_reported_non_current_liabilities,last_reported_preferred_stock,recent_report_type)

    sales_growth, earnings_growth, growth = get_growth(latest_trading_date, market_cap_on_current_day,recent_five_annual_shares, recent_report_type)

    style_factors_exposure = pd.concat([market_portfolio_beta_exposure, momentum, size, earnings_yield, residual_volatility, growth, book_to_price,leverage, liquidity, non_linear_size], axis=1)

    style_factors_exposure.columns = ['beta', 'momentum', 'size', 'earnings_yield', 'residual_volatility', 'growth','book_to_price', 'leverage', 'liquidity', 'non_linear_size']

    atomic_descriptors_exposure = pd.concat(
        [daily_standard_deviation, cumulative_range, historical_sigma, one_month_share_turnover,
         three_months_share_turnover, twelve_months_share_turnover, \
         earnings_to_price_ratio, cash_earnings_to_price_ratio, market_leverage, debt_to_assets, book_leverage,
         sales_growth, earnings_growth], axis=1)

    atomic_descriptors_exposure.columns = ['daily_standard_deviation', 'cumulative_range', 'historical_sigma',
                                           'one_month_share_turnover', 'three_months_share_turnover',
                                           'twelve_months_share_turnover', \
                                           'earnings_to_price_ratio', 'cash_earnings_to_price_ratio', 'market_leverage',
                                           'debt_to_assets', 'book_leverage', 'sales_growth', 'earnings_growth']

    # 提取财务数据的时候，会提取当前全市场股票的数据，因此 dataframe 中可能包含计算日期未上市的股票，需要对 style_factors_exposure 取子集

    atomic_descriptors_exposure = atomic_descriptors_exposure.loc[stock_list]

    style_factors_exposure = style_factors_exposure.loc[stock_list]

    # 用回归方法处理细分因子的缺失值

    imputed_atomic_descriptors = pd.DataFrame()

    for atomic_descriptor in atomic_descriptors_exposure.columns:
        imputed_atomic_descriptors[atomic_descriptor] = individual_factor_imputation(stock_list, atomic_descriptors_exposure[atomic_descriptor], market_cap_on_current_day,latest_trading_date.strftime('%Y-%m-%d'))

    # 用回归方法处理风格因子暴露度的缺失值

    imputed_style_factors_exposure = style_factors_imputation(style_factors_exposure, market_cap_on_current_day,latest_trading_date.strftime('%Y-%m-%d'))

    # 若经过缺失值处理后因子暴露度依旧存在缺失值，使用全市场股票进行回归，填补缺失值

    if imputed_style_factors_exposure.isnull().sum().sum() > 0:

        imputed_style_factors_exposure = factor_imputation(market_cap_on_current_day,imputed_style_factors_exposure)

    if imputed_atomic_descriptors.isnull().sum().sum() > 0:

        imputed_atomic_descriptors = factor_imputation(market_cap_on_current_day,imputed_atomic_descriptors)

    if stock_beta.isnull().sum().sum() > 0:

        stock_beta = factor_imputation(market_cap_on_current_day, stock_beta)

    return imputed_atomic_descriptors, imputed_style_factors_exposure, stock_beta
