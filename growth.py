import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
from sklearn import linear_model


import rqdatac

# rqdatac.init("ricequant", "Ricequant123", ('rqdatad-pro.ricequant.com', 16004))
rqdatac.init('ricequant', '8ricequant8',('q-tools.ricequant.com', 16010))


def recent_annual_report(date):
    latest_trading_date = str(
        rqdatac.get_previous_trading_date(datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)))

    previous_year = datetime.strptime(latest_trading_date, '%Y-%m-%d').year - 1

    # 取出最近一期财务报告类型，例如 '2016q3' 或  '2016q4'， 其中 '2016q3' 表示前三季度累计； '2016q4' 表示年报

    recent_report_type = rqdatac.get_fundamentals(rqdatac.query(rqdatac.fundamentals.income_statement.net_profit),
                                                  entry_date=latest_trading_date, interval='1y', report_quarter=True)[
        'report_quarter']

    annual_report_type = recent_report_type.copy()  # 深拷贝

    # recent_report_type 和 annual_report_type 均为 dataframe 格式，输出时转为 Series 格式

    # 若上市公司未发布今年的财报，且未发布去年的年报，则取前年的年报为最新年报

    if recent_report_type.T.iloc[0].values[0][:4] == str(previous_year):

        annual_report_type[annual_report_type != str(previous_year) + 'q4'] = str(previous_year - 1) + 'q4'

        # 若上市公司已发布今年的财报，则取去年的年报为最新年报

    else:
        annual_report_type[annual_report_type != str(previous_year) + 'q4'] = str(previous_year) + 'q4'

    annual_report_type_last_year = annual_report_type.T[latest_trading_date].copy()
    annual_report_type_2_year_ago = annual_report_type.T[latest_trading_date].copy()
    annual_report_type_3_year_ago = annual_report_type.T[latest_trading_date].copy()
    annual_report_type_4_year_ago = annual_report_type.T[latest_trading_date].copy()

    for stock in annual_report_type.T.index.tolist():
        if annual_report_type.T[latest_trading_date][stock][:4] == str(previous_year):
            annual_report_type_last_year[stock] = str(previous_year - 1) + 'q4'
            annual_report_type_2_year_ago[stock] = str(previous_year - 2) + 'q4'
            annual_report_type_3_year_ago[stock] = str(previous_year - 3) + 'q4'
            annual_report_type_4_year_ago[stock] = str(previous_year - 4) + 'q4'
        else:
            annual_report_type_last_year[stock] = str(previous_year - 2) + 'q4'
            annual_report_type_2_year_ago[stock] = str(previous_year - 3) + 'q4'
            annual_report_type_3_year_ago[stock] = str(previous_year - 4) + 'q4'
            annual_report_type_4_year_ago[stock] = str(previous_year - 5) + 'q4'

    return recent_report_type.T[latest_trading_date], annual_report_type.T[latest_trading_date], \
        annual_report_type_last_year, annual_report_type_2_year_ago, annual_report_type_3_year_ago, annual_report_type_4_year_ago


def sales_growth(date):

    recent_report,annual_report,annual_report_last_year,annual_report_2_year_ago,annual_report_3_year_ago,annual_report_4_year_ago = recent_annual_report(date)
    growth_listed_date_threshold = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1825)).strftime("%Y-%m-%d")
    growth_qualified_stocks = [i for i in annual_report.index.tolist() if rqdatac.instruments(i).listed_date < growth_listed_date_threshold]

    factor = pd.DataFrame(index=growth_qualified_stocks)

    for stock in growth_qualified_stocks:
        query = rqdatac.query(rqdatac.financials.income_statement.operating_revenue).filter(
            rqdatac.financials.stockcode.in_([stock]))
        sales_recent = rqdatac.get_financials(query,annual_report[stock],'1q')

        latest_trading_date_recent = str(
            rqdatac.get_previous_trading_date(datetime.strptime(annual_report[stock][:4]+'-12-31', '%Y-%m-%d') + timedelta(days=1)))

        shares_recent = rqdatac.get_shares(stock,start_date=latest_trading_date_recent,
                                           end_date=latest_trading_date_recent,fields='total')

        sales_per_share_recent = sales_recent.values/shares_recent.values

        sales_last_year = rqdatac.get_financials(query, annual_report_last_year[stock], '1q')

        latest_trading_date_last_year = str(
            rqdatac.get_previous_trading_date(
                datetime.strptime(annual_report_last_year[stock][:4] + '-12-31', '%Y-%m-%d') + timedelta(days=1)))

        shares_last_year = rqdatac.get_shares(stock, start_date=latest_trading_date_last_year,
                                              end_date=latest_trading_date_last_year, fields='total')

        sales_per_share_last_year = sales_last_year.values / shares_last_year.values
        
        sales_2_year_ago = rqdatac.get_financials(query, annual_report_2_year_ago[stock], '1q')

        latest_trading_date_2_year_ago = str(
            rqdatac.get_previous_trading_date(
                datetime.strptime(annual_report_2_year_ago[stock][:4] + '-12-31', '%Y-%m-%d') + timedelta(days=1)))

        shares_2_year_ago = rqdatac.get_shares(stock, start_date=latest_trading_date_2_year_ago,
                                               end_date=latest_trading_date_2_year_ago, fields='total')

        sales_per_share_2_year_ago = sales_2_year_ago.values / shares_2_year_ago.values

        sales_3_year_ago = rqdatac.get_financials(query, annual_report_3_year_ago[stock], '1q')

        latest_trading_date_3_year_ago = str(
            rqdatac.get_previous_trading_date(
                datetime.strptime(annual_report_3_year_ago[stock][:4] + '-12-31', '%Y-%m-%d') + timedelta(days=1)))

        shares_3_year_ago = rqdatac.get_shares(stock, start_date=latest_trading_date_3_year_ago,
                                               end_date=latest_trading_date_3_year_ago, fields='total')

        sales_per_share_3_year_ago = sales_3_year_ago.values / shares_3_year_ago.values

        sales_4_year_ago = rqdatac.get_financials(query, annual_report_4_year_ago[stock], '1q')

        latest_trading_date_4_year_ago = str(
            rqdatac.get_previous_trading_date(
                datetime.strptime(annual_report_4_year_ago[stock][:4] + '-12-31', '%Y-%m-%d') + timedelta(days=1)))

        shares_4_year_ago = rqdatac.get_shares(stock, start_date=latest_trading_date_4_year_ago,
                                               end_date=latest_trading_date_4_year_ago, fields='total')

        sales_per_share_4_year_ago = sales_4_year_ago.values / shares_4_year_ago.values

        regression = linear_model.LinearRegression()
        sales_per_share = np.array(
            [sales_per_share_recent, sales_per_share_last_year, sales_per_share_2_year_ago, sales_per_share_3_year_ago,
             sales_per_share_4_year_ago])
        regression.fit(sales_per_share.reshape(-1, 1), ([4, 3, 2, 1, 0]))
        factor[stock] = float(regression.coef_)/sales_per_share.mean()

    return factor


def earnings_growth(date):
    recent_report,annual_report,annual_report_last_year,annual_report_2_year_ago,annual_report_3_year_ago,annual_report_4_year_ago = recent_annual_report(date)
    growth_listed_date_threshold = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1825)).strftime("%Y-%m-%d")
    growth_qualified_stocks = [i for i in annual_report.index.tolist() if rqdatac.instruments(i).listed_date < growth_listed_date_threshold]

    factor = pd.DataFrame(index=growth_qualified_stocks)

    for stock in growth_qualified_stocks:
        query = rqdatac.query(rqdatac.financials.financial_indicator.earnings_per_share).filter(rqdatac.financials.stockcode.in_([stock]))
        eps_recent = rqdatac.get_financials(query, annual_report[stock], '1q')
        eps_last_year = rqdatac.get_financials(query, annual_report_last_year, '1q')
        eps_2_year_ago = rqdatac.get_financials(query, annual_report_2_year_ago, '1q')
        eps_3_year_ago = rqdatac.get_financials(query, annual_report_3_year_ago, '1q')
        eps_4_year_ago = rqdatac.get_financials(query, annual_report_4_year_ago, '1q')

        regression = linear_model.LinearRegression()
        eps = np.array(
            [eps_recent, eps_last_year, eps_2_year_ago, eps_3_year_ago,eps_4_year_ago])
        regression.fit(eps.reshape(-1, 1), ([4, 3, 2, 1, 0]))
        factor[stock] = float(regression.coef_ )/ eps.mean()

    return factor



date = '2018-02-02'
sale_grow = sales_growth(date)
earning_grow = earnings_growth(date)
growth = pd.concat([sale_grow,earning_grow],axis=1)
growth.to_csv('/Users/rice/Desktop/Barr factor/growth.csv', index=True, na_rep='NaN', header=True)


