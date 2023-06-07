import pandas as pd

from Utils import UtilsL, Utils_Yfinance
from LogRoot.Logging import Logger

df_earnings = None


def __get_tr_rows_info_from_investing_earnings(full_html):
    '''
    Recibe los mini df que corresponde a cada columana y los extrae los datos
    :param df_ratios:
    :return:
    '''
    df_d = UtilsL.get_trs_dataframe(full_html)
    df_d.columns = df_d.columns.str.replace(' Date', '_Date').str.replace('-', '').str.replace('/', '').str.replace(' ',
                                                                                                                    '').str.replace(
        '.', '_')
    df_d['Release_Date'] = pd.to_datetime(df_d['Release_Date'], format="%b %d, %Y").dt.strftime(
        "%Y-%m-%d")  # "Apr 22, 2022"
    df_d['PeriodEnd'] = pd.to_datetime(df_d['PeriodEnd'], format="%m/%Y").dt.strftime("%Y-%m-%d")

    df_d = df_d.replace(['/'], [''], regex=True)
    df_d = UtilsL.clean_float_columns(df_d, "Forecast_1")
    df_d = UtilsL.clean_float_columns(df_d, "EPS")
    df_d = UtilsL.clean_float_columns(df_d, "Forecast")
    df_d = UtilsL.clean_float_columns(df_d, "Revenue")
    # df_d['Payment_Date'] = pd.to_datetime(df_d['Payment_Date'], format="%b %d, %Y").dt.strftime("%Y-%m-%d")
    df_d.columns = df_d.columns + "_ear"
    return df_d


def get_df_financial_earnings(stockid,country="US"):
    global df_earnings
    try:
        # url = "https://www.investing.com/equities/apple-computer-inc-ratios"
        url = UtilsL.Url_stocks_pd.get_url(stockid, country) + "-earnings"
        Logger.logr.info("get_df_financial_dividends: " + stockid + "Url: " + url)
        full_html = Utils_Yfinance.get_GET_root_from_url(url)  # InvestingGetWebInfo.get_root_from_url(url)

        full_html = \
        full_html.xpath('/html/body/div/section/table[@class="genTbl openTbl ecoCalTbl earnings earningsPageTbl"]')[0]

        if full_html is None:
            Logger.logr.debug(" resquest is NONE url: " + url)
            return None
        if len(full_html.xpath('//table/tbody/tr')) == 0:
            Logger.logr.debug(" resquest is does have data tr url: " + url)
            return None

        df_r = None
        df_r = __get_tr_rows_info_from_investing_earnings(full_html)

        if df_earnings is None:
            df_earnings = df_r
        else:
            df_earnings = pd.merge(df_earnings, df_r, how='outer')
        return df_earnings
    except Exception as e:
        Logger.logr.warn(" GET financial: " + stockid + "Exception :" + str(e))
    return None

# stockid = "MELI"  # "VWDRY" # "MELI" #"VWDRY"
# country = "united states"
#
# df = get_df_financial_earnings(stockid)
# df.to_csv(str(stockid) + "_financial_inv_earn.csv", sep="\t")
# Logger.logr.debug(" Generated File info: " + str(stockid) + "_financial_inv_earn.csv")
