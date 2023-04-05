import pandas as pd

from Utils import UtilsL, Utils_Yfinance
from LogRoot.Logging import Logger


df_divid = None

def __get_tr_rows_info_from_investing_dividends(full_html):
    '''
    Recibe los mini df que corresponde a cada columana y los extrae los datos
    :param df_ratios:
    :return:
    '''
    df_d = UtilsL.get_trs_dataframe(full_html)

    df_d = df_d.drop(columns=['Type'],errors='ignore')
    df_d.columns = df_d.columns.str.replace(' Date', '_Date').str.replace('-', '')
    df_d.columns = df_d.columns.str.replace('Yield', 'Yield_per_div')
    df_d['Yield_per_div'] = df_d['Yield_per_div'].map(lambda x: x.replace("%", "")).astype(float)
    df_d['ExDividend_Date'] = pd.to_datetime(df_d['ExDividend_Date'], format="%b %d, %Y").dt.strftime("%Y-%m-%d")#"Apr 22, 2022"
    df_d['Payment_Date'] = pd.to_datetime(df_d['Payment_Date'], format="%b %d, %Y").dt.strftime("%Y-%m-%d")

    return df_d


def get_df_financial_dividends(stockid,country="US"):
    global df_divid
    try:
        #url = "https://www.investing.com/equities/apple-computer-inc-ratios"
        url  = UtilsL.Url_stocks_pd.get_url(stockid, country) + "-dividends"
        Logger.logr.info("get_df_financial_dividends: " + stockid + "Url: " + url)
        full_html = Utils_Yfinance.get_GET_root_from_url(url)
        full_html = full_html.xpath('/html/body/div/section/table[@class="genTbl closedTbl dividendTbl"]')[0]

        if full_html is None:
            Logger.logr.debug(" resquest is NONE url: " + url)
            return None
        if len(full_html.xpath('//table/tbody/tr')) == 0:
            Logger.logr.debug(" resquest is does have data tr url: " + url)
            return None

        df_r = None
        df_r = __get_tr_rows_info_from_investing_dividends(full_html)

        if df_divid is None:
            df_divid = df_r
        else:
            df_divid = pd.merge(df_divid, df_r, how='outer')
        return df_divid
    except Exception as e:
        Logger.logr.warn(" GET financial: " + stockid + "Exception :" + str(e))
    return None



# stockid = "MELI" #"VWDRY" # "MELI" #"VWDRY"
# country = "united states"
#
# df = get_df_financial_dividends(stockid)
# df.to_csv( str(stockid) + "_financial_inv_dividends.csv", sep="\t")
# Logger.logr.debug(" Generated File info: " +  str(stockid) + "_financial_inv_dividends.csv.csv")
