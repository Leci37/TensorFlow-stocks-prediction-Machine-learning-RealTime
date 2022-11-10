import pandas as pd

from Utils import UtilsL, Utils_Yfinance
from LogRoot.Logging import Logger

df_ratios = None

def __get_tr_rows_info_from_investing_ratios(full_html):
    '''
    Recibe los mini df que corresponde a cada columana y los extrae los datos
    :param df_ratios:
    :return:
    '''
    df_r = UtilsL.get_trs_dataframe(full_html)
    df_r = df_r[~df_r["Name"].astype(str).str.contains(r"\\t|\\n|\\r", "\t|\n|\r",regex=True)]#clean de df moving al \n elements rows

    df_r.loc[df_r['Company'].str.contains("%"), 'Name'] += "_per"  # df.loc[m, 'cola'] = df.loc[m, 'cola'] + 20
    df_r = UtilsL.remove_chars_in_columns(df_r, "Name")
    df_r = UtilsL.clean_float_columns(df_r, "Company")
    df_r = UtilsL.clean_float_columns(df_r, "Industry")

    return df_r


def get_df_financial_ratios(stockid, country="US"):
    global df_ratios
    try:
        #url = "https://www.investing.com/equities/apple-computer-inc-ratios"
        url  = UtilsL.Url_stocks_pd.get_url(stockid, country) + "-ratios"
        Logger.logr.info("get_df_financial_ratios: " + stockid + "Url: " + url)
        full_html = Utils_Yfinance.get_GET_root_from_url(url)
        full_html = full_html.xpath('/html/body/div/section/table[@class="genTbl reportTbl ratioTable"]')[0]

        if full_html is None:
            Logger.logr.debug(" resquest is NONE url: " + url)
            return None
        if len(full_html.xpath('//table/tbody/tr')) == 0:
            Logger.logr.debug(" resquest is does have data tr url: " + url)
            return None

        df_r = None
        df_r = __get_tr_rows_info_from_investing_ratios(full_html)

        if df_ratios is None:
            df_ratios = df_r
        else:
            df_ratios = pd.merge(df_ratios, df_r, how='outer')
        return df_ratios
    except Exception as e:
        Logger.logr.warn(" GET financial: " + stockid + "Exception :" + str(e))
    return None



# stockid = "RIVN" # "VWDRY" # "MELI" #"VWDRY"
# country = "united states"
#
# df = get_df_financial_ratios(stockid)
# df.to_csv( str(stockid) + "_financial_inv_ratios.csv", sep="\t")
# Logger.logr.debug(" Generated File info: " +  str(stockid) + "_financial_inv_ratios.csv")