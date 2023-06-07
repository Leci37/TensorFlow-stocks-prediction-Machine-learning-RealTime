# https://stackoverflow.com/questions/38917958/convert-html-into-csv
# import NumPy as py
import investpy
import pandas as pd
import numpy as np

from Utils import UtilsL, Utils_Yfinance
from datetime import datetime
from LogRoot.Logging import Logger
from lxml.etree import tostring

FINANCIAL_SUMMARY_PERIODS = {"annual": "Annual", "quarterly": "Interim"}
FINALCIAL_TYPE_SHEET = {
    "INC": "Income Statement",
    "BAL": "Balance Sheet",
    "CAS" : "Cash Flow"
}

# url = 'https://www.investing.com/equities/amazon-com-inc-income-statement'
# https://www.investing.com/equities/amazon-com-inc-income-statement&period_type=Annual



df_summary = None


def __get_tr_rows_info_from_investing(full_html):
    '''
    Recibe los mini df que corresponde a cada columana y los extrae los datos
    :param df_peri_time:
    :return:
    '''
    df_d = UtilsL.get_trs_dataframe_rev(full_html, delete_emply_column_name_Unnamed= True)
    #df_d = df_d[~df_d.iloc[:,0].astype(str).str.contains(r"\\t|\\n|\\r", "\t|\n|\r", regex=True)]#df.ix[:,0] primera columna
    #df_d['PeriodEnd'] = pd.to_datetime(df_d['PeriodEnd'], format="%m/%Y").dt.strftime("%Y-%m-%d")
    col = []
    col.append(df_d.columns[0])
    col = col + [datetime.strptime(d, "%Y%d/%m").strftime("%Y-%m-%d") for d in df_d.columns[1:]]
    df_d.columns = col
    Logger.logr.debug(" Get data info number shape: " + str(df_d.shape))
    return df_d
    print(df_d)
    df_peri_time = None

    trs = full_html.xpath('//tr')
    for t in trs:
        # eliminar las sub tables solo queremos las principal de la serie de tablas
        for elem in t.xpath('//table/tbody/tr'):
            elem.getparent().remove(elem)
        inner_html = tostring(t)
        inner_html = "<table>" + str(inner_html) + "</table>"
        df_table = pd.read_html(inner_html)[0]
        if df_peri_time is None:
            df_peri_time = df_table
            col.append(df_peri_time.columns[0])
            col = col + [datetime.strptime(d, "%Y%d/%m").strftime("%Y-%m-%d") for d in df_peri_time.columns[1:]]
            df_peri_time.columns = col
        else:
            #df_table.columns = col
            df_peri_time = pd.concat(
                [df_peri_time, df_table])  # pd.merge(df, df_table,  how='outer')#on=['Date'],
            df_peri_time = df_peri_time.replace('', np.nan)
            df_peri_time = df_peri_time.replace('-', np.nan)
            df_peri_time = df_peri_time.dropna(how='any')

    return df_peri_time



def __get_df_financial_full_RAW(pair_ID):
    global df_summary
    for report_type in FINALCIAL_TYPE_SHEET.keys():  # Income Statement, Balance Sheet
        for f in FINANCIAL_SUMMARY_PERIODS.values():
            url = "https://www.investing.com/instruments/Financials/changereporttypeajax?action=change_report_type&pair_ID=" + str(
                pair_ID) + "&report_type=" + report_type + "&period_type=" + f

            "https://www.investing.com/instruments/Financials/changereporttypeajax?action=change_report_type&pair_ID=6435&report_type=CAS&period_type=Annual"
            #print(url)
            full_html = Utils_Yfinance.get_GET_root_from_url(url)

            if full_html is None:
                Logger.logr.debug(" resquest is NONE url: " + url)
                continue
            if len(full_html.xpath('//table/tbody/tr')) == 0:
                Logger.logr.debug(" resquest is does have data tr url: " + url)
                continue
            if report_type == "CAS": #Clean duplicates "th"
                for elem in full_html.xpath('//th/..')[1:]: #ignore the first position    #full_html.xpath('//table/tbody/tr[position()>1]'):
                   for e in elem:
                       if e.tag == "th":
                           e.tag = "td"

            df_periodo_time = None
            df_periodo_time = __get_tr_rows_info_from_investing(full_html)

            if df_summary is None:
                df_summary = df_periodo_time
            else:
                df_summary = pd.merge(df_summary, df_periodo_time,  how='outer')
    return df_summary
            #print(df_periodo_time.columns)



def get_df_financial_full_date_axis(stockid, country="united states"):

    try:
        pair_ID = investpy.stocks.get_stock_id_investing(country, stockid)
        Logger.logr.debug(
            "Looking up the full finance data Investing  stock : " + stockid + " country: " + country + " pair_ID: " + str(
                pair_ID))

        df_financial = __get_df_financial_full_RAW(pair_ID)
        if df_financial is None:
            Logger.logr.warning("No financial data investing in stock : " + stockid + " country: " + country)
            return None
        df_financial = df_financial.dropna(how='all')
        df_financial = UtilsL.remove_chars_in_columns(df_financial, "Period Ending:")
        df_financial = df_financial.drop_duplicates(subset=["Period Ending:"])
        df_financial = df_financial.drop_duplicates()
        df_financial = df_financial.loc[:, ~df_financial.columns.duplicated()]#retirar la segunga columna duplicada

        df_financial = df_financial.rename_axis(None)
        df_financial.set_index(df_financial.columns[0], inplace=True)
        df_financial = df_financial.T  # You can use df = df.T to transpose the dataframe.
        #df_financial.index.name = "Date"
        df_financial['Date'] = df_financial.index
        df_financial.reset_index(drop=True, inplace=True)
        #df_financial.insert(0, 'Ticker', stockid)
        #df_financial = df_financial.rename(columns={df_financial.columns[0]: "Date"})

        #df_financial["Date"] = df_financial["Date"].astype(str)

        df_financial = df_financial.sort_values('Date', ascending=False)
        Logger.logr.info("Found financial data investing.com in stock : " + stockid + " columnsNumbers : " + str(
            len(df_financial.columns)) + " country: " + country)
        return df_financial
    except Exception as e:
        Logger.logr.warn(" GET financial: " + stockid + "Exception :" + str(e))
    return None


# stockid = "TWLO" #"VWDRY" # "MELI" #"VWDRY"
# country = "united states"
#
# df = get_df_financial_full_date_axis(stockid, country)
# df.to_csv( str(stockid) + "_financial_inv_summary.csv", sep="\t")
# Logger.logr.debug(" Generated File info: " +  str(stockid) + "_financial_inv_summary.csv")