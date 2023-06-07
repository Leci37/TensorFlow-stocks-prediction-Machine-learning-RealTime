import investpy
import pandas as pd

from Utils import UtilsL, Utils_Yfinance
from LogRoot.Logging import Logger

def get_wDate_summary_MRQ_TTM(stockid, country):
    pair_ID = investpy.stocks.get_stock_id_investing(country, stockid)
    #apple 6408
    #"https://www.investing.com/instruments/Financials/changesummaryreporttypeajax?action=change_report_type&pid=6408&financial_id=6408&ratios_id=6408&period_type=Annual"
    #"https://www.investing.com/instruments/Financials/changesummaryreporttypeajax?action=change_report_type&pid=6408&financial_id=6408&ratios_id=6408&period_type=Interim"
    url = "https://www.investing.com/instruments/Financials/changesummaryreporttypeajax?action=change_report_type&pid=" + str(
                pair_ID) + "&financial_id=" + str(
                pair_ID) + "&ratios_id=" + str(
                pair_ID) + "&period_type=Annual"
    print(url)
    full_html = Utils_Yfinance.get_GET_root_from_url(url)
        #"/html/body/div[1]/div[1]/div[1]"
    #
    if full_html is None:
        Logger.logr.debug(" resquest is NONE url: " + url)
        return None
    if len(full_html.xpath('//table/tbody/tr')) == 0:
        Logger.logr.debug(" resquest is does have data tr url: " + url)
        return None
    div_infos_lines = full_html.xpath("//div[contains(@class, \"info float_lang_base_2\")]/div[contains(@class, \"infoLine\")]")
    list_name = []
    list_value = []
    for d in div_infos_lines: # [['aaaaaa' if x is None else x for x in c] for c in d] map(lambda x: '' if x == None else x, d)
        header = ""
        value = ""
        if d[0] is not None and d[0].text is not None:
            header += d[0].text
        if d[1] is not None and d[1].text is not None:
            header += d[1].text
        if d[2] is not None and d[2].text is not None:
            value  += d[2].text
        list_name.append(header)
        list_value.append(value)

    df_f = pd.DataFrame(columns=['fNames', 'fValue'])
    df_f['fNames'] = list_name
    df_f['fValue'] = list_value
    df_f = UtilsL.remove_chars_in_columns(df_f, "fNames")

    df_f.loc[df_f['fValue'].str.contains("%"), 'fNames'] += "_per" #df.loc[m, 'cola'] = df.loc[m, 'cola'] + 20
    df_f['fValue'] = df_f['fValue'].map(lambda x: x.replace("%", "")).astype(float)
    Logger.logr.info("investpy.stocks.get_stock_id_investing Stock: "+ stockid+ " Numbers of dates: "+ str(len(df_f.index)))
    return df_f


# stockid = "MELI" #"VWDRY" # "MELI" #"VWDRY"
# country = "united states"

#df = get_wDate_summary_MRQ_TTM(stockid, country)
#df.to_csv( str(stockid) + "_financial_inv_wDate_summary.csv", sep="\t")
#Logger.logr.debug(" Generated File info: " +  str(stockid) + "_financial_inv_wDate_summary.csv")