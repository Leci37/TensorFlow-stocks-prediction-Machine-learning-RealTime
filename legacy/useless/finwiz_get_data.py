from Utils import UtilsL, Utils_Yfinance
from LogRoot.Logging import Logger
import pandas as pd

FINWIZ_URL = 'https://finviz.com/quote.ashx?t='
news_tables = {}


def get_data_finviz(stockid: str):
    '''
    va a la web de noticias y recoge las noticias segun la Stock_ID
    lo devuelve en forma de DataSet
    :param stock_id:
    :return:
    '''
    try:
        url = FINWIZ_URL + stockid
        Logger.logr.info("get data_finviz : " + stockid + " Url: " + url)
        full_html = Utils_Yfinance.get_GET_root_from_url(url)
        full_html = full_html.xpath('//div[@class="fv-container"]/table[@class="snapshot-table2"]')[0]

        if full_html is None:
            Logger.logr.debug(" resquest is NONE url: " + url)
            return None
        if len(full_html.xpath('//table')) == 0:
            Logger.logr.debug(" resquest is does have data tr url: " + url)
            return None


        df_d = UtilsL.get_trs_dataframe(full_html)
        #df_d.cols = ["A","B","3","4","5","6","7","8","9","10","11"]
        df_1 = df_d.rename(columns={df_d.columns[0]: "key", df_d.columns[1]: "value"})[['key', 'value']]
        df_2 = df_d.rename(columns={df_d.columns[2]: "key", df_d.columns[3]: "value"})[['key', 'value']]
        df_3 = df_d.rename(columns={df_d.columns[4]: "key", df_d.columns[5]: "value"})[['key', 'value']]
        df_4 = df_d.rename(columns={df_d.columns[6]: "key", df_d.columns[7]: "value"})[['key', 'value']]
        df_5 = df_d.rename(columns={df_d.columns[8]: "key", df_d.columns[9]: "value"})[['key', 'value']]
        df_6 = df_d.rename(columns={df_d.columns[10]: "key", df_d.columns[11]: "value"})[['key', 'value']]

        pdList = [df_1, df_2, df_3,df_4,df_5,df_6 ]  # List of your dataframes
        df_sal = pd.concat(pdList)
        df_sal = df_sal.rename(columns={"value": "finviz_data"})
        df_sal.loc[df_sal['finviz_data'].str.contains("%"), 'key'] += "_per"
        df_sal = UtilsL.remove_chars_in_columns(df_sal, 'key')
        df_sal = df_sal.drop(df_sal[(df_sal.key_model == "Earnings")].index)#remove beacuse is string
        df_sal = UtilsL.clean_float_columns(df_sal, "finviz_data")
        df_sal['key'] = "finviz_"+df_sal['key']
        return  df_sal
    except Exception as e:
        Logger.logr.warning(" GET finviz.com financial: " + stockid + "Exception :" + str(e))
    return None



list_company = ["RIVN", "VWDRY", "TWLO", "OB","ASML","SNOW","ADBE","LYFT","UBER","ZI","BXP","ANET","MITK","QCOM","PYPL","JBLU","IAG.L","NATGAS","GE","SPOT","F","SAN.MC","TMUS","MBG.DE","INTC","TRIG.L",
                "UBSG.ZU","NDA.DE","TWTR","ITX.MC","PFE","FER.MC","AA","ABBN.ZU","RUN","IBE.MC","ESP35","BAYN.DE","GTLB","IBM","NESN.ZU","MDB","NVDA","CSCO","AMD","ADSK","AMZN",
                "RR.L","BABA","MBT","AAPL","NFLX","BA","VWS.CO","FFIV","GOOG","MSFT","AIR.PA","ABNB","BTC","TSLA","FB","REP.MC"]

df = None
pd.set_option('mode.chained_assignment', None)


def get_json_finwiz_data(stockid):
    df = get_data_finviz(stockid)
    if df is None:
        Logger.logr.warn(" NOT data fount Generated finviz.com data  info: " + str(stockid))
    else:
        df_csv = df
        df_csv['Date'] = pd.Timestamp("today").strftime("%Y-%m-%d")
        df_csv.to_csv("d_info_profile/" + stockid + "_finviz_data.csv", sep="\t")
        Logger.logr.debug(" Generated File info: d_info_profile/" + str(stockid) + "_financial_inv_dividends.csv")

        df.reset_index(drop=True, inplace=True)
        dict_json = df.set_index('key').T.to_dict('index')  # df.to_dict('records') .to_dict('index'

        dict_json = {
            pd.Timestamp("today").strftime("%Y-%m-%d"):dict_json
        }
        import json
        with open("d_info_profile/" + stockid + "_finviz_data.json", 'w') as fp:
            json.dump(dict_json, fp)
        Logger.logr.debug(" Generated File info: d_info_profile/" + stockid + "_finviz_data.json")

#get_json_finwiz_data("BA")
# for stockid in list_company:
#     get_json_finwiz_data(stockid)



