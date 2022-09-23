import pandas as pd
import numpy as np
import re
from lxml.etree import tostring
import math

from LogRoot.Logging import Logger

DF_URLS_STOCKS = pd.read_csv('URL_Stocks_DONTDEL.csv', sep='|')

ALL_COLUMNS_NAME = ["Date", "Open", "High", "Low", "Close", "Volume", "per_Close", "per_Volume", "buy_sell_point",
                    "has_preMarket", "per_preMarket",

                    "olap_BBAND_UPPER", "olap_BBAND_MIDDLE", "olap_BBAND_LOWER","olap_BBAND_UPPER_crash", "olap_BBAND_LOWER_crash",  "olap_HT_TRENDLINE", "olap_MIDPOINT",
                    "olap_MIDPRICE", "olap_SAR", "olap_SAREXT", "olap_ALMA_10_60_085", "olap_HWMA_02_01_01",
                    "olap_JMA_7_0", "olap_MCGD_10", "olap_PWMA_10", "olap_SINWMA_14", "olap_SSF_10_2", "olap_SWMA_10",
                    "olap_VWMA_10",

                    "mtum_ADX", "mtum_ADXR", "mtum_APO", "mtum_AROON_down", "mtum_AROON_up", "mtum_AROONOSC",
                    "mtum_BOP", "mtum_CCI", "mtum_CMO", "mtum_DX", "mtum_MACD", "mtum_MACD_signal", "mtum_MACD_list","mtum_MACD_crash",
                    "mtum_MACD_ext", "mtum_MACD_ext_signal", "mtum_MACD_ext_list","mtum_MACD_ext_crash", "mtum_MACD_fix",
                    "mtum_MACD_fix_signal", "mtum_MACD_fix_list", "mtum_MACD_fix_crash", "mtum_MFI", "mtum_MINUS_DI", "mtum_MINUS_DM",
                    "mtum_MOM", "mtum_PLUS_DI", "mtum_PLUS_DM", "mtum_PPO", "mtum_ROC", "mtum_ROCP", "mtum_ROCR",
                    "mtum_ROCR100", "mtum_RSI", "mtum_STOCH_k", "mtum_STOCH_d", "mtum_STOCH_kd","mtum_STOCH_crash", "mtum_STOCH_Fa_k",
                    "mtum_STOCH_Fa_d", "mtum_STOCH_Fa_kd","mtum_STOCH_Fa_crash", "mtum_STOCH_RSI_k", "mtum_STOCH_RSI_d", "mtum_STOCH_RSI_kd","mtum_STOCH_RSI_crash",
                    "mtum_TRIX", "mtum_ULTOSC", "mtum_WILLIAMS_R", "mtum_AO_5_34", "mtum_BIAS_SMA_26", "mtum_AR_26",
                    "mtum_BR_26", "mtum_CFO_9", "mtum_CG_10", "mtum_CTI_12", "mtum_DMP_14", "mtum_DMN_14", "mtum_ER_10",
                    "mtum_BULLP_13", "mtum_BEARP_13", "mtum_FISHERT_9_1", "mtum_FISHERTs_9_1", "mtum_INERTIA_20_14",
                    "mtum_K_9_3", "mtum_D_9_3", "mtum_J_9_3", "mtum_PGO_14", "mtum_PSL_12", "mtum_PVO_12_26_9",
                    "mtum_PVOh_12_26_9", "mtum_PVOs_12_26_9", "mtum_QQE_14_5_4236_RSIMA", "mtum_QQEl_14_5_4236",
                    "mtum_QQEs_14_5_4236", "mtum_RSX_14", "mtum_STC_10_12_26_05", "mtum_STCmacd_10_12_26_05",
                    "mtum_STCstoch_10_12_26_05", "mtum_SMI_5_20_5", "mtum_SMIs_5_20_5",
                    "mtum_SMIo_5_20_5", "mtum_murrey_math", "mtum_td_seq", "mtum_td_seq_sig",

                    "perf_CUMLOGRET_1", "perf_CUMPCTRET_1", "perf_z_30_1", "perf_ha", "tend_LDECAY_5", "tend_PSARl_002_02",
                    "tend_PSARs_002_02", "tend_PSARaf_002_02", "tend_PSARr_002_02", "tend_VHF_28", "tend_renko_TR",
                    "tend_renko_ATR", "tend_renko_brick", "tend_renko_change",

                    "volu_Chaikin_AD", "volu_Chaikin_ADOSC", "volu_OBV", "volu_EFI_13", "volu_NVI_1", "volu_PVI_1",
                    "volu_PVOL", "volu_PVR", "volu_PVT",

                    "vola_ATR", "vola_NATR", "vola_TRANGE", "vola_HWM", "vola_HWU", "vola_HWL", "vola_KCLe_20_2",
                    "vola_KCBe_20_2", "vola_KCUe_20_2", "vola_RVI_14", "vola_THERMO_20_2_05", "vola_THERMOma_20_2_05",
                    "vola_THERMOl_20_2_05", "vola_THERMOs_20_2_05", "vola_TRUERANGE_1", "vola_UI_14",

                    "cycl_DCPERIOD", "cycl_DCPHASE",
                    "cycl_PHASOR_inph", "cycl_PHASOR_quad", "cycl_SINE_sine", "cycl_SINE_lead",
                    "cycl_HT_TRENDMODE", "cycl_EBSW_40_10",

                    "sti_BETA", "sti_CORREL", "sti_LINEARREG", "sti_LINEARREG_ANGLE", "sti_LINEARREG_INTERCEPT",
                    "sti_LINEARREG_SLOPE", "sti_STDDEV", "sti_TSF", "sti_VAR", "sti_ENTP_10", "sti_KURT_30",
                    "sti_TOS_STDEVALL_LR", "sti_TOS_STDEVALL_L_1", "sti_TOS_STDEVALL_U_1", "sti_TOS_STDEVALL_L_2",
                    "sti_TOS_STDEVALL_U_2", "sti_TOS_STDEVALL_L_3", "sti_TOS_STDEVALL_U_3", "sti_ZS_30",

                    "ma_DEMA_5", "ma_EMA_5", "ma_KAMA_5", "ma_SMA_5", "ma_T3_5", "ma_TEMA_5", "ma_TRIMA_5", "ma_WMA_5",
                    "ma_DEMA_10", "ma_EMA_10", "ma_KAMA_10", "ma_SMA_10", "ma_T3_10", "ma_TEMA_10", "ma_TRIMA_10",
                    "ma_WMA_10", "ma_DEMA_20", "ma_EMA_20", "ma_KAMA_20", "ma_SMA_20", "ma_T3_20", "ma_TEMA_20",
                    "ma_TRIMA_20", "ma_WMA_20", "ma_DEMA_50", "ma_EMA_50", "ma_KAMA_50", "ma_SMA_50", "ma_T3_50",
                    "ma_TEMA_50", "ma_TRIMA_50", "ma_WMA_50", "ma_DEMA_100", "ma_EMA_100", "ma_KAMA_100", "ma_SMA_100",
                    "ma_T3_100", "ma_TEMA_100", "ma_TRIMA_100", "ma_WMA_100",

                    "trad_s3", "trad_s2", "trad_s1", "trad_pp", "trad_r1", "trad_r2", "trad_r3", "clas_s3", "clas_s2",
                    "clas_s1", "clas_pp", "clas_r1", "clas_r2", "clas_r3", "fibo_s3", "fibo_s2", "fibo_s1", "fibo_pp",
                    "fibo_r1", "fibo_r2", "fibo_r3", "wood_s3", "wood_s2", "wood_s1", "wood_pp", "wood_r1", "wood_r2",
                    "wood_r3", "demark_s1", "demark_pp", "demark_r1", "cama_s3", "cama_s2", "cama_s1", "cama_pp",
                    "cama_r1", "cama_r2", "cama_r3",

                    "ti_acc_dist", "ti_chaikin_10_3", "ti_choppiness_14", "ti_coppock_14_11_10", "ti_donchian_lower_20",
                    "ti_donchian_center_20", "ti_donchian_upper_20", "ti_ease_of_movement_14", "ti_force_index_13",
                    "ti_hma_20", "ti_kelt_20_lower", "ti_kelt_20_upper", "ti_mass_index_9_25", "ti_supertrend_20",
                    "ti_vortex_pos_5", "ti_vortex_neg_5", "ti_vortex_pos_14", "ti_vortex_neg_14",

                    "ichi_tenkan_sen", "ichi_kijun_sen", "ichi_senkou_a", "ichi_senkou_b", "ichi_isin_cloud",
                    "ichi_crash", "ichi_chikou_span",


                    "cdl_2CROWS", "cdl_3BLACKCROWS", "cdl_3INSIDE", "cdl_3LINESTRIKE", "cdl_3OUTSIDE",
                    "cdl_3STARSINSOUTH", "cdl_3WHITESOLDIERS", "cdl_ABANDONEDBABY", "cdl_ADVANCEBLOCK", "cdl_BELTHOLD",
                    "cdl_BREAKAWAY", "cdl_CLOSINGMARUBOZU", "cdl_CONCEALBABYSWALL", "cdl_COUNTERATTACK",
                    "cdl_DARKCLOUDCOVER", "cdl_DOJI", "cdl_DOJISTAR", "cdl_DRAGONFLYDOJI", "cdl_ENGULFING",
                    "cdl_EVENINGDOJISTAR", "cdl_EVENINGSTAR", "cdl_GAPSIDESIDEWHITE", "cdl_GRAVESTONEDOJI",
                    "cdl_HAMMER", "cdl_HANGINGMAN", "cdl_HARAMI", "cdl_HARAMICROSS", "cdl_HIGHWAVE", "cdl_HIKKAKE",
                    "cdl_HIKKAKEMOD", "cdl_HOMINGPIGEON", "cdl_IDENTICAL3CROWS", "cdl_INNECK", "cdl_INVERTEDHAMMER",
                    "cdl_KICKING", "cdl_KICKINGBYLENGTH", "cdl_LADDERBOTTOM", "cdl_LONGLEGGEDDOJI", "cdl_LONGLINE",
                    "cdl_MARUBOZU", "cdl_MATCHINGLOW", "cdl_MATHOLD", "cdl_MORNINGDOJISTAR", "cdl_MORNINGSTAR",
                    "cdl_ONNECK", "cdl_PIERCING", "cdl_RICKSHAWMAN", "cdl_RISEFALL3METHODS", "cdl_SEPARATINGLINES",
                    "cdl_SHOOTINGSTAR", "cdl_SHORTLINE", "cdl_SPINNINGTOP", "cdl_STALLEDPATTERN", "cdl_STICKSANDWICH",
                    "cdl_TAKURI", "cdl_TASUKIGAP", "cdl_THRUSTING", "cdl_TRISTAR", "cdl_UNIQUE3RIVER",
                    "cdl_UPSIDEGAP2CROWS", "cdl_XSIDEGAP3METHODS"
                    ]

def first_n_digits(num, n):
    return num // 10 ** (int(math.log(num, 10)) - n + 1)

def singleton(cls):
    instances = {}

    def get_instance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]

    return get_instance()


@singleton
class Url_stocks_pd():
    def __init__(self):
        self.pd = DF_URLS_STOCKS

    def get_url(self, stock, country="US"):
        u = ""
        try:
            u = self.pd.loc[(self.pd['STOCK'] == stock) & (self.pd['COUNTRY'] == country)]['URL'].values[
                0]  # Todo muy larga
        except Exception as e:
            Logger.logr.warn("The company stock does NOT have linked url.  stock: " + stock + " country: " + country)
            u = None
        return u



def change_date_in_weekend_monday(df_weekend):
    '''
    trasnladar los dias domingo a lunes y los sabados (apenas hay)  a viernes
    :param df_weekend:
    :return:
    '''
    df_weekend['Date'] = pd.to_datetime(df_weekend.Date, format='%Y-%m-%d')

    df_weekend.loc[df_weekend.Date.dt.dayofweek == 6, 'Date'] = df_weekend['Date'] + pd.DateOffset(
        days=1)  # las noticias del domingo pasan al lunes
    df_weekend.loc[df_weekend.Date.dt.dayofweek == 5, 'Date'] = df_weekend['Date'] + pd.DateOffset(days=-1)

    df_weekend['Date'] = df_weekend['Date'].dt.strftime("%Y-%m-%d")
    # df_weekend['Date'] = pd.to_datetime(df_weekend.Date, format='%Y-%m-%d')#al final para eliminar los minitos que ha puesto

    return df_weekend

def remove_weekend_data_values(df_weekr):
    '''
    trasnladar los dias domingo a lunes y los sabados (apenas hay)  a viernes
    :param df_weekend:
    :return:
    '''
    df_weekr['Date'] = pd.to_datetime(df_weekr['Date'], format='%Y-%m-%d')
    df_weekr = df_weekr[~ ((df_weekr.Date.dt.dayofweek == 6) | (df_weekr.Date.dt.dayofweek == 5))]#quitar los fines de semana de los datos
    df_weekr['Date'] = df_weekr['Date'].dt.strftime("%Y-%m-%d")

    return df_weekr


#inner_html = UtilsL.remove_tags_open_close(inner_html, "div")
def remove_tags_open_close(htlm_text, text_tag="table"):
    TAG_RE = re.compile(r'<[\/]{0,1}' + text_tag + r'[^>]*>')
    return TAG_RE.sub('', str(htlm_text))


def get_all_text_beteew_tags_remove_rest(htlm_text, text_tag="tr"):
    TAG_RE = re.compile(r'<' + text_tag + r'.*?>(.+?)<\/'+r'>')
    return TAG_RE.sub('', str(htlm_text))

def remove_chars_in_columns(df, columName):
    df[columName] = df[columName].map(
        lambda x: x.replace(" ", "").replace(":", '').replace(",", '').replace("/", '').replace(".", '').replace("-",
                                                                                                                 '').replace(
            "(", '').replace(")", '').replace('\\', '').replace('\'', ''))  #
    return df

def replace_bat_chars_in_columns_name(df, char_new ="_"):
    for c in df.columns:
        new_colum = str(c).replace(" ",char_new).replace("%", "per").replace(":", char_new).replace(",", char_new).replace("/", char_new).replace(".", char_new).replace("-",                                                                                           char_new).replace(
            "(", char_new).replace(")", '').replace('\\', char_new).replace('\'', char_new)
        df = df.rename(columns={ c:  new_colum})
    return df


    df[columName] = df[columName].map(
        lambda x: x.replace(" ",char_new).replace("%", "per").replace(":", char_new).replace(",", char_new).replace("/", char_new).replace(".", char_new).replace("-",
                                                                                                                 char_new).replace(
            "(", char_new).replace(")", '').replace('\\', char_new).replace('\'', char_new))  #
    return df

def clean_float_columns(df_f, columnName):
    '''
    Elimina el % , el K lo cambia por 000 y el M los cambia por 000000
    :param df_f:
    :param columnName:
    :return:
    '''
    df_f[columnName] = df_f[columnName].map(lambda x: x.replace("%", ""))
    # df_f['Company'] = df_f['Company'].map(lambda x: x.replace(r"(\d+)\.(\d+)[Kk]", r"\1000\.\2",regex=True))
    #TODO los decimales no se multiplican bien
    df_f[columnName] = df_f[columnName].astype(str)
    #df_f[columnName].replace('--', np.nan)
    #a = df_f.loc[df_f[columnName].str.contains(r"[Bb]"), columnName].astype(str).str.replace(r"[Bb]", r"", regex=True).astype(float) * 1000000000
    df_f[columnName] = df_f[columnName].astype(str).str.replace(',', '')

    df_f.loc[df_f[columnName].str.contains(r"[Bb]"), columnName] = df_f.loc[df_f[columnName].str.contains(r"[Bb]"), columnName].astype(str).str.replace(r"[Bb ]", r"", regex=True).astype(float) * 1000000000
    df_f[columnName] = df_f[columnName].astype(str)

    df_f.loc[df_f[columnName].str.contains(r"[Mm]"), columnName] = df_f.loc[df_f[columnName].str.contains(r"[Mm]"), columnName].str.replace(r"[Mm ]", r"",regex=True).astype(float) * 1000000
    df_f[columnName] = df_f[columnName].astype(str)

    df_f.loc[df_f[columnName].str.contains(r"[Kk]"), columnName] = df_f.loc[df_f[columnName].str.contains(r"[Kk]"), columnName].astype(str).str.replace(r"[Kk ]", r"",regex=True).astype(float) * 1000
    df_f[columnName] = df_f[columnName].astype(str)

    #df_f[columnName] = df_f[columnName].astype(float)
    #NO BORRRAR EL COMENTARIO:
    #df_f[columnName] = df_f[columnName].str.replace(r"(\d+)(\.)(\d+)[Kk]", r"\1_000_\2\3", regex=True).str.replace("_","")
    #df_f[columnName] = df_f[columnName].str.replace(r"(\d+)(\.)(\d+)[Mm]", r"\1_000000_\2\3", regex=True).str.replace("_","")
    return  df_f



def get_trs_dataframe(full_html):
    inner_html = tostring(full_html)
    inner_html = get_all_text_beteew_tags_remove_rest(inner_html, "tr")  # clean keep only the data
    inner_html = "<table>" + str(inner_html) + "</table>"
    df_r = pd.read_html(inner_html)[0]
    df_r = df_r.replace('', np.nan)
    df_r = df_r.replace('-', np.nan)
    df_r = df_r.dropna(how='any')
    return df_r


def get_trs_dataframe_rev(full_html, is_moving_averages=False, delete_emply_column_name_Unnamed= False):
    #Si ahy dos th header table , la segunda es pasada a td, estandar row
    #for elem in full_html.xpath('//th/..')[1:]: #ignore the first position    #full_html.xpath('//table/tbody/tr[position()>1]'):
    #    for e in elem:
    #        if e.tag == "th":
    #            e.tag = "td"

    #full_html = [ (h.tag = "tr") for h in full_html.xpath('//th/..')[1:]]
    inner_html = tostring(full_html)
    inner_html = remove_tags_open_close(inner_html, "table")  # clean keep only the data
    inner_html = remove_tags_open_close(inner_html, "span")  # clean keep only the data
    inner_html = remove_tags_open_close(inner_html, "tbody")
    inner_html = remove_tags_open_close(inner_html, "div")
    if is_moving_averages:
        inner_html = re.sub("<br/>", "</td><br><td>", inner_html)

    inner_html = re.sub(r"\\t|\\n|\\r", '', inner_html)
    inner_html = re.sub(r"\t|\n|\r", '', inner_html)

    inner_html = get_all_text_beteew_tags_remove_rest(inner_html, "tr")  # clean keep only the data
    inner_html = "<table>" + str(inner_html) + "</table>"
    df_r = pd.read_html(inner_html)[0]

    if delete_emply_column_name_Unnamed:
        df_r = df_r.drop(df_r.columns[df_r.columns.str.contains("Unnamed:") ], axis=1)

    df_r = df_r.replace('', np.nan)
    df_r = df_r.replace('-', np.nan)
    df_r = df_r.dropna(axis=1, how='all') #si hay columnas vacias
    df_r = df_r.dropna(how='any')
    return df_r


import datetime as date_dict
def replace_list_in_sub_keys_dicts(dict_sub_list):
    '''
    cambio de los elementos que estan en ["dato"] por "dato" , quitar las listas individuales
    :param dict_sub_list:
    :return:
    '''
    for key in dict_sub_list.keys():
        if type(dict_sub_list[key]) == dict:
            for k2, v2 in dict_sub_list[key].items():
                if type(v2) == list and len(v2) > 1 and all(isinstance(ele, date_dict.datetime) for ele in v2) :#     isinstance(v2, date_dict.datetime):
                    #dict_sub_list[key][k2] = v2[0].strftime('%Y-%m-%d')
                    dict_sub_list[key][k2] = [i.strftime('%Y-%m-%d') for i in v2]
                elif type(v2) == list and len(v2) == 1:
                    try:
                        dict_sub_list[key][k2] = maybe_make_number(v2[0])#cast to float if it is allow
                    except ValueError:
                        dict_sub_list[key][k2] = v2[0]
                elif type(v2) == list and len(v2) > 1:
                    dict_sub_list[key][k2] = list(map(maybe_make_number, (v2)))

    return  dict_sub_list

def all_equal(list):
    '''
    https://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-identical
    :param iterator: 
    :return: 
    '''
    return len(set(list)) <= 1


def maybe_make_number(s):
    """Returns a string 's' into a integer if possible, a float if needed or
    returns it as is.
    https://stackoverflow.com/questions/7368789/convert-all-strings-in-a-list-to-int
    """

    # handle None, "", 0
    if not s:
        return s
    try:
        f = float(s)
        i = int(f)
        return i if f == i else f
    except ValueError:
        return s
    except TypeError:
        return s

# def cast_list_to_float(list):
#     for l in list:
#         try:
#             l = float(l)  # cast to float if it is allow
#         except ValueError:
#             l = l
#     return list


def dict_drop_duplicate_subs_elements(dict_drop):
    #https://stackoverflow.com/questions/8749158/removing-duplicates-from-dictionary
    result = {}

    for key in dict_drop.keys():
        if type(dict_drop[key]) == dict:

            #[dict(t) for t in {tuple(d.items()) for d in l}]

            for key_2, value in dict_drop[key].items():#hay sub dicionarios
                if value not in result.values():
                    result[key_2] = value

    return result


def add_rename_all_columns_df(df, prefix="", sufix=""):
    for c in df.columns:
        df = df.rename(columns={ c:  str(prefix)+ c + str(sufix)})

    return df

#https://stackoverflow.com/questions/31323499/sklearn-error-valueerror-input-contains-nan-infinity-or-a-value-too-large-for
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)