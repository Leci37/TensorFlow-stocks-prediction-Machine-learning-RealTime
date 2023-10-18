from threading import Thread

import pandas as pd
import numpy as np
import re
from lxml.etree import tostring
import math

from LogRoot.Logging import Logger

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def first_n_digits(num, n):
    return num // 10 ** (int(math.log(num, 10)) - n + 1)

def singleton(cls):
    instances = {}

    def get_instance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]

    return get_instance()


# @singleton
# class Url_stocks_pd():
#     def __init__(self):
#         self.pd = pd.read_csv('Utils/URL_dict_stocks_news.csv', sep='|')
#
#     def get_url(self, stock, country="US"):
#         u = ""
#         try:
#             u = self.pd.loc[(self.pd['STOCK'] == stock) & (self.pd['COUNTRY'] == country)]['URL'].values[
#                 0]  # Todo muy larga
#         except Exception as e:
#             Logger.logr.warn("The company stock does NOT have linked url.  stock: " + stock + " country: " + country)
#             u = None
#         return u



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


def fill_last_values_of_colum_with_previos_value(df, colum_name):
    try:
        list_sort_columns_1 = ['ticker', 'Date']
        list_sort_columns_2 = ['Date', 'ticker']
        if 'ticker' not in  df.columns:
            list_sort_columns_1 = ['Date']
            list_sort_columns_2 = ['Date']
        if 'Date' not in df.columns:
            Logger.logr.warning("In the entered data frame there is neither Date nor ticker, it is not possible to make the correct order, the default order is used.    Shape: " + str(df.shape))
            df = df[[colum_name]].fillna(method='ffill')
            return df

        pd.options.mode.chained_assignment = None
        df[colum_name] = df.sort_values(list_sort_columns_1, ascending=True)[[colum_name]].fillna(method='ffill')
        pd.options.mode.chained_assignment = 'warn'
        #df.loc[:, [colum_name]] = df.sort_values(['ticker', 'Date'], ascending=True)[[colum_name]].fillna(method='ffill')
        df = df.sort_values(list_sort_columns_2, ascending=True)
    except Exception as e:
        print(e)
    return df



def get_recent_dates(df_dates, format_strftime = "%Y%m%d"):
    max_recent_date = pd.to_datetime(df_dates['Date'].max()).strftime(format_strftime)
    min_recent_date = pd.to_datetime(df_dates['Date'].min()).strftime(format_strftime)
    return max_recent_date, min_recent_date


# as per recommendation from @freylis, compile once only
CLEANR = re.compile('<.*?>')
def clean_html_tags(raw_html):
  cleantext = re.sub(CLEANR, '', raw_html)
  return cleantext

import os, glob
def remove_files_starwith(file):
    for filename in glob.glob(file + "*"):
        os.remove(filename)


def remove_column_name_repeted_last_one(columnas_para_retirar, df_in_pure):
    num_times_is_repeted  = 0
    len_cols = len(columnas_para_retirar)
    for i_col in range(len(df_in_pure.columns), len_cols + 1, -1):  # Tiene que ir para coger primero las ultimas
        if set(df_in_pure.columns[(i_col - len_cols):(i_col)]) == set(columnas_para_retirar):
            num_times_is_repeted +=  1
            for i in range((i_col-1), (i_col - (len_cols + 1)), -1):  # Tiene que ir de arriba a abajo para no pisarse
                # https://stackoverflow.com/questions/20297317/python-dataframe-pandas-drop-column-using-int
                # delete colum by index, borrar columna por indice
                df_in_pure = df_in_pure.iloc[:, [j for j, c in enumerate(df_in_pure.columns) if j != i]]

    return df_in_pure , num_times_is_repeted



#Antes de llamar deben estar organizadas de viejos a nuevos
# df_S_raw = df_S_raw.sort_values('Date', ascending=True)
def union_3last_rows_to_one_OLHLV(df_S_raw):
    # df_S_raw.index = pd.to_datetime(df_S_raw['Date'])  #df_his['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    # print( date_min.resample('15T').sum()['Volume'] )

    df_S_raw['Open'] = df_S_raw['Open'].shift(-2)
    df_S_raw['High'] = df_S_raw['High'].rolling(window=3).max().shift(-2)
    df_S_raw['Low'] = df_S_raw['Low'].rolling(window=3).min().shift(-2)
    df_S_raw['Close'] = df_S_raw['Close']
    df_S_raw['Volume'] = df_S_raw['Volume'].rolling(window=3).sum().shift(-2)
    return df_S_raw


def thread_list_is_alive(list_thread,producer,consumer ):
    # list_is_alive = [t for t in list_thread if not t.is_alive()]
    list_is_alive =[]
    for thr in  list_thread:
        if thr.is_alive():
            list_is_alive.append(True)
        else:
            Logger.logr.warning(" The thread is not alive, proceed to restart.  Name: " + str(thr.name))
            list_is_alive.append(False)
            if thr.name.startswith("PROD"):
                producer_thr = Thread(target=producer, args=(), name='PROD_AU')
                producer_thr.start()
            if thr.name.startswith("CONS"):
                consumer_thr_1 = Thread(target=consumer, args=(1,), name='CONS_AU')
                consumer_thr_1.start()
    return list_is_alive

def remove_strong_correlations_columns(df_cor , factor:float):
    # Create correlation matrix https://stackoverflow.com/questions/29294983/how-to-calculate-correlation-between-all-columns-and-remove-highly-correlated-on
    corr_matrix = df_cor.corr().abs()  # Select upper triangle of correlation matrix
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))  # Find features with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > factor)]
    print("\tDEBUG Columns more correlated than factor, will be Removed. Factor: ",factor, " Columns: ",",".join(to_drop) )
    df_cor.drop(to_drop, axis=1, inplace=True)  # Drop features
    return df_cor

def log_versions_libs():
    # This Python 3 environment comes with many helpful analytics libraries installed
    # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

    #load packages
    import sys #access to system parameters https://docs.python.org/3/library/sys.html
    print("Python version: {}". format(sys.version))

    import pandas as pd #collection of functions for data processing and analysis modeled after R dataframes with SQL like features
    print("pandas version: {}". format(pd.__version__))

    import matplotlib #collection of functions for scientific and publication-ready visualization
    print("matplotlib version: {}". format(matplotlib.__version__))

    import numpy as np #foundational package for scientific computing
    print("NumPy version: {}". format(np.__version__))

    import scipy as sp #collection of functions for scientific computing and advance mathematics
    print("SciPy version: {}". format(sp.__version__))

    import IPython
    from IPython import display #pretty printing of dataframes in Jupyter notebook
    print("IPython version: {}". format(IPython.__version__))

    import sklearn #collection of machine learning algorithms
    print("scikit-learn version: {}". format(sklearn.__version__))