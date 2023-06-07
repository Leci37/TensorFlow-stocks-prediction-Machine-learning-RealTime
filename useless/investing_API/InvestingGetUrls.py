import investpy
import pandas as pd
from LogRoot.Logging import Logger

# https://www.investing.com/equities/apple-computer-inc-

Logger.logr.info("")
list_stock_countries = investpy.get_stocks_list("united states")
df = pd.DataFrame(columns=['STOCK', 'COUNTRY', 'URL'])
# df2 = {'STOCK': 'Vikram', 'Last Name': 'Aruchamy', 'Country': 'India'}
# df = df.append(df2, ignore_index = True)
for s in list_stock_countries:
    try:
        df_US = investpy.get_stock_company_profile(stock=s, country="united states", language="english")
    except Exception as e:
        Logger.logr.warn( s+ "  Exception: "+ str(e))
    df.loc[len(df.index)] = [s, "US", df_US['url'].replace("-company-profile", "")]
    # df.loc[-1] = [s,"US", df_US['url'].replace("-company-profile","")]  # adding a row
    Logger.logr.info(s+ "  => "+ df_US['url'].replace("-company-profile", ""))

list_stock_countries = investpy.get_stocks_list("spain")

# df2 = {'STOCK': 'Vikram', 'Last Name': 'Aruchamy', 'Country': 'India'}
# df = df.append(df2, ignore_index = True)
for s in list_stock_countries:
    try:
        df_SP = investpy.get_stock_company_profile(stock=s, country="spain", language="english")
    except Exception as e:
        Logger.logr.warn(" "+ s+ "  Exception:"+ str(e))
    df.loc[len(df.index)] = [s, "SP", df_US['url'].replace("-company-profile", "")]
    Logger.logr.info(s+ "  => "+ df_SP['url'].replace("-company-profile", ""))

# df = df_US._append(df_SP)
Logger.logr.debug(df)
df.to_csv("Utils/URL_dict_stocks_news.csv", sep='|')
