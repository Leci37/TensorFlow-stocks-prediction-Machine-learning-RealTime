import requests
import pandas as pd
from LogRoot.Logging import Logger

headers = {
    'Connection': 'keep-alive',
    'Cache-Control': 'max-age=0',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'Sec-Fetch-Site': 'cross-site',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-User': '?1',
    'Sec-Fetch-Dest': 'document',
    'Referer': 'https://www.jpx.co.jp/english/markets/index.html',
    'Accept-Language': 'en-US',
}
PARAMS = {
    "referrer": "https://es.investing.com/equities/pfizer-earnings",
    "body": "pairID=7989&last_timestamp=2019-10-29",
    "method": "POST",
    "mode": "cors"
}
#//*[@id="showMoreEarningsHistory"]/a

response = requests.post('https://es.investing.com/equities/pfizer-earnings', headers=headers, params = PARAMS)
#https://es.investing.com/equities/pfizer-earnings
#https://es.investing.com/equities/morehistory
#print(response.text)
table = pd.read_html(response.text, attrs={"class": "genTbl openTbl ecoCalTbl earnings earningsPageTbl"})[0]
Logger.logr.debug(table.sample(6))