from seleniumwire import webdriver  # Import from seleniumwire
import numpy as np
import pandas as pd
# Create a new instance of the Chrome driver
from webdriver_manager.chrome import ChromeDriverManager

from mitmproxy import ctx
import zlib

#https://stackoverflow.com/questions/71986111/selenium-mitmproxy
def response_flow(text):
    try:
        content = zlib.decompress(text, 16+zlib.MAX_WBITS)
    except:
        print("Error in response_flow decompress")
        content = text
    return content

driver = webdriver.Chrome(ChromeDriverManager().install())
# driver = webdriver.Chrome()
URL = "https://www.barchart.com/stocks/quotes/MELI/interactive-chart"
# Go to the Google home page
driver.get(URL)

# Access requests via the `requests` attribute
for request in driver.requests:
    if request.response:
        # print(
        #     request.url,
        #     request.response.status_code,
        #     request.response.headers['Content-Type']
        # )
        #INTERVALO 5MINm intrerdia
        #https://www.barchart.com/proxies/timeseries/queryminutes.ashx?symbol=MELI
        # &interval=5&maxrecords=640&volume=contract&order=asc&dividends=false&backadjust=false&daystoexpiration=1&contractroll=expiration
        #INTERVALO 30 min interdia 5 dias
        #https://www.barchart.com/proxies/timeseries/queryminutes.ashx?symbol=MELI&
        # interval=30&maxrecords=640&volume=contract&order=asc&dividends=false&backadjust=false&daystoexpiration=1&contractroll=expiration
        if "barchart.com/proxies/timeseries/" in request.url:
            print(
                request.url,
                request.response.status_code,
                request.response.headers['Content-Type']
            )
            csv_respon = str(response_flow(request.response.body))
            csv_respon = csv_respon.replace('\b', '').replace('\'', '')
            df = pd.DataFrame([x.split(',') for x in csv_respon.split('\\n')], columns=['ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
            print("AQUI ESTA EL VOLUMEN ", csv_respon)
            driver.get(request.url)