# http://www.geonames.org/export/web-services.html
#
# Example http://api.geonames.org/postalCodeSearch?postalcode=9011&maxRows=10&username=demo
#
# This service is also available in JSON format : http://api.geonames.org/postalCodeSearchJSON?postalcode=9011&maxRows=10&username=demo
# importing the requests library
import requests
import lxml.html as LH
from LogRoot.Logging import Logger

def text(elt):
    return elt.text_content().replace(u'\xa0', u' ')

# api-endpoint
#http://api.geonames.org/findNearbyPostalCodesJSON?postalcode=09569&country=ES&radius=10&username=demo
API_ENDPOINT = "http://api.geonames.org/findNearbyPostalCodesJSON"
# https://www.geeksforgeeks.org/get-post-requests-using-python/
# defining a params dict for the parameters to be sent to the API
PARAMS = {'postalcode': '09569' , 'country' :"ES", 'username':"demo" }

HTML_ENDPOINT =  "https://www.geonames.org/advanced-search.html" # ?q=09569&country=ES

def Get_Locations(postalCode, country ="ES"):
    # sending get request and saving the response as response object
    get_Params = {'q': postalCode, 'country': country}
    r = requests.get(url=HTML_ENDPOINT, params=get_Params)
    r.content

    root = LH.fromstring(r.content)
    listVillages = []
    for table in root.xpath('/html/body/div[1]/table'):
        header = [text(th) for th in table.xpath('//th')]
        for td in table.xpath('//tr/td[2]/a'):#for td in table.xpath('//tr/td[2][not(span)]')[6:]:
            #td = td.xpath('td/a')
            t = text(td)
            if  t not in "\\u" :
                listVillages.append(u"{}".format(text(td)))
            #Logger.logr.debug(text(td))

    Logger.logr.info("With the following parameters has been received from GeoNames. Parametres: "+str(get_Params)+" Locations: "+str(listVillages) )
    return listVillages
