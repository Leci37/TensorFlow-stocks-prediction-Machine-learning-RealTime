import json
from endpoints import Controller
from LogRoot.Logging import Logger
from endpoints.decorators import param
from geoNames import Get_Locations
import pycountry


FILE_LOG_NAME_endpoints = 'LogEndPoints.log'
COUNTRY_CODES_LIST = [c.alpha_2 for c in pycountry.countries]
#Log = Logging.Log_Base().Logging()

class Default(Controller):
    def GET(self, **kwargs):pass

    def POST(self, **kwargs):pass

def is_not_blank_or_empty(s):
    return bool(s and s.strip())


class Streetaddress(Controller):
    @param('UserName')
    @param('PostalCode')
    @param('Country', default="ES")
    def GET(self, **kwargs):
        bool_Update_DDBB = True
        userName = kwargs['UserName']
        postalCode = kwargs['PostalCode']
        countryCode = kwargs['Country']
        if not countryCode:
            countryCode = "ES"

        fail_message = self.valitate_param(countryCode, postalCode, userName)
        if fail_message is not None:
            #Log inside of funtion valitate_param
            dictResponse_error = {'Code': '200', 'Result': str(fail_message), 'UserName': userName, 'PostalCode': postalCode,'Country': kwargs['Country']}
            return json.dumps(dictResponse_error)


        #Logger.logr.("The following parametters have been received. UserName:" + userName + " PostalCode: " + postalCode+ " Country: " + country)
        listVillages = Get_Locations(postalCode,countryCode)

        if listVillages.__len__() == 0:
            Logger.logr.warning("There are no locations with these parameters UserName:" + userName + " PostalCode: " + postalCode+ " Country: " + countryCode)
            listVillages = "There are no locations with these parameters"
            bool_Update_DDBB = False

        jsonVillages = json.dumps(listVillages)
        dictResponse = {'Code': '200', 'Locations' : str(jsonVillages) , 'UserName': userName, 'PostalCode': postalCode, 'Country': countryCode}
        jsonResponse = json.dumps(dictResponse)

        if bool_Update_DDBB:
            import DateBase
            DateBase.Update_DDBB(dictResponse)
        Logger.logr.info("Get received and processed. UpdateDDBB: "+ str(bool_Update_DDBB)+" RequestJSON: " + jsonResponse)
        return jsonResponse
        #return "The following parametters have been received. UserName: " + userName + " PostalCode: " + postalCode + " Country: " + country +" Locations: "+str(listVillages)

    def valitate_param(self, countryCode, postalCode, userName):
        if not userName:
            Logger.logr.warning("UserName cannot be null or empty  UserName:" + userName)
            return "UserName cannot be null or empty"
        if not postalCode or not str(postalCode).isnumeric():
            Logger.logr.warning("PostalCode must have a numerical value  PostalCode: " + postalCode)
            return "PostalCode must have a numerical value"
        if not countryCode in COUNTRY_CODES_LIST:
            Logger.logr.warning("Country code is not valid [pycountry - Countries (ISO 3166)]  Country: " + countryCode)
            return "Country code is not valid [pycountry - Countries (ISO 3166)]"

        return None

#http://localhost:8000/Streetaddress?UserName=Luis&PostalCode=09569