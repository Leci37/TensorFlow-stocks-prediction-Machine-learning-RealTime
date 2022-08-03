# Python program showing
# abstract class cannot
# be an instantiation
# 3.- Almacenamos la información en bbdd, en dos tablas. En la tabla master guardamos el nombre del usuario. En la tabla detalle guardamos su cp y su ciudad.
# import abc
# from abc import ABC, abstractmethod

#https://www.geeksforgeeks.org/abstract-classes-in-python/
class CommonDDBB():
    def __init__(self, fCP, fCountryCode):
        self.CP = fCP
        self.CountryCode = fCountryCode

    def CP(self):
        return self.CP

    def CountryCode(self):
        return self.CountryCode

    def DataInfo(self):
        return " CP: {0}, CountryCode: {1} ".format(self.CP, self.CountryCode)


class MasterUserDDBB(CommonDDBB):
    def __init__(self,fUserName, fCP, fCountryCode):
        super().__init__(fCP, fCountryCode)
        self.UserName = fUserName

    def __str__(self):
        return "MasterDDBB Username: {0}, {1}".format(self.UserName, super().DataInfo())

    def valueFormatQuery(self):#('Pepe','43001','ES')
        return "( '{0}' , '{1}' , '{2}' )".format(self.UserName, self.CP, self.CountryCode)


class DetallesLocationsDDBB(CommonDDBB):
    def __init__(self, fCP, fCountryCode,fLocations):
        CommonDDBB.__init__(self,fCP, fCountryCode)
        self.Locations = fLocations

    def __str__(self):
        return "DetallesDDBB Locations: {0}, {1}".format(self.Locations, super().DataInfo())

    def valueFormatQuery(self):#('43000','ES','Montijo')
        return "( '{0}' , '{1}' , '{2}' )".format( self.CP, self.CountryCode, self.Locations)
