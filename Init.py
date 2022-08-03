# 1.- Implementar una endpoint de una api REST en python que reciba peticiones con los siguientes parámetros:
#         Nombre de usuario
#         Código Postal
# 2.- A partir del CP, y utilizando el servicio de geonames, obtenemos la ciudad del usuario (http://www.geonames.org)
# 3.- Almacenamos la información en bbdd, en dos tablas. En la tabla master guardamos el nombre del usuario. En la tabla detalle guardamos su cp y su ciudad.
# 4.- El servidor responde en formato JSON, indicando si todo ha ido bien o si ha habido algún tipo de error.
#
#
# Se debe pensar que este endpoint formará parte de un conjunto de endpoints distintos que no se deben programar para este ejercicio, pero se debe tener en cuenta a la hora de crear la estructura de software que permita extender/añadir nuevas funcionalidades de forma simple.
#
# Entregables:
#
#     Código fuente, tests y un pequeño archivo de texto explicando los aspectos más relevantes de la implementación así como las decisiones tomadas.


import os

from LogRoot.Logging import Logger

END_POINT_COMMAND = " endpoints --prefix=Controllers --host=localhost:8000 "

def Strart_End_Point():
    Logger.logr.info("EndPoint listening process is about to begin. Command: "+END_POINT_COMMAND )
    end_Point_Porcess = os.system(END_POINT_COMMAND)
    Logger.logr.debug("EndPoint listening process has been completed")


#Log = Logging.Log_Base().Logging()
Strart_End_Point()

#Get_Villages('48013')
# https://github.com/georgezouq/awesome-ai-in-finance
# https://github.com/VivekPa/AIAlpha



