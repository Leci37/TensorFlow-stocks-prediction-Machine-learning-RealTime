import sqlite3
from sqlite3 import Error
from LogRoot.Logging import Logger
import DateBaseObjects

PATH_DDBB = 'DDBB\database.db'


def get_connection():
    return sqlite3.connect(PATH_DDBB)

class StoreException(Exception):
    def __init__(self, message, *errors):
        Exception.__init__(self, message)
        self.errors = errors


# domains
#En la tabla master guardamos el nombre del usuario.

# base store class
class Store():
    def __init__(self):
        try:
            self.conn = get_connection()
        except Exception as e:
            raise StoreException(*e.args, **e.kwargs)
        self._complete = False

    def __enter__(self):
        return self

    def __exit__(self, type_, value, traceback):
        # can test for type and handle different situations
        self.close()

    def complete(self):
        self._complete = True

    def close(self):
        if self.conn:
            try:
                if self._complete:
                    self.conn.commit()
                else:
                    self.conn.rollback()
            except Exception as e:
                raise StoreException(*e.args)
            finally:
                try:
                    self.conn.close()
                except Exception as e:
                    raise StoreException(*e.args)

    def execute_query(self, query):
        try:
            c = self.conn.cursor()
            c.execute(query)
            self.conn.commit()
            return  c.fetchall()
        except Exception as e:
            Logger.logr.warning(e)
            raise StoreException('error storing user')

class Store_User(Store):

    def add_user(self, User_BBDD):
        query = "INSERT INTO Master (UserName, CP,CountryCode ) VALUES " + User_BBDD.valueFormatQuery()
        Logger.logr.debug("Execute query: " + query)
        return self.execute_query(query)

    def select_user_by_name(self, name_String):
        query = "SELECT * FROM Master WHERE UserName = \""+name_String+"\" ; "##TODO ignore case
        Logger.logr.debug("Execute query: " + query)
        return self.execute_query(query)


class Store_CP(Store):

    def add_CP(self, CP_DDBB):
        query = "INSERT INTO Detalles (CP, CountryCode, Locations ) VALUES " + CP_DDBB.valueFormatQuery()
        Logger.logr.debug("Execute query: " + query)
        return self.execute_query(query)

    def select_user_by_CP(self, CP_string):
        query = "SELECT * FROM Detalles WHERE CP = \""+CP_string+"\" ; "##TODO ignore case
        Logger.logr.debug("Execute query: " + query)
        return self.execute_query(query)

def Update_DDBB(dict_result):
    try:
        with Store_CP() as store_CP:
            store_CP.add_CP(DateBaseObjects.DetallesLocationsDDBB( dict_result['PostalCode'], dict_result['Country'], dict_result['Locations']))
        with Store_User() as store_user:
            store_user.add_user(DateBaseObjects.MasterUserDDBB(dict_result['UserName'], dict_result['PostalCode'], dict_result['Country']))



    except StoreException as e:
        # exception handling here
        Logger.logr.warning(e)

