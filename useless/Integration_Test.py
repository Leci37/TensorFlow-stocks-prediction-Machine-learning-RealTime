import requests
import unittest

from Test.DictTestData import DICT_VALUE_TEST, DICT_FORMA_URL_TEST
CUT_START = 12

class TestMethods(unittest.TestCase):
    def execute_code(self,url):
        try:
            r1 = requests.get(url)
            return  str(r1.status_code) + " - " + str(r1.json())
        except Exception as e:
            return e

    def test_values(self):
        print("Test de valores: ")
        i = 0
        r = ""
        for key, value in DICT_VALUE_TEST.items():
            i += 1
            self.for_each_dictcionary(i, key, value)
            #self.assertTrue(value.startswith(result1))

    def test_url_format(self):
        print("Test de formatos: ")
        i = 0
        r =""
        for key, value in DICT_FORMA_URL_TEST.items():
            i += 1
            self.for_each_dictcionary(i, key, value)



    def for_each_dictcionary(self, i, key, value):
        try:
            result1 = self.execute_code(key)
            self.assertEqual(value[:CUT_START], str(result1)[:CUT_START])
            r = "OK"
        except AssertionError as e:
            self.verificationErrors.append(str(e))
            r = "FAIL_assert"
        except Exception as ex:
            print(ex)
            r = "FAIL"
        print(i, " ", r, " - ", value[:CUT_START], "   ", str(result1)[:CUT_START], "   ", key)



if __name__ == '__main__':
    unittest.main()
