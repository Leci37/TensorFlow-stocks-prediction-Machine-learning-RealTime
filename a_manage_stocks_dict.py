from enum import Enum

DICT_COMPANYS = {
    "@FAV":
        ["MELI", "TWLO", "RIVN", "SNOW", "UBER", "U" , "PYPL", "GTLB", "MDB", "TSLA", "DDOG"],
    "@ROLL" :
        ["MELI", "TWLO","RIVN","SNOW", "UBER", "U" , "PYPL", "GTLB","MDB", "TSLA", "DDOG","SHOP", "NIO","RBLX", "TTD", "APPS", "ASAN",  "DOCN", "AFRM", "PINS"],
    "@VOLA" :
        ["UPST", "RIVN", "SNOW", "LYFT", "SPOT", "GTLB", "MDB", "HUBS", "TTD", "APPS", "ASAN", "AFRM", "DOCN", "DDOG", "SHOP", "NIO", "U", "RBLX"]

    }

class MODEL_TYPE_COLM(Enum):
    VGOOD = "_vgood16_"
    GOOD = "_good9_"
    REG = "_reg4_"
