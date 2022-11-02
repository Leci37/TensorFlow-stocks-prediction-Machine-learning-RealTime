from enum import Enum

DICT_COMPANYS = {
    "@FAV":
        ["MELI", "TWLO", "RIVN", "SNOW", "UBER", "U" , "PYPL", "GTLB", "MDB", "TSLA", "DDOG"],
    "@ROLL" :
        ["MELI", "TWLO","RIVN","SNOW", "UBER", "U" , "PYPL", "GTLB","MDB", "TSLA", "DDOG","SHOP", "NIO","RBLX", "TTD", "APPS", "ASAN",  "DOCN", "AFRM", "PINS"],
    "@VOLA" :
        ["UPST", "RIVN", "SNOW", "LYFT", "SPOT", "GTLB", "MDB", "HUBS", "TTD", "APPS", "ASAN", "AFRM", "DOCN", "DDOG", "SHOP", "NIO", "U", "RBLX"],
    #top LAS MÁS activas por Volumen  realizado por https://www.webull.com/quote/us/actives
    "@TOP200": #Xpath //p[@class="tit bold"]/text()   good side https://codebeautify.org/Xpath-Tester
        ["AMD", "TSLA", "AAPL", "MULN", "NVDA", "AMZN", "AGFY", "CCL", "NIO", "BAC",  "F", "INTC", "T", "FNHC", "VALE", "ACI", "ILAG", "IMRA", "PLTR", "SOFI", "AAL", "WFC", "NU", "JPM", "XPEV", "SHOP", "MSFT", "NOK", "DBGI", "C", "PBR", "CMCSA", "SNAP", "HBAN", "DNA", "PLUG", "SWN", "VZ", "LASE", "NCLH", "GOOGL", "PTON", "CEI", "PCG", "ZVO", "META", "LCID", "ABEV", "MMAT", "GOOG", "UBER", "RIVN", "DAL", "AMC", "DKNG", "WISH", "WBD", "MU", "UMC", "KR", "CSCO", "TSM", "USB", "GOLD", "OPEN", "NKLA", "PBTS", "OXY", "ARVL", "BEKE", "TLRY", "KMI", "BB", "PFE", "MPW", "FFIE", "GRAB", "SQ", "HPE", "CCJ", "ASX", "KEY", "KO", "MARA", "CSX", "RIG", "NFLX", "SIRI", "XOM", "BCS", "KGC", "CS", "RBLX", "SYTA", "LUMN", "CPG", "AGNC", "FCX", "NTNX", "COIN", "BABA", "MS", "INFY", "LU", "SCHW", "LYG", "AUY", "MRVL", "JD", "SMRT", "ACB", "ET", "GM", "PYPL", "PINS", "AFRM", "SLB", "PING", "NUTX", "MO", "HAL", "MRO", "UAL", "CGC", "TELL", "KHC", "NVTA", "HPQ", "CVE", "BP", "WBA", "RUN", "AMAT", "ERIC", "CVNA", "CLF", "TOP", "TWTR", "BMY", "RIOT", "MRK", "NEM", "IDEX", "X", "TFC", "LYFT", "BTG", "UEC", "RF", "PDD", "KDP", "DVN", "FCEL", "VTGN", "HST", "RTO", "NEE", "CPNG", "FTCH", "MF", "JBLU", "CARR", "COMS", "AG", "VRM", "FTI", "TEVA", "MOS", "ROKU", "CHPT", "DIS", "IBN", "ORCL", "QCOM", "VOD", "MDLZ", "GILD", "NLY", "NKE", "LI", "GFI", "BSX", "M", "HMY", "COP", "APA", "NYCB", "IQ", "HUT", "VTRS", "TXN", "AZN", "AMCR", "HL", "QS", "BBBY", "IMUX", "PATH", "CVX"],
    "@TOP100":
        ["AMD", "TSLA", "AAPL", "MULN", "NVDA", "AMZN", "AGFY", "CCL", "NIO", "BAC",  "F", "INTC", "T", "FNHC", "VALE", "ACI", "ILAG", "IMRA", "PLTR", "SOFI", "AAL", "WFC", "NU", "JPM", "XPEV", "SHOP", "MSFT", "NOK", "DBGI", "C", "PBR", "CMCSA", "SNAP", "HBAN", "DNA", "PLUG", "SWN", "VZ", "LASE", "NCLH", "GOOGL", "PTON", "CEI", "PCG", "ZVO", "META", "LCID", "ABEV", "MMAT", "GOOG", "UBER", "RIVN", "DAL", "AMC", "DKNG", "WISH", "WBD", "MU", "UMC", "KR", "CSCO", "TSM", "USB", "GOLD", "OPEN", "NKLA", "PBTS", "OXY", "ARVL", "BEKE", "TLRY", "KMI", "BB", "PFE", "MPW", "FFIE", "GRAB", "SQ", "HPE", "CCJ", "ASX", "KEY", "KO", "MARA", "CSX", "RIG", "NFLX", "SIRI", "XOM", "BCS", "KGC", "CS", "RBLX", "SYTA", "LUMN", "CPG", "AGNC", "FCX"],
    "@TOP50":
        ["AMD", "TSLA", "AAPL", "MULN", "NVDA", "AMZN", "AGFY", "CCL", "NIO", "BAC",  "F", "INTC", "T", "FNHC", "VALE", "ACI", "ILAG", "IMRA", "PLTR", "SOFI", "AAL", "WFC", "NU", "JPM", "XPEV", "SHOP", "MSFT", "NOK", "DBGI", "C", "PBR", "CMCSA", "SNAP", "HBAN", "DNA", "PLUG", "SWN", "VZ", "LASE", "NCLH", "GOOGL", "PTON", "CEI", "PCG", "ZVO", "META", "LCID", "ABEV"],
    "@TOP25":
        ["AMD", "TSLA", "AAPL", "MULN", "NVDA", "AMZN", "AGFY", "CCL", "NIO", "BAC", "ATXI", "F", "INTC", "T", "FNHC", "VALE", "ACI", "ILAG", "IMRA", "PLTR", "SOFI", "AAL", "WFC", "NU", "JPM"],

    #lista de seguimiento @FOLO1, lista Aux seguimiento @FOLO2 ambas @FOLO3 19/10/2022
    "@FOLO1":
        ["UPST", "MELI", "TWLO", "RIVN", "SNOW", "LYFT", "ADBE", "UBER", "ZI", "QCOM", "PYPL", "SPOT", "RUN", "GTLB", "MDB", "NVDA", "AMD", "ADSK", "AMZN", "BABA", "NFLX", "FFIV", "GOOG", "MSFT", "ABNB", "TSLA", "META"],
    "@FOLO2":
        ["DBX", "PTON", "CRWD", "NVST", "HUBS", "EPAM", "PINS", "TTD", "SNAP", "APPS", "ASAN", "AFRM", "DOCN", "ETSY", "DDOG", "SHOP", "NIO", "U", "GME", "RBLX", "CRSR"],
    "@FOLO3":
        ["UPST", "MELI", "TWLO", "RIVN", "SNOW", "LYFT", "ADBE", "UBER", "ZI", "QCOM", "PYPL", "SPOT", "RUN", "GTLB", "MDB", "NVDA", "AMD", "ADSK", "AMZN", "BABA", "NFLX", "FFIV", "GOOG", "MSFT", "ABNB", "TSLA", "META", "DBX", "PTON", "CRWD", "NVST", "HUBS", "EPAM", "PINS", "TTD", "SNAP", "APPS", "ASAN", "AFRM", "DOCN", "ETSY", "DDOG", "SHOP", "NIO", "U", "GME", "RBLX", "CRSR"]
    }

class MODEL_TYPE_COLM(Enum):
    VGOOD = "_vgood16_"
    GOOD = "_good9_"
    REG = "_reg4_"
    LOW = "_low1_"

class Option_Historical(Enum):
    YEARS_3 = 1
    MONTH_3 = 2
    MONTH_3_AD = 3
    DAY_6 = 4
    DAY_1 = 5

class ExtendedEnum(Enum):
    @classmethod
    def list_values(cls):
        return list(map(lambda c: c.value, cls))
    @classmethod
    def list(cls):
        return list(map(lambda c: c, cls))
class Op_buy_sell(ExtendedEnum):
    BOTH = "both"
    POS = "pos"
    NEG = "neg"



DICT_WEBULL_ID = {"UPST" : 950177837,
    "MELI" : 913323930,
    "TWLO" : 913254831,
    "RIVN" : 950188536,
    "SNOW" : 950173560,
    "LYFT" : 950116149,
    "ADBE" : 913256192,
    "UBER" : 950121423,
    "ZI" : 950157730,
    "QCOM" : 913323878,
    "PYPL" : 913256043,
    "SPOT" : 925418520,
    "RUN" : 913256036,
    "GTLB" : 950188178,
    "MDB" : 925377113,
    "NVDA" : 913257561,
    "AMD" : 913254235,
    "ADSK" : 913256187,
    "AMZN" : 913256180,
    "BABA" : 913254558,
    "NFLX" : 913257027,
    "FFIV" : 913256674,
    "GOOG" : 913303964,
    "MSFT" : 913323997,
    "ABNB" : 950178075,
    "TSLA" : 913255598,
    "META" : 913303928,
    "DBX" : 925418496,
    "PTON" : 950138392,
    "CRWD" : 950126602,
    "NVST" : 950135009,
    "HUBS" : 913254682,
    "EPAM" : 913254390,
    "PINS" : 950118597,
    "TTD" : 913431510,
    "SNAP" : 925186755,
    "APPS" : 913253434,
    "ASAN" : 950172459,
    "AFRM" : 950178219,
    "DOCN" : 950181409,
    "ETSY" : 913255993,
    "DDOG" : 950136998,
    "SHOP" : 913254746,
    "NIO" : 950076017,
    "U" : 950172451,
    "GME" : 913255341,
    "RBLX" : 950178170,
    "CRSR" : 950172441}


