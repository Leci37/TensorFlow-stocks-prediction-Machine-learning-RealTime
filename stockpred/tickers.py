"""Stock lists by key (from the old _KEYS_DICT.py). Scripts pick one, e.g. DICT_COMPANYS["@CHILL"]."""

DICT_COMPANYS = {
    "@CHILL":
        ["UBER", "PYPL"],
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
    "@FOLO3": #"META", ERROR no buy points "GOOG", "MSFT", "TSLA", #"
        [ "GOOG","MSFT", "TSLA","UPST", "MELI", "TWLO", "RIVN", "SNOW", "LYFT", "ADBE", "UBER", "ZI", "QCOM", "PYPL", "SPOT", "GTLB", "MDB", "NVDA", "AMD" , "ADSK", "AMZN", "CRWD", "NVST", "HUBS", "EPAM", "PINS", "TTD", "SNAP", "APPS", "ASAN", "AFRM", "DOCN", "ETSY", "DDOG", "SHOP", "NIO", "U", "GME", "RBLX", "CRSR"],
                #"PTON", error callearly no se xq
         #[  "CRWD", "NVST", "HUBS", "EPAM", "PINS", "TTD", "SNAP", "APPS", "ASAN", "AFRM", "DOCN", "ETSY", "DDOG", "SHOP", "NIO", "U", "GME", "RBLX", "CRSR"],
    "@CHIC":# "MU", "CRM", "SNPS", "DHI", "MPWR", "CZR", "NOW", "BBWI",
        [   "TER", "KLAC", "ALGN", "UONE", "SPG", "STAG" ],#"O", "CARV","ATHE", "DXCM","PSEC"  ERROR
    "@CHIC3":#"ATHE", "MU", "CRM", "SNPS", "DHI", "MPWR", "CZR", "NOW", "BBWI", "DXCM",
        ["DBX", "BABA", "FFVI","CVNA",  "KLAC", "ALGN", "CARV", "SPG", "STAG"], #, "FFVI","CVNA",  "TER", "KLAC", "ALGN", "CARV", "UONE", "SPG", "STAG", "O", "PSEC"],
    "@CRT":#etoro and xtb use
        ["BTC-USD", "ETH-USD", "DASH-USD", "LTC-USD", "XLM-USD",  "LINK-USD", "DOGE-USD",'BNB-USD', "ADA-USD",'DOT-USD', 'DYDX-USD'], #"ZEC-USD",,'BCH-USD'
        #['BTC-USD', 'ETH-USD', 'USDT-USD', , 'XRP-USD', 'ADA-USD', 'HEX-USD','SOL-USD', 'AVAX-USD', 'DOGE-USD', 'DOT-USD', 'DOT-USD', 'SHIB-USD', 'MATIC-USD'
        # ["BTC", "ETH", "BCH", "XRP", "DASH", "LTC", "ETC", "API3", "CRO", "SKL", "IMX", "ADA", "MIOTA", "XLM", "EOS", "NEO", "TRX", "ZEC", "BNB", "XTZ", "DOT", "MKR", "COMP", "LINK", "UNI", "YFI", "DOGE", "AAVE", "FIL", "ALGO", "ATOM", "MANA", "APE", "LRC", "ENJ", "BICO", "BAT", "BNT", "OGN", "MATIC", "FLR", "GALA", "ALICE", "CHZ", "HBAR", "DYDX", "SOL", "THETA", "FTM", "GRT"]

    }
