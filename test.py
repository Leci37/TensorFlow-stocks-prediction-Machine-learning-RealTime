

import pandas as pd

df = pd.DataFrame(columns=["Close"])

df["Close"] = range(100, 1, -1)

p = lambda x : x/100
def apply_perc(num, change):
    if change > 0:
        num += num*p(change)
    elif change < 0:
        num -= num*p(abs(change))
    return num


def numConcat(num1, num2):
       
    # Convert both the numbers to
    # strings
    num1 = str(int(num1))
    num2 = str(int(num2))
        
    # Concatenate the strings
    num1 += num2
        
    return int(num1)

def calc_profits(x):
    start_value = x.iat[0]
    end_value = x.iat[-1]
    change = ((end_value - start_value)/start_value)*100
    
    return change

df["pl"] = df["Close"].shift(-1).rolling( min_periods = 2, window=2).apply(calc_profits)

SALDO = 10000
def balance_final(x):
    global SALDO
    change = x.iat[0]
    SALDO = apply_perc(SALDO, change)
    return SALDO


df["SALDO"] = df["pl"].rolling(window=1).apply(balance_final)

print(SALDO)