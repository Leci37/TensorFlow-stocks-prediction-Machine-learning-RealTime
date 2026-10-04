from typing import List
import numpy as np
import math

def vol(returns: List[float]) -> float:
    # Return the standard deviation of returns
    return np.std(returns)

def lpm(returns: List[float], threshold: float, order: float) -> float:
    # This method returns a lower partial moment of the returns
    # Create an array he same length as returns containing the minimum return threshold
    threshold_array = np.empty(len(returns))
    threshold_array.fill(threshold)
    # Calculate the difference between the threshold and the returns
    diff = threshold_array - returns
    # Set the minimum of each to 0
    diff = diff.clip(min=0)
    # Return the sum of the different to the power of order
    return np.sum(diff ** order) / len(returns)

def hpm(returns: List[float], threshold: float, order: float) -> float:
    # This method returns a higher partial moment of the returns
    # Create an array he same length as returns containing the minimum return threshold
    threshold_array = np.empty(len(returns))
    threshold_array.fill(threshold)
    # Calculate the difference between the returns and the threshold
    diff = returns - threshold_array
    # Set the minimum of each to 0
    diff = diff.clip(min=0)
    # Return the sum of the different to the power of order
    return np.sum(diff ** order) / len(returns)

def prices(returns: List[float], base: float):
    # Converts returns into prices
    s = [base]
    for i in range(len(returns)):
        s.append(base * (1 + returns[i]))
    return np.array(s)

def dd(returns: List[float], tau: float):
    # Returns the draw-down given time period tau
    values = prices(returns, 100)
    pos = len(values) - 1
    pre = pos - tau
    drawdown = float('+inf')
    # Find the maximum drawdown given tau
    while pre >= 0:
        dd_i = (values[pos] / values[pre]) - 1
        if dd_i < drawdown:
            drawdown = dd_i
        pos, pre = pos - 1, pre - 1
    # Drawdown should be positive
    return abs(drawdown)

def max_dd(returns: List[float]):
    # Returns the maximum draw-down for any tau in (0, T) where T is the length of the return series
    max_drawdown = float('-inf')
    for i in range(0, len(returns)):
        drawdown_i = dd(returns, i)
        if drawdown_i > max_drawdown:
            max_drawdown = drawdown_i
    # Max draw-down should be positive
    return abs(max_drawdown)

def sharpe_ratio(er: float, returns: List[float], rf: float):
    a = (er - rf)
    b = vol(returns)
    if b == 0: return a
    else: return a/b

def omega_ratio(er: float, returns: List[float], rf: float, target=0):
    a = (er - rf)
    b = lpm(returns, target, 1)
    if b == 0: return a
    else: return a / b

def sortino_ratio(er: float, returns: List[float], rf: float, target=0):
    a = (er - rf)
    b = math.sqrt(lpm(returns, target, 2))
    if b == 0: return a
    else: return a / b

def kappa_three_ratio(er: float, returns: List[float], rf: float, target=0):
    a = (er - rf)
    b = math.pow(lpm(returns, target, 3), float(1/3))
    if b == 0: return a
    else: return a/b

def gain_loss_ratio(returns: List[float], target=0):
    a = hpm(returns, target, 1)
    b = lpm(returns, target, 1)
    if b == 0: return a
    else: return a / b

def upside_potential_ratio(returns: List[float], target=0):
    a = hpm(returns, target, 1)
    b = math.sqrt(lpm(returns, target, 2))
    if b == 0: return a
    else: return a / b

def calmar_ratio(er: float, returns: List[float], rf: float):
    a = (er - rf)
    b = max_dd(returns)
    if b == 0: return a
    else: return a / b