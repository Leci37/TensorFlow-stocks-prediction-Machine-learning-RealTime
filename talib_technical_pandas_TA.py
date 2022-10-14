#https://github.com/twopirllc/pandas-ta
import numpy as np
import pandas as pd
import pandas_ta as ta

import UtilsL

def get_all_pandas_TA_tecnical(df_ta):
    #https://github.com/twopirllc/pandas-ta#available-metrics

    ### Cycles (1)
    # Even Better Sinewave: ebsw
    df_ta.ta.ebsw(prefix="cycl", cumulative=True, append=True)
    ### Momentum (41)
    # Awesome Oscillator: ao
    df_ta.ta.ao(prefix="mtum", cumulative=True, append=True)
    # Bias: bias
    df_ta.ta.bias(prefix="mtum", cumulative=True, append=True)
    # BRAR: brar
    df_ta.ta.brar(prefix="mtum", cumulative=True, append=True)
    # Chande Forecast Oscillator: cfo
    df_ta.ta.cfo(prefix="mtum", cumulative=True, append=True)
    # Center of Gravity: cg
    df_ta.ta.cg(prefix="mtum", cumulative=True, append=True)
    # Correlation Trend Indicator: cti
    df_ta.ta.cti(prefix="mtum", cumulative=True, append=True)
    # Directional Movement: dm
    df_ta.ta.dm(prefix="mtum", cumulative=True, append=True)
    # Efficiency Ratio: er
    df_ta.ta.er(prefix="mtum", cumulative=True, append=True)
    # Elder Ray Index: eri
    df_ta.ta.eri(prefix="mtum", cumulative=True, append=True)
    # Fisher Transform: fisher
    df_ta.ta.fisher(prefix="mtum", cumulative=True, append=True)
    # Inertia: inertia
    df_ta.ta.inertia(prefix="mtum", cumulative=True, append=True)
    # KDJ: kdj
    df_ta.ta.kdj(prefix="mtum", cumulative=True, append=True)
    # Pretty Good Oscillator: pgo
    df_ta.ta.pgo(prefix="mtum", cumulative=True, append=True)
    # Psychological Line: psl
    df_ta.ta.psl(prefix="mtum", cumulative=True, append=True)
    # Percentage Volume Oscillator: pvo
    df_ta.ta.pvo(prefix="mtum", cumulative=True, append=True)
    # Quantitative Qualitative Estimation: qqe
    df_ta.ta.qqe(prefix="mtum", cumulative=True, append=True)
    # Relative Strength Xtra: rsx
    df_ta.ta.rsx(prefix="mtum", cumulative=True, append=True)
    # Schaff Trend Cycle: stc
    df_ta.ta.stc(prefix="mtum", cumulative=True, append=True)
    # SMI Ergodic smi
    df_ta.ta.smi(prefix="mtum", cumulative=True, append=True)
    # TD Sequential: td_seq No me gusta el resultado , se usara otro en   TD Sequential ichimucu.py
    # df.ta.td_seq(prefix="mtum", cumulative=True, append=True)

    ### Overlap (33)
    # Arnaud Legoux Moving Average: alma
    df_ta.ta.alma(prefix="olap", cumulative=True, append=True)
    # Holt-Winter Moving Average: hwma
    df_ta.ta.hwma(prefix="olap", cumulative=True, append=True)
    # Jurik Moving Average: jma
    df_ta.ta.jma(prefix="olap", cumulative=True, append=True)
    # McGinley Dynamic: mcgd
    df_ta.ta.mcgd(prefix="olap", cumulative=True, append=True)
    # Pascal's Weighted Moving Average: pwma
    df_ta.ta.pwma(prefix="olap", cumulative=True, append=True)
    # Sine Weighted Moving Average: sinwma
    df_ta.ta.sinwma(prefix="olap", cumulative=True, append=True)
    # Ehler's Super Smoother Filter: ssf
    df_ta.ta.ssf(prefix="olap", cumulative=True, append=True)
    # Symmetric Weighted Moving Average: swma
    df_ta.ta.swma(prefix="olap", cumulative=True, append=True)
    # Volume Weighted Average Price: vwap
    # df.ta.vwap(prefix="olap", cumulative=True, append=True)
    # Volume Weighted Moving Average: vwma
    df_ta.ta.vwma(prefix="olap", cumulative=True, append=True)

    ### Performance (3)
    # Draw Down: drawdown   It does not exist
    # df.ta.draw_down(prefix="perf", cumulative=True, append=True)
    # Log Return: log_return
    df_ta.ta.log_return(prefix="perf", cumulative=True, append=True)
    # Percent Return: percent_return
    df_ta.ta.percent_return(prefix="perf", cumulative=True, append=True)

    df_ta["perf_z_30_1"] = df_ta.ta.cdl_z(prefix="ta")["ta_close_Z_30_1"]
    df_ta["perf_ha"] = df_ta.ta.ha(prefix="ta")["ta_HA_close"]

    ### Statistics (11)
    # Entropy: entropy
    df_ta.ta.entropy(prefix="sti", cumulative=True, append=True)
    # Kurtosis: kurtosis
    df_ta.ta.kurtosis(prefix="sti", cumulative=True, append=True)
    # Think or Swim Standard Deviation All: tos_stdevall
    df_ta.ta.tos_stdevall(prefix="sti", cumulative=True, append=True)
    # Z Score: zscore
    df_ta.ta.zscore(prefix="sti", cumulative=True, append=True)

    ### Trend (18)
    # Formally: linear_decay
    df_ta.ta.decay(prefix="tend", cumulative=True, append=True)
    # Parabolic Stop and Reverse: psar
    df_ta.ta.psar(prefix="tend", cumulative=True, append=True)
    # Trend Signals: tsignals
    df_ta.ta.tsignals(prefix="tend", cumulative=True, append=True)
    # Vertical Horizontal Filter: vhf
    df_ta.ta.vhf(prefix="tend", cumulative=True, append=True)
    # Cross Signals: xsignals
    df_ta.ta.xsignals(prefix="tend", cumulative=True, append=True)

    ### Volatility (14)
    # Holt-Winter Channel: hwc
    df_ta.ta.hwc(prefix="vola", cumulative=True, append=True)
    # Keltner Channel: kc
    df_ta.ta.kc(prefix="vola", cumulative=True, append=True)
    # Relative Volatility Index: rvi
    df_ta.ta.rvi(prefix="vola", cumulative=True, append=True)
    # Elder's Thermometer: thermo
    df_ta.ta.thermo(prefix="vola", cumulative=True, append=True)
    # True Range: true_range
    df_ta.ta.true_range(prefix="vola", cumulative=True, append=True)
    # Ulcer Index: ui
    df_ta.ta.ui(prefix="vola", cumulative=True, append=True)
    ### Volume (15)
    # Elder's Force Index: efi
    df_ta.ta.efi(prefix="volu", cumulative=True, append=True)
    # Klinger Volume Oscillator: kvo tarda muchas filas en dar datos
    # df.ta.kvo(prefix="volu", cumulative=True, append=True)
    # Negative Volume Index: nvi
    df_ta.ta.nvi(prefix="volu", cumulative=True, append=True)
    # Positive Volume Index: pvi
    df_ta.ta.pvi(prefix="volu", cumulative=True, append=True)
    # Price-Volume: pvol
    df_ta.ta.pvol(prefix="volu", cumulative=True, append=True)
    # Price Volume Rank: pvr
    df_ta.ta.pvr(prefix="volu", cumulative=True, append=True)
    # Price Volume Trend: pvt
    df_ta.ta.pvt(prefix="volu", cumulative=True, append=True)
    # Volume Profile: vp solo da una fila
    # df.ta.vp(prefix="volu", cumulative=True, append=True)
    # result = ta.cagr(df.close)
    df_ta = df_ta.drop(columns=['mtum_QQE_14_5_4.236'])
    df_ta[['mtum_QQEl_14_5_4.236', 'mtum_QQEs_14_5_4.236', 'tend_PSARl_0.02_0.2', 'tend_PSARs_0.02_0.2']] = df_ta[
        ['mtum_QQEl_14_5_4.236', 'mtum_QQEs_14_5_4.236', 'tend_PSARl_0.02_0.2', 'tend_PSARs_0.02_0.2']].replace(np.nan,0)

    df_ta = UtilsL.replace_bat_chars_in_columns_name(df_ta, "")

    return df_ta



# df = yhoo_history_stock.get_historial_data_3_month("MELI", prepos=False) #df = pd.DataFrame() # Empty DataFrame
#
# df = get_all_pandas_TA_tecnical(df)
#
# df.head()
