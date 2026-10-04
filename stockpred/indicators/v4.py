from ta import add_all_ta_features
import pandas as pd

def extract_features(df: pd.DataFrame):
    df = df.copy()
    df = add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume")
    df = df.drop(columns=['spread', 'trend_psar_down', 'trend_psar_up'])
    return df.dropna()