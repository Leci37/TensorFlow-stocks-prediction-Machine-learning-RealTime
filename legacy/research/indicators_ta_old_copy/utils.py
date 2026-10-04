def get_crash_points(df, col_name_A, col_name_B, col_result, highlight_result_in_next_cell = 1 ):
    df["diff"] = df[col_name_A] - df[col_name_B]
    df[col_result] = 0

    df.loc[((df["diff"] >= 0) & (df["diff"].shift() < 0)), col_result] = 1
    df.loc[((df["diff"] <= 0) & (df["diff"].shift() > 0)), col_result] = -1
    #TODO test with oder numer than 1
    if highlight_result_in_next_cell > 0:
        df.loc[((df[col_result].shift(highlight_result_in_next_cell) == 1)), col_result] = 1
        df.loc[((df[col_result].shift(highlight_result_in_next_cell) == -1)), col_result] = -1

    df = df.drop(columns=['diff'])

    return df


def replace_bat_chars_in_columns_name(df, char_new ="_"):
    for c in df.columns:
        new_colum = str(c).replace(" ",char_new).replace("%", "per").replace(":", char_new).replace(",", char_new).replace("/", char_new).replace(".", char_new).replace("-",                                                                                           char_new).replace(
            "(", char_new).replace(")", '').replace('\\', char_new).replace('\'', char_new)
        df = df.rename(columns={ c:  new_colum})
    return df


def add_rename_all_columns_df(df, prefix="", sufix=""):
    for c in df.columns:
        df = df.rename(columns={ c:  str(prefix)+ c + str(sufix)})

    return df