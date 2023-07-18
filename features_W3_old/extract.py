import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
import pickle
from itertools import combinations

def pca_extraction(df: pd.DataFrame, pca_comp=25, pca_path="data/pca.pkl", scaler_path="data/scaler.pkl", split_ratio=None, debug=False):
    df = df.copy()

    # Part 4 : Dimension Reduction (PCA need to standardised before fitting)
    columns = df.columns
    if pca_comp is not None:
        # pca_comp = True -> Transform only
        if (pca_comp == True) and (isinstance(pca_comp, bool)):
            pca: PCA = pickle.load(open(pca_path, 'rb'))
            scaler: StandardScaler = pickle.load(open(scaler_path, 'rb'))
        else:
            train_size = int(len(df) * split_ratio)

            scaler = StandardScaler()
            train_data = scaler.fit_transform(df[:train_size])

            pca = PCA(n_components=pca_comp)
            pca = pca.fit(train_data)
            pickle.dump(pca, open(pca_path, 'wb'))
            pickle.dump(scaler, open(scaler_path, 'wb'))
        if debug: print(f'Reducing Dimensionality from {len(columns)} features to {pca_comp}')
        index = df.index
        df = pca.transform(df)
        df = pd.DataFrame(df, index=index)
    return df

def xgb_selection(df: pd.DataFrame, y=None, num_features=25, n_estimators=200, xgb_path="data/boost.json", split_ratio=None, debug=False):    
    df = df.copy()

    # Part 4 : Feature Selection
    if y is not None:
        y = y.copy()
        y = y[y.index.isin(df.index)]
        df = df[df.index.isin(y.index)]

        assert len(df) == len(y)
        assert sorted(df.index) == sorted(y.index)

        model = XGBClassifier(n_estimators=n_estimators, objective='binary:logistic')

        train_size = int(len(df) * split_ratio)
        X_train = df[:train_size]
        y_train = y[:train_size]
        model.fit(X_train, y_train)
        model.save_model(xgb_path)
    else:
        model = XGBClassifier(n_estimators=n_estimators, objective='binary:logistic')
        model.load_model(xgb_path)

    top_features_indices = np.argsort(model.feature_importances_)[::-1][:num_features]
    columns = df.columns[top_features_indices]

    return df[columns]

def uncorrelate_selection(df: pd.DataFrame, num_features=25, filter_binary=True, split_ratio=None, column_path='data/columns.pkl', debug=False):
    def is_binary(series: pd.Series): return (series.nunique() < 30)

    df = df.copy()

    # Only Do it on training data
    if split_ratio is not None:
        train_size = int(len(df) * split_ratio)
        train_data = df[:train_size].copy()

        # Remove Binary Columns
        if filter_binary:
            binary_cols = [col for col in train_data.columns if is_binary(train_data[col])]
            train_data.drop(binary_cols, axis=1, inplace=True)
        train_data.dropna(inplace=True)

        # Get the correlation table
        corr = train_data.corr()
        cols = set(corr.columns)
        output = set()

        # Find minimum correlation features
        for _ in range(num_features):
            min_corr = 1
            min_comb = None
            
            for col1, col2 in combinations(cols, 2):
                if abs(corr[col1][col2]) < min_corr:
                    min_corr = abs(corr[col1][col2])
                    # print(min_corr)
                    min_comb = (col1, col2)

            col1, col2 = min_comb
            output.add(col1)
            output.add(col2)
            
            cols.remove(col1)
            cols.remove(col2)
            corr = corr.drop(col1, axis=1).drop(col1, axis=0)
            corr = corr.drop(col2, axis=1).drop(col2, axis=0)

        pickle.dump(list(output), open(column_path, 'wb'))
        return df[list(output)]
    else:
        output = pickle.load(open(column_path, 'rb'))
        return df[output]
    
def kbest_chi2_selection(df: pd.DataFrame, y=None, num_features=25, split_ratio=None, column_path='data/columns.pkl'):
    from sklearn.feature_selection import SelectKBest, chi2
    from sklearn.preprocessing import MinMaxScaler

    df = df.copy()

    # Only do it on training data
    if split_ratio is not None:
        train_df = df.copy()[:int(len(df) * split_ratio)]
        y = y[y.index.isin(train_df.index)]
        x_scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(train_df)
        selector = SelectKBest(chi2, k=num_features)
        selector = selector.fit(x_scaled, y)
        top_n_idx = selector.get_support(indices=True) % len(train_df.columns)
        top_n_features = [train_df.columns[index] for index in top_n_idx]
        pickle.dump(top_n_features, open(column_path, 'wb'))
    else:
        top_n_features = pickle.load(open(column_path), 'rb')
    
    return df[top_n_features]


''' ExtraTreesClassifier '''
def get_tree_correlation(features_X_ALL, target_Y, num_features = 20):
    # print(" ExtraTreesClassifier ")
    model = ExtraTreesClassifier()
    model.fit(features_X_ALL, target_Y)
    # print(model.feature_importances_)  # use inbuilt class feature_importances of tree based classifiers
    # plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=features_X_ALL.columns)
    # feat_importances.nlargest(20).plot(kind='barh')
    # plt.show()
    feat_importances = feat_importances.sort_values(ascending=False)[:num_features]
    feat_importances = feat_importances.reset_index(level=0)
    df_result = pd.DataFrame()
    df_result['ExtraTrees'] = feat_importances['index']
    df_result['ExtraTrees_points'] = feat_importances[0]
    return df_result['ExtraTrees'].values