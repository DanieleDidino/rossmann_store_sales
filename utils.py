import numpy as np
import pandas as pd
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
 

def convert_to_others(df: pd.DataFrame, feature: str, N_counts: int):
    """
    df: data frame
    feature: feature to transform
    N_counts: categories with less than "N_counts" counts are converted to "others" 
    """

    df_count = (
        df
        .groupby(feature)[feature]
        .value_counts()
        .reset_index()
    )

    # Select the categories with less than N_counts
    df_other = df_count.loc[df_count["count"] < N_counts, feature]

    # Name for the new column with some categories converted to "others"
    new_col_name = feature + "_others"

    # Copy original column
    df[new_col_name] = df[feature]
    # Categories with less than "N_counts" counts (this info is #
    # stored in the data frame "df_other") are set to "others"
    df.loc[df[feature].isin(df_other), [new_col_name]] = "others"

    return df


def RMSPE(y, y_pred):
     rmspe = np.sqrt(np.sum(((y - y_pred) / y)**2) / len(y))
     return rmspe


def RMSPE_score_func(y, y_pred, **kwargs):
     rmspe = np.sqrt(np.sum(((y - y_pred) / y)**2) / len(y))
     return rmspe


def print_best_model_metrics(gs, X, y):
    """"
    gs: fitted GridSearch object
    X: DataFrame with features
    y: actual target
    """
    print(f"Best parameters:\n{gs.best_params_}")

    print(f"\nBest score:")
    print(f"neg_mean_squared_error: {gs.best_score_:.3f}")
    print(f"RMSE: {np.sqrt(-1*gs.best_score_):.3f}")

    score = gs.score(X, y)
    print("\nConsidering the full train set:")
    print(f"neg_mean_squared_error: {score:.3f}")
    print(f"RMSE: {np.sqrt(-1*score):.3f}")

    y_pred = gs.predict(X)
    # print(f"Mean squared error = {mean_squared_error(y, y_pred, squared=False):.2f}")
    print(f"Root Mean Square Percentage Error: {RMSPE(y, y_pred):.2f}")
