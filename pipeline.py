import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.metrics import mean_squared_error
 

# TransformerMixin: add method ".fit_transform()"
# BaseEstimator: add methods ".get_params()" and ".set_params()"
# We need 3 methods:
# 1) .fit()
# 2) .transform()
# 3) .fit_transform() (provided by "TransformerMixin")
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    # avoid "*args" or "**kargs" in "__init__"
    def __init__(self):
        self.mean_Month = pd.DataFrame()
        self.mean_Store = pd.DataFrame()
        self.mean_DayOfWeek = pd.DataFrame()
        self.mean_Promo_Store = pd.DataFrame()

    # fit is needed later for the pipilene
    def fit(self, X, y=None):

        # Date
        Date_2 = pd.to_datetime(X["Date"], format="%Y-%m-%d")
        X["Month"] = Date_2.dt.month
        self.mean_Month = self.mean_encode(X, "Month", "Sales")

        # Store
        self.mean_Store = self.mean_encode(X, "Store", "Sales")

        # DayOfWeek
        self.mean_DayOfWeek = self.mean_encode(X, "DayOfWeek", "Sales")

        # Promo (separately for each Store)
        self.mean_Promo_Store = self.mean_encode_2(X, "Promo", "Store", "Sales")

        return self
    
    def transform(self, X):
        # Since I use MEAN ENCODING, "X" must include
        # the terget variable. Below, just before returning
        # the transformed X, the target variable is dropped.

        # Date
        Date_2 = pd.to_datetime(X["Date"], format="%Y-%m-%d")
        X["Month"] = Date_2.dt.month
        X = pd.merge(X, self.mean_Month, how="left", on="Month")
        # drop: "Date" and "Month"

        # Store
        #X = self.mean_encode(X, "Store", "Sales")
        X = pd.merge(X, self.mean_Store, how="left", on="Store")
        # drop: "Store"

        # DayOfWeek
        # X = self.mean_encode(X, "DayOfWeek", "Sales")
        X = pd.merge(X, self.mean_DayOfWeek, how="left", on="DayOfWeek")
        # drop: "DayOfWeek"

        # Promo (separately for each Store)
        # X = self.mean_encode_2(X, "Promo", "Store", "Sales")
        X = pd.merge(X, self.mean_Promo_Store, how="left", on=["Promo", "Store"])
        # drop: "Promo" and "Store"

        # SchoolHoliday
        X.loc[X.SchoolHoliday=="0", :] = 0.0
        # keep: "SchoolHoliday"

        # StoreType: keep, no transformation

        # Assortment: keep, no transformation

        # Promo2: keep, no transformation

        # CompetitionDistance
        nb = 10 # number of bins
        clip_upper = 10000
        X["CD_clip"] = X["CompetitionDistance"].clip(upper=clip_upper)
        CD_clip_bins = pd.cut(
            X["CD_clip"],
            bins=nb,
            labels=[i for i in range(nb)])
        X['CD_clip_bins'] = pd.to_numeric(CD_clip_bins)
        X["CD_clip_bins_clip"] = X["CD_clip_bins"].clip(upper=clip_upper) # 
        # drop: "CompetitionDistance", "CD_clip", "CD_clip_bins"

        # Drop unused columns
        cols_to_drop = [
            "Date", "Month", "Store", "DayOfWeek", "Customers", "Open", "Promo",
            "StateHoliday", "CompetitionDistance", "CD_clip", "CD_clip_bins",
            "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear", "Promo2SinceWeek",
            "Promo2SinceYear", "PromoInterval"]
        X.drop(columns=cols_to_drop, inplace=True)

        # Drop the target
        with_target = sum([col == "Sales" for col in X.columns])
        if with_target > 0:
            target_to_drop = ["Sales"]
            X.drop(columns=target_to_drop, inplace=True)

        return X
    

    def mean_encode(self, df: pd.DataFrame, feature: str, target: str):
        """
        df: dataframe with "feature" and "target" columns
        feature: feature to transform
        target: target variable
        """
        new_col_name = feature + "_mean"
        df_enc = (
            # select columns
            df.loc[:, [feature, target]]
            # group by feature
            .groupby(feature)
            # aggregate over feature using target mean
            .agg(tmp_name=(target, np.mean))
            # index (i.e., feature categories) as a column
            .reset_index()
            # rename the column with the aggregated means
            .rename(columns={"tmp_name":new_col_name})
        )
    
        return df_enc
        
        
    def mean_encode_2(self, df: pd.DataFrame, feature1: str, feature2: str, target: str):
        """
        Same as "mean_encode" but with 2 features.
        df: dataframe with "feature" and "target" columns
        feature: feature to transform
        target: target variable
        """
        new_col_name = feature1 + feature2 + "_mean"
        df_enc = (
            # select columns
            df.loc[:, [feature1, feature2, target]]
            # group by feature
            .groupby([feature1, feature2])
            # aggregate over feature using target mean
            .agg(tmp_name = (target, np.mean))
            # index (i.e., feature categories) as a column
            .reset_index()
            # rename the column with the aggregated means
            .rename(columns={"tmp_name":new_col_name})
            )
    
        return df_enc
    

# Numerical pipeline
#
# All (except the last) estimators must be transformers (i.e., they
# must have a ".fit_transform()" method).
num_pipeline = Pipeline([
    # replace NA with mean
    ('imputer', SimpleImputer(strategy='mean')),
    # standardize the variables: z = (x - mean) / SD
    ('std_scaler', StandardScaler())])


# Categorical pipeline
#
# All (except the last) estimators must be transformers (i.e., they
# must have a ".fit_transform()" method).
cat_pipeline = Pipeline([
    # replace NA with mode
    ('imputer', SimpleImputer(strategy='most_frequent')),
    # apply "OneHotEncoder()"
    ('one_hot', OneHotEncoder(drop='if_binary'))])


list_num_attribs = ["SchoolHoliday", "Promo2", "Month_mean", "Store_mean",
                    "DayOfWeek_mean", "PromoStore_mean", "CD_clip_bins_clip"]
list_cat_attribs = ["StoreType", "Assortment"]


# ColumnTransformer requires tuples with:
# - a name
# - a transformer
# - a list of names (or indices) of columns to which the transformer is applied
cols_transformer = ColumnTransformer([
    # apply "num_pipeline" to numerical columns
    ('num', num_pipeline, list_num_attribs),
    # apply "cat_pipeline" to categorical columns
    ('cat', cat_pipeline, list_cat_attribs)])


full_pipeline = Pipeline([
    # transform/add columns
    ('attribs_adder', CombinedAttributesAdder()),
    # Transform numerical and categorical attributes
    ("cols_transformer", cols_transformer)])
