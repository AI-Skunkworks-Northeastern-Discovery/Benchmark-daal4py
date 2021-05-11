'''
Author - Abhishek Maheshwarappa and Kartik Kumar

Converts all non numeric columns or categorical 
to numerical values

'''





import pandas as pd
import numpy as np
import time
from sklearn import preprocessing


class Numeric():

    def __init__(self, latency):
        self.latency = latency

    def convert_to_numeric(self, df,target,classification = True):

        # getting categorical, continous dtypes and  boolean dtypes

        categorical_columns = df.select_dtypes(include=['object']).columns.to_list()

        bool_columns = df.select_dtypes(include=['bool']).columns.to_list()

        continou_columns = df.select_dtypes(exclude=['object']).columns.to_list()

        continous_columns = [x for x in continou_columns if not x in bool_columns]

        # convert boolean into numerical values
        for col in bool_columns:
            df[col] = df[col].astype(int)

        print("Continous",continous_columns)

        # handling missing values
        for col in categorical_columns:
            df[col] = df[col].fillna(df[col].mode())
        for col in continous_columns:
            df[col] = df[col].fillna(df[col].mean())
        for col in bool_columns:
            df[col] = df[col].fillna(df[col].mode())

        if target in continous_columns: continous_columns.remove(target)


        for col in continous_columns:
            if len(df[col].unique())==2:
                continous_columns.remove(col)

        # Create x, where x the 'scores' column's values as floats
        x = df[continous_columns].values.astype(float)

        # Create a minimum and maximum processor object
        min_max_scaler = preprocessing.MinMaxScaler()

        # Create an object to transform the data to fit minmax processor
        x_scaled = min_max_scaler.fit_transform(x)

        # Run the normalizer on the dataframe
        df[continous_columns] = pd.DataFrame(x_scaled)

        # dict
        df_dict = {}

        print("before for", categorical_columns)


        # label encode the columns with two categorical values
        for col in categorical_columns:            
            if len(df[col].unique())>=2:
                print(col ,' - ', len(df[col].unique()))
                df[col] = df[col].astype("category")
                df_dict[col] = dict(
                    enumerate(df[col].cat.categories))
                df[col] = df[col].cat.codes

        if target in categorical_columns: categorical_columns.remove(target)

        unwated_cols = []

        '''
        Remove columns with too 
        many unqiue values in 
        categorical columns to 
        reduce the space of the data
        
        '''
        
        for col in categorical_columns:

            l = len(df[col].unique())    
            if l>5:
                unwated_cols.append(col)

        df = df.drop(unwated_cols,axis = 1)
        updated_cat_columns = [x for x in categorical_columns if not x in unwated_cols ]


        '''
        Generate dummy variables
        for the categorical columns
        '''
        
        if updated_cat_columns:
            data=pd.get_dummies(df[updated_cat_columns])
            data = data.apply(np.int64)
            df=df.drop(updated_cat_columns ,axis = 1)
            df = df.join(data)

        return df,df_dict