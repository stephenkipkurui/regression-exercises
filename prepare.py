# Clean data function
import acquire
import os
import env
import pandas as pd


iris = acquire.get_iris_data()

def prep_iris(iris):

    iris = iris.drop_duplicates()

    iris = iris.drop(columns=['Unnamed: 0'])

    iris_dummy = pd.get_dummies(iris[['sepal_width', 'petal_width']],
                                drop_first=[True, True])

    iris = pd.concat([iris, iris_dummy], axis=1)

    return iris.drop(columns=['sepal_width', 'petal_width'])




row_titanic = acquire.get_titanic_data()

def prep_titanic(row_titanic):
    titanic_df = row_titanic.drop_duplicates()

    titanic_df = titanic_df.drop(
        columns=['deck', 'embarked', 'class', 'age', 'Unnamed: 0'])

    titanic_df['embark_town'] = titanic_df.embark_town.fillna(
        value='Southampton')

    dummy_titanic = pd.get_dummies(
        titanic_df[['sex', 'embark_town']], drop_first=[True, True])

    titanic = pd.concat([dummy_titanic, titanic_df], axis=1)

    return titanic.drop(columns=['sex', 'embark_town'])




telco_data = acquire.get_telco_data()

def prep_telco(telco_data):
    
    t_df = telco_data.drop_duplicates()
    
    t_df = t_df.drop(columns = ['Unnamed: 0', 'tech_support', 'streaming_tv', 'paperless_billing', 
                   'streaming_movies', 'device_protection', 'online_backup', 'online_security'])
    
    dummy_telco_data = pd.get_dummies(t_df[['churn','internet_service_type','payment_type']], 
                                      drop_first = [True, True, True])
    
    t_df = pd.concat([t_df, dummy_telco_data], axis = 1)
    
    return t_df.drop(columns = ['churn','internet_service_type','payment_type'])