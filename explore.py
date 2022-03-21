from operator import imod
import env
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split


from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression 


def connect():
    db = 'telco_churn'
    url = f'mysql+pymysql://{env.username}:{env.password}@{env.host}/{db}'
    return url


def get_telco_data(use_cache=True):

    telco_file = 'telco.csv'

    if os.path.exists(telco_file) and use_cache:

        print('Program Status: Acquiring local cached Telco data..')

        return pd.read_csv(telco_file)

    sql_query = '''
                  SELECT churn, tenure, monthly_charges, total_charges, payment_type_id, phone_service, multiple_lines, paperless_billing  FROM customers;
    
                '''
    print('Program Status: Acquiring data from online resource...')

    telco = pd.read_sql(sql_query, connect())

    print('Program Status: Saving resourced file to local csv file...')

    telco.to_csv(telco_file, index=False)


def telco_prepare():

    # call get_zillow_data function above to acquire the data from db/ local cache
    telco = get_telco_data()

    # Replace white spaces values with NaN values
    telco = telco.replace(r'^\s*$', np.nan, regex = True)

    # Drop all rows with NaN values
    telco = telco.dropna()
    
#     # Cast as type int
#     telco = telco['tenure', 'payment_type_id'].astype('int')
#     telco = telco['monthly_charges', 'total_charges'].astype('float')

    
    # Reset index
    telco = telco.reset_index(drop=True)
    
    # Split the data into train, validate and test
    train_validate, test = train_test_split(telco, test_size = 0.2, random_state = 1349)
    train, validate = train_test_split(train_validate, train_size = 0.7, random_state = 1349)
    
    return train, validate, test


def plot_variable_pairs(train):
    
#     plt.figure(figsize = (16, 10))
    
#     plt.title('Pair Plot Tenure VS Monthly Chargesxs')

    sns.pairplot(train[['tenure', 'monthly_charges']], corner = True)
     
    plt.show()
    
    
    
def months_to_years():
    
    df = get_telco_data()
    
    years = round((df.tenure/ 12), 1)
    
    draw = sns.lmplot(x="monthly_charges", y= 'tenure', data=df, line_kws={'color': 'red'})
    
    return draw
    
    

                 
                 
    
    
    
    
















