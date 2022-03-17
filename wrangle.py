from operator import imod
import env
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split


def connect():
    db = 'zillow'
    url = f'mysql+pymysql://{env.username}:{env.password}@{env.host}/{db}'
    return url


def get_zillow_data(use_cache=True):

    zillow_file = 'zillow.csv'

    if os.path.exists(zillow_file) and use_cache:

        print('Program Status: Acquiring local cached zillow data..')

        return pd.read_csv(zillow_file)

    sql_query = '''
         SELECT bedroomcnt, bathroomcnt, 
                calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips 
         FROM properties_2017 p17
         LEFT JOIN propertylandusetype plt ON (p17.propertylandusetypeid = plt.propertylandusetypeid)
         WHERE plt.propertylandusedesc = 'Single Family Residential';    
                '''
    print('Program Status: Acquiring data from online resource...')

    zillow = pd.read_sql(sql_query, connect())

    print('Program Status: Saving resourced file to local csv file...')

    zillow.to_csv(zillow_file, index=False)


def wrangle_zillow():

    # call get_zillow_data function above to acquire the data from db/ local cache
    zillow = get_zillow_data()

    # Replace white spaces values with NaN values
    zillow = zillow.replace(r'^\s*$', np.nan, regex=True)

    # Drop all rows with NaN values
    zillow = zillow.dropna()

    # Convert all columns to int64 dtypes
#     zillow = zillow.astype('int')
    
#     col = ['bedroom', 'bathroom','square_feet','tax_value','year_built','tax_amount','fips']
    
#     # Rename columns 
#     zillow = zillow.rename(columns = col)
    
    # Reset index
    zillow = zillow.reset_index(drop = True)
    
    # Split the data into train, validate and test
    train_validate, test = train_test_split(zillow, test_size = 0.2, random_state = 1349)
    train, validate = train_test_split(train_validate, train_size = 0.7, random_state = 1349)
    
    return train, validate, test


# SELECT p16.bedroomcnt AS bedroomcnt16, p16.bathroomcnt AS bathroomcnt16, 
#                 p16.calculatedfinishedsquarefeet AS calculatedfinishedsqrfeet16, p16.taxvaluedollarcnt AS taxvaluedollarcnt16, 
#                 p16.yearbuilt AS yearbuilt16, p16.taxamount AS taxamount16, p16.fips AS fips16,p17.bedroomcnt AS bedroomcnt17, 
#                 p17.bathroomcnt AS bathroomcnt17, p16.calculatedfinishedsquarefeet AS calculatedfinishedsqrfeet17,
#                 p17.taxvaluedollarcnt AS taxvaluedollarcnt17, p17.yearbuilt AS yearbuilt17, p17.taxamount AS taxamount17, 
#                 p17.fips AS fips17
#          FROM properties_2016 p16
#          JOIN properties_2017 p17 USING(parcelid)
#          JOIN propertylandusetype plt ON ((p16.propertylandusetypeid = plt.propertylandusetypeid) AND (p17.propertylandusetypeid = plt.propertylandusetypeid) )
#          WHERE plt.propertylandusedesc = 'Single Family Residential';
