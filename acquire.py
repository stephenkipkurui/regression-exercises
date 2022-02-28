from genericpath import isfile
import os
import env
import pandas as pd


def conn(db, username=env.username, host=env.host, password=env.password):

    # url = f'mysql+pymysql://{env.username}:{env.password}@{env.host}/{db}'

    return f'mysql+pymysql://{env.username}:{env.password}@{env.host}/{db}'


def get_titanic_data():

    db = 'titanic_db'

    if os.path.isfile(db):

        return pd.read_csv(db)
    else:

        qry = '''SELECT * FROM passengers'''

        df_titanic = pd.read_sql(qry, conn(db))

        df_titanic = pd.DataFrame(df_titanic)

        df_titanic.to_file(db)

        return df_titanic


def get_iris_data():

    db = 'iris_db'

    if os.path.isfile(db):

        return pd.read_csv(db)

    else:

        iris_query = '''SELECT * FROM species'''

        df_iris_species = pd.read_sql(iris_query, conn(db))

        df_iris_species = pd.DataFrame(df_iris_species)

        df_iris_species.to_file(db)

        return df_iris_species

def get_telco_data():

    db = 'telco_churn'

    if os.path.isfile(db):

        return pd.read_csv(db)

    else:

        telco_qry = '''
                SELECT *
                FROM customers
                JOIN internet_service_types USING (internet_service_type_id)
                JOIN contract_types USING (contract_type_id)
                JOIN payment_types USING (payment_type_id)
                                
                # SELECT cp.monthly_charges, cp.total_charges, ist.internet_service_type, ct.contract_type_id
                # FROM  customer_payments cp 
                # JOIN customers c ON cp.payment_type_id = c.payment_type_id 
                # JOIN contract_types ct ON c.contract_type_id = ct.contract_type_id
                # JOIN internet_service_types ist ON c.internet_service_type_id = ist.internet_service_type_id;             
             '''
        telco_df = pd.read_sql(telco_qry, conn(db))

        telco_df = pd.DataFrame(telco_df)

        telco_df.to_file(db)

        return telco_df
