from pydataset import data 
import pandas as pd

def get_tips_data():
    '''
        This function call the pydataset and return the tips data
    '''
    
    tips = data('tips')
    
    return tips


def prep_tips():
    
    tips = get_tips_data()
    
    # drop missing values
    tips = tips.dropna()
    
    # drop duplicates
    tips = tips.drop_duplicates()
    
    return tips