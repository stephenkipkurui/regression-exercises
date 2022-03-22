from sklearn.model_selection import train_test_split
import sklearn.preprocessing
from sklearn.preprocessing import StandardScaler
from pydataset import data


def get_tips_data():
    '''
        This function call the pydataset and return the tips data
    '''
    tips = data('tips')
    
     # drop missing values
    tips = tips.dropna()
    
    # Encode categorical data
    tips['sex'] = tips.sex.map({'Male': 1, 'Female': 0})
    tips['smoker'] = tips.smoker.map({'Yes': 1, 'No': 0})
    tips['day'] = tips.day.map({'Sun': 7, 'Mon': 1, 'Tue':3, 'Wed': 3, 'Thur': 4, 'Fri':5, 'Sat':6})
    tips['time'] = tips.time.map({'Dinner': 1, 'Lunch': 0})
    
     # drop duplicates
    tips = tips.drop_duplicates()
    
    tips = tips[tips.tip.notnull()]
                     
    return tips


def train_split(df):
    
    '''
        This function takes in df and splits the data into train, validate and test: Ratio: 56%, 26%, 24% respectively
    '''
    
    train_validate, test = train_test_split(df, test_size= 0.2, random_state = 123)
    
    validate, train = train_test_split(train_validate, train_size = 0.8, random_state = 123)
    
    return train, validate, test



def scale_tips_data(train, validate, test):
    
    '''
        This function takes in train, validate and test df and scales them then \'spits\' scaled df
      
    '''
    # call the scaler 
    scaler = StandardScaler()
    
    # Fit the scaler
    scaler.fit(train)
    
    X_train_scaled = scaler.transform(train)
    X_validate_scaled = scaler.transform(validate)
    X_test_scaled = scaler.transform(test)
    
    return X_train_scaled, X_validate_scaled, X_test_scaled

    
    
    
    
    
