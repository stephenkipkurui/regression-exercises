from sklearn.model_selection import train_test_split
import sklearn.preprocessing
from sklearn.preprocessing import StandardScaler
from pydataset import data
from sklearn.feature_selection import SelectKBest, f_regression, RFE, SequentialFeatureSelector
from sklearn.linear_model import LinearRegression



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

def get_swiss_data():
    '''
        This function call the pydataset and return the swiss data
    '''
    swiss = data('swiss')
    
     # drop missing values
    swiss = swiss.dropna()
    
     # drop duplicates
    swiss = swiss.drop_duplicates()
    
    swiss = swiss[swiss.Fertility.notnull()]
                     
    return swiss


def train_split(df):
    
    '''
        This function takes in df and splits the data into train, validate and test: Ratio: 56%, 26%, 24% respectively
    '''
    # Split the data into train, validate and test
    train_validate, test = train_test_split(df, test_size = 0.2, random_state = 1349)
    train, validate = train_test_split(train_validate, train_size = 0.7, random_state = 1349)
    
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


def select_kbest(predictors, target, num_features):
    
    '''
        This function takes in predictors, and the target variables and the number of 
        features desired and returns the names of the top k selected features based on the SelectKBest class. 
    '''
    num_features = int(input('Enter count of SelectKBest features to return: '))
    
    kbest = SelectKBest(f_regression, k = num_features)
    
    kbest.fit(predictors, target)
    
    return predictors.columns[kbest.get_support()]


def rfe(predictors, target, num_features):
    '''
        This function takes in predictors, and the target variables and the number of 
        features desired and returns the names of the top Recussion Feature Elimination(RFE) features 
        based on the SelectKBest class. 
    '''
    model = LinearRegression()
    
    num_features = int(input('Enter count of RFE features to return: '))
    
    rfe = RFE(model, n_features_to_select = num_features)
    
    rfe.fit(predictors, target)
    
    result = rfe.get_support()
    
    return predictors.columns[result]

    
    
    
    
    
    
    
