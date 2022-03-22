from pydataset import data 
import pandas as pd
import matplotlib.pyplot as plt


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


def get_clean_mpg_data():
    '''
        This function acquires and cleans mpg data from pydataset
    '''

    mpg = data('mpg')

    # Drop nulls
    mpg = mpg.dropna()

    # Drop duplicates
    mpg = mpg.drop_duplicates()

    return mpg


def visualize_tips(tips_features, total_bill, tip):
    plt.figure(figsize=(16, 10))

    # Plot regression line
    plt.plot(total_bill, tips_features.yhat_predicted, color='darkseagreen', linewidth=3)
    # add label to the regression line
    plt.annotate('', xy=(87, 93), xytext=(90, 89), xycoords='data',
                 textcoords='data', arrowprops={'arrowstyle': '<-', 'color': 'black'})
    plt.text(87, 93,  r'$\hat{y}=12.5 + .85x$', {'color': 'black',
             'fontsize': 11, 'ha': 'right', 'va': 'center'})
    plt.text(80.5, 95,  'This line!', {
             'color': 'black', 'fontsize': 11, 'ha': 'left', 'va': 'center'})

    # Plot the data points
    plt.scatter(total_bill, tip, color='dimgray')

    # add title
    title_string = 'Where is the line of best fit?'
    plt.title(title_string, fontsize=12, color='black')

    # add axes labels
    plt.ylabel('Tips amount (USD)')
    plt.xlabel('Total Bill (USD)')

    # add baseline
    plt.annotate('', xy=(70, tip.mean()), xytext=(100, tip.mean()), xycoords='data',
                 textcoords='data', arrowprops={'arrowstyle': '-', 'color': 'darkseagreen'})
    plt.text(100, 83,  'or this line.', {
             'color': 'black', 'fontsize': 11, 'ha': 'left', 'va': 'center'})

    # add line connecting min and max of y
    plt.annotate('', xy=(70, tip.min()), xytext=(100, tip.max()), xycoords='data',
                 textcoords='data', arrowprops={'arrowstyle': '-', 'color': 'darkseagreen'})
    plt.text(100.5, 96,  'or this line...', {
             'color': 'black', 'fontsize': 11, 'ha': 'left', 'va': 'center'})

    # add line that is translated up the y-axis a few points from the min/max line
    plt.annotate('', xy=(70, tip.min()+3), xytext=(100, tip.max()+3), xycoords='data',
                 textcoords='data', arrowprops={'arrowstyle': '-', 'color': 'darkseagreen'})
    plt.text(100, 99,  'Not this line...', {
             'color': 'black', 'fontsize': 11, 'ha': 'left', 'va': 'center'})

    return plt.show()


def visual_tip(tips_features, total_bill, tip):
    
    plt.figure(figsize=(16, 10))
    plt.scatter(total_bill, tip, color='dimgray')
    
     # Plot regression line
    plt.plot(total_bill, tips_features.yhat_predicted, color='darkseagreen', linewidth=3)

#     # add the residual line at y=0
#     plt.annotate('', xy=(70, 0), xytext=(100, 0), xycoords='data',
#                  textcoords='data', arrowprops={'arrowstyle': '-', 'color': 'darkseagreen'})

    # set titles
    plt.title(r'Baseline Residuals', fontsize=12, color='black')
    # add axes labels
#     plt.ylabel(r'$\hat{y}-y$')
    plt.ylabel('Tipped Amount (USD)')

    plt.xlabel('Total Bill (USD')

    # add text
    plt.text(85, 15, r'', ha='left', va='center', color='black')

    return plt.show()