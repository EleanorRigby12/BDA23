
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier # Importing the algorithm
from sklearn.metrics import accuracy_score # importing "accuracy_score" from "sklearn.metrics"

############################
## Creating the FUNCTIONS ##
############################


###################################
### Function - prep_churn_train ###
###################################
def prep_churn_train (df):

    df = df.rename(columns=str.lower) # Rename columns to lower letters
    df.churn = (df.churn=='Yes').astype('int') # Label to numeric
    c_id = df['customerid']
    df = df.drop('customerid', axis=1) # Drop customerid before get_dummies

    df.totalcharges = pd.to_numeric(df.totalcharges,errors='coerce')
    df.isna().sum()
    df[np.isnan(df['totalcharges'])]

    #saving indices of empty rows for inspection 
    check_nulls = df.index[df['totalcharges'].isnull()].tolist()
    
    # replacing null with monthlycharges
    df.totalcharges = df.totalcharges.fillna(df.monthlycharges)
    
    
    df = df.astype({"totalcharges": 'float'})
    df = pd.get_dummies(df)
    df = pd.get_dummies(df, drop_first=True) 
    
    # change to numeric values. 
    for i,v in enumerate(df.dtypes):
        if v == 'bool':
           df[df.columns[i]] = df[df.columns[i]].astype(int)
    
    df = df.astype(float) # Convert all data to float because some modules warn against other types
    df['customerid'] = c_id # Restore CustomerID data

   
    print('')
    
    # Check for nulls
    null = df.isna().sum().sum()
    print (f'There are {null} null values')
    
    # Check for DataType
    dtype = df.dtypes.unique()[0]
    print (f'There are {dtype} dtype')
    
    print('')
    

    return df


##################################
### Function - prep_churn_new  ###
##################################
def prep_churn_new (df):
    
    df = df.rename(columns=str.lower) # Rename columns to lower letters

    c_id = df['customerid']
    df = df.drop('customerid', axis=1) # Drop customerid before get_dummies
  
    df = df.astype({"totalcharges": 'float'})
    
    df = pd.get_dummies(df)
    df = pd.get_dummies(df, drop_first=True) 
    
    # change to numeric values. 
    for i,v in enumerate(df.dtypes):
        if v == 'bool':
           df[df.columns[i]] = df[df.columns[i]].astype(int)
    
    df = df.astype(float) # Convert all data to float because some modules warn against other types
    df['customerid'] = c_id # Restore CustomerID data
   
    print('')
    
    # Check for nulls
    null = df.isna().sum().sum()
    print (f'There are {null} null values')
    
    # Check for DataType
    dtype = df.dtypes.unique()[0]
    print (f'There are {dtype} dtype')
    
    
    print('')

    return df


####################################
### Function - split_churn_train ###
####################################
def split_churn_train(df):

    label = 'churn'
    cid = 'customerid'

    x_train = df.drop(label, axis=1)
    x_train = x_train.drop(cid, axis=1)
    y_train = df[label]
    cid_train = df[cid]
    
    return x_train,y_train,cid_train


##################################
### Function - split_churn_new ###
##################################
def split_churn_new(df):
    cid = 'customerid'

    x_new = df.drop(cid, axis=1)
    cid_new = df[cid]
    
    return x_new, cid_new


#########################################
### Function - training_random_forest ###
#########################################
def training_random_forest(n,m,r,x_train,y_train):

    model = RandomForestClassifier(n_estimators=n, max_depth=m, random_state=r)
    model.fit(x_train, y_train)
    
    return model


###########################################
### Function - prediction_random_forest ###
###########################################
def prediction_random_forest(model,x_new,df_orig):

    y_new = model.predict(x_new) 
    y_new = pd.Series(y_new,name='predict')
    output = df_orig.join(y_new)
    
    return output


###################################################
### Function - random_forest_feature_importance ###
###################################################
def random_forest_feature_importance(model,x_new):

    feature_importances = model.feature_importances_ # applying the method "feature_importances_" on the algorithm
    features = x_new.columns # all the features
    stats = pd.DataFrame({'feature':features, 'importance':feature_importances}) # creating the data frame
    print(stats.sort_values('importance', ascending=False)) # Sorting the data frame
    
    fig, ax = plt.subplots(figsize=(14, 10))
    stats_sort = stats.sort_values('importance', ascending=True)
    stats_sort.plot(y='importance', x='feature', kind='barh', ax=ax)
    for index, value in enumerate(stats_sort['importance']):
        ax.text(value, index, f'{value:.4f}', va='center', ha='left', fontsize=10, color='black')
    
    plt.title('Feature Importance of Random Forest')
    plt.tight_layout()
    plt.show()


###############################################
### Function - saving_random_forest_results ###
###############################################
def saving_random_forest_results(df):
    df.to_csv('new customers churn prediction.csv', index=False)

    print("DataFrame reordered and saved to 'new customers churn prediction.csv'")

