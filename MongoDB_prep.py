import pymongo
import numpy as np
import pandas as pd
from pymongo import MongoClient


##################################
### Connect to MongoDB Client  ###
##################################
def client_connection():
    client = MongoClient('mongodb://localhost:27017/')
    db = client['BDA']
    return db


######################################################################################################
### Using MQL to capture the data - Show all the documents with all the relevant fields except _id ###
######################################################################################################
def get_cusotmer_data(db):
        
    result = db.Customers.find( 
        {
            "customerID": {"$exists": True},	
            "gender" : {"$exists": True},
            "SeniorCitizen": {"$exists": True},
            "Partner": {"$exists": True},
            "Dependents": {"$exists": True},
            "tenure": {"$exists": True},
            "Services": {"$exists": True},
            "Contract": {"$exists": True},
            "PaperlessBilling": {"$exists": True},
            "PaymentMethod": {"$exists": True},
            "MonthlyCharges": {"$exists": True},
            "TotalCharges": {"$exists": True}
        },
        {"_id": 0}  # Exclude `_id` field)   
    )                      
    result.rewind()
    list_cur = list(result)
    return list_cur


#####################################
### Converting Collection to `df` ###
#####################################
def convert_df(data):
    return pd.DataFrame(data) #result
    
#############################
### Clean Data & Save CSV ###
#############################

def clean_and_save(df):
        
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].round(2) # Round to 2 decimal places 
        
    df['MonthlyCharges'] = df['MonthlyCharges'].round(2) # Round to 2 decimal places

    df['tenure'] = df['tenure'].round().astype(float) # Round and change to float

    df['PaperlessBilling'] = df['PaperlessBilling'].str[0] # Extract the Yes / No from the array

    services_df = pd.json_normalize(df['Services']) # Extract Services sub-documents into separate columns

    df = pd.concat([df.drop(columns=['Services']), services_df], axis=1) # Concatenate the normalized columns with the original dataframe

    column_order = [
            "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
            "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", 
            "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", 
            "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", 
            "MonthlyCharges", "TotalCharges"
        ]
    df = df[column_order] 

    df = df.dropna(subset=['TotalCharges']) # Remove Nulls in totalcharges

    df.to_csv('churn_new_customers.csv', index=False)  
    df.isna().sum() 

    return df