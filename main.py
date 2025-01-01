
import pandas as pd
import Churn_functions as cf
import MongoDB_prep as mdbp

def main():
    
    ### Connect to Client ###
    db = mdbp.client_connection()

    ### Get Customer Data ###
    customer_data = mdbp.get_cusotmer_data(db)

    ### Convert to DataFrame ###
    df = mdbp.convert_df(customer_data)

    ### Clean and Save Dataframe to CSV
    df = mdbp.clean_and_save(df)

    ### Load churn_train
    churn_train_orig = pd.read_csv('churn.csv')
    churn_train_orig.shape

    ### Load churn_new
    churn_new_orig = pd.read_csv('churn_new_customers.csv')
    churn_new_orig.shape

    ### Using prep_churn_train
    churn_train = cf.prep_churn_train(churn_train_orig)

    ### Using prep_churn_new
    churn_new = cf.prep_churn_new(churn_new_orig)

    ### Using split_churn_train
    x_train,y_train,cid_train = cf.split_churn_train(churn_train)

    ### Using split_churn_new
    x_new, cid_new = cf.split_churn_new(churn_new)

    ### Using training_random_forest
    model = cf.training_random_forest(36,11,1,x_train,y_train)

    ### Using prediction_random_forest
    churn_new_orig_with_predict = cf.prediction_random_forest(model,x_new,churn_new_orig)

    ### Using saving_random_forest_results
    cf.saving_random_forest_results(churn_new_orig_with_predict)

    ### Using random_forest_feature_importance
    cf.random_forest_feature_importance(model,x_new)

if __name__ == "__main__":
    main()
