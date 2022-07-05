import os
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVR
import pandas as pd
import numpy as np
##
# from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score
# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))
# SVM Classifier
def model_svm(df,X_train,y_train,X_test,y_test,split,scaler):
    # try:
    #     os.remove('./results/problem_5_SVM.txt')
    # except OSError:
    #     pass
    # kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    # arr_kernels=[]
    # for kernel in kernels:
    #     clf = make_pipeline(StandardScaler(), SVR(kernel=kernel, gamma='scale'))
    #     clf.fit(X_train, y_train)
    #     accuracy = clf.score(X_test, y_test)
    #     #print(f"The SVM classification accuracy with {kernel} Kernel is: {accuracy * 100:0.2f}%")
    #     arr_kernels.append(f"{accuracy * 100:0.3f}%")
    # df_kernels= pd.DataFrame(arr_kernels,columns=['linear','poly','rbf','sigmoid'])
    # return df_kernels
    ####################
    #svr_poly = SVR(kernel='poly', C=1e3,degree = 2)
    #svr_poly.fit(X_train,y_train)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_rbf.fit(X_train,y_train)
    #svr_sigmoid = SVR(kernel='sigmoid', C=1e3, gamma=0.1)
    #svr_sigmoid.fit(X_train,y_train)
    ###
    y_test_scaled_svm = scaler.inverse_transform(y_test.reshape(-1, 1))
    svm_prediction = svr_rbf.predict(X_test)
    
    y_predictions_svm = scaler.inverse_transform(svm_prediction.reshape(-1, 1))
    
    ###
    df_svm = pd.DataFrame(df)
    df_svm = df_svm[split:-15]
    df_svm['y_test'] = y_test_scaled_svm
    df_svm['Predict_svm'] = y_predictions_svm
    #
    svm_confidence = svr_rbf.score(y_test, svm_prediction)
    #acc_score = accuracy_score(y_test_scaled_svm, y_predictions_svm)
    ###
    #########
    ###
    ### forecast
    forecast_svm = np.array(df)[-15:]
    forecast_svm = scaler.fit_transform(forecast_svm.reshape(-1,1))
###
    svm_forecast= svr_rbf.predict(forecast_svm)
    svm_forecast = scaler.inverse_transform(svm_forecast.reshape(-1,1))
    #svm_forecast = abs(svm_forecast)
    df_forecast_svm = pd.DataFrame(df[-15:])
    df_forecast_svm['Fore_15_next_svm'] = svm_forecast
    #df_forecast_svm = pd.DataFrame(svm_forecast,columns=['Fore_15_next_svm'])


    return df_svm,df_forecast_svm,svm_confidence