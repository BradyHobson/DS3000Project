import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
import seaborn as sns


def linear_regress(df, meter1, meter2):
    """ Performs Linear Regression Algorithm on data set """
    # Split x and y into numpy arrays
    X = df[meter1].to_numpy()
    X = X.reshape(-1, 1)
    y = df[meter2].to_numpy()
    y = y.reshape(-1, 1)

    # Split x and y into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
    lr_model = LinearRegression(fit_intercept=True)
    lr_model = lr_model.fit(X_train, y_train)

    # Find r squared value
    r_sqrd = lr_model.score(X, y)
    print("R Squared Value: ", r_sqrd)

    # Predict y values based on model
    y_pred = lr_model.predict(X_test)

    # Plot a scatter plot
    plt.scatter(X_test, y_test)
    plt.plot(X_test, y_pred, color="blue", linewidth=3)
    plt.show()


def find_best_k(X_train, X_test, y_train, y_test):
    # build the k-nn model, experiment with different values of k and plot the results
    accuracy = []
    for i in range(1, 20):
        knn = KNeighborsClassifier(n_neighbors=i).fit(X_train, y_train)
        prediction = knn.predict(X_test)
        accuracy.append(accuracy_score(y_test, prediction))

    return accuracy.index(max(accuracy))


def knn(df, meter1, meter2):
    X = df[meter1]
    y = df[meter2]

    # Create test and train set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

    # Find best K value to use
    best_k = find_best_k(X_train, X_test, y_train, y_test)

    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train, y_train)
    prediction = knn.predict(X_test)
    report = classification_report(y_test, prediction)
    return report, prediction

def classification(df, meter1, meter2):
    X = df[meter1]
    y = df[meter2]
    
    #test and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7, stratify=y)
    
    #copy over
    X_train = X_train.copy()
    X_test  = X_test.copy()
    
    #normalize the numeric features
    scaler = StandardScaler()

    #train
    scaler.fit(X_train[meter1])
    X_train[meter1] = scaler.transform(X_train[meter1]) 
    
    #initialize the classifier and fit
    svm = SVC()
    svm.fit(X_train, y_train) 
    
    #encode & scale the new/test data
    X_test[meter1] = scaler.transform(X_test[meter1])

    #predict the labels for the test set
    prediction = svm.predict(X_test)
    
    report = classification_report(y_test, prediction)
    
    return report, prediciton


if __name__ == "__main__":
    df = pd.read_csv("Attrition_dummy.csv")
    print(df.columns)
    linear_regress(df, "JobLevel", "MonthlyIncome")
    linear_regress(df, "TotalWorkingYears", "MonthlyIncome")
    linear_regress(df, "Gender", "MonthlyIncome")

    report, prediction = knn(df, ["Age", "Gender", "JobLevel", "MonthlyIncome", "YearsSinceLastPromotion",
                                  "StockOptionLevel"], "Attrition")
    print("KNN predictions: ", prediction)
    print("KNN Report:", report)
    
