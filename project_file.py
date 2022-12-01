import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def linear_regress(df, meter1, meter2):
    X = df[meter1].to_numpy()
    X = X.reshape(-1, 1)
    y = df[meter2].to_numpy()
    y = y.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
    lr_model = LinearRegression(fit_intercept=True)
    lr_model = lr_model.fit(X_train, y_train)

    r_sqrd = lr_model.score(X, y)
    print(r_sqrd)

    y_pred = lr_model.predict(X_test)

    plt.scatter(X_test, y_test)
    plt.plot(X_test, y_pred, color="blue", linewidth=3)
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("Attrition_dummy.csv")
    print(df.columns)
    linear_regress(df, "JobLevel", "MonthlyIncome")
    linear_regress(df, "TotalWorkingYears", "MonthlyIncome")
    linear_regress(df, "Gender", "MonthlyIncome")
