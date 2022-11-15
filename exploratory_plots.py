import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def scatter_plot(df, col1, col2):
    ax = sns.scatterplot(x=col1, y=col2, hue=df["Attrition"], data=df, palette=["Blue", "Orange"])
    legend_labels, _ = ax.get_legend_handles_labels()
    ax.legend(legend_labels, ["Yes", 'No'], title="Attrition")
    ax.set_title(str(col1) + " v.s " + str(col2))

    plt.savefig("Scatter plot " + col1 + " v.s " + col2 + ".png")

    return plt


def main():
    df = pd.read_csv("Attrition_dummy.csv")
    # a = scatter_plot(df, "Age", "JobLevel")
    # b = scatter_plot(df, "Age", "TotalWorkingYears")
    # c = scatter_plot(df, "JobLevel", "MonthlyIncome")
    # d = scatter_plot(df, "JobLevel", "TotalWorkingYears")
    # e = scatter_plot(df, "JobLevel", "YearsAtCompany")
    # f = scatter_plot(df, "MaritalStatus", "StockOptionLevel")
    # g = scatter_plot(df, "MonthlyIncome", "TotalWorkingYears")
    # h = scatter_plot(df, "MonthlyIncome", "YearsAtCompany")
    # i = scatter_plot(df, "PercentSalaryHike", "PerformanceRating")
    # j = scatter_plot(df, "TotalWorkingYears", "YearsAtCompany")
    # k = scatter_plot(df, "YearsAtCompany", "YearsInCurrentRole")
    # l = scatter_plot(df, "YearsAtCompany", "YearsSinceLastPromotion")
    # m = scatter_plot(df, "YearsAtCompany", "YearsWithCurrManager")
    # n = scatter_plot(df, "YearsInCurrentRole", "YearsSinceLastPromotion")
    # o = scatter_plot(df, "YearsInCurrentRole", "YearsWithCurrManager")
    # p = scatter_plot(df, "YearsSinceLastPromotion", "YearsWithCurrManager")


if __name__ == "__main__":
    main()
