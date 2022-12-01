import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def scatter_plot(df, col1, col2):
    ax = sns.scatterplot(x=col1, y=col2, hue=df["Attrition"], data=df, palette=["Blue", "Orange"])
    legend_labels, _ = ax.get_legend_handles_labels()
    ax.legend(legend_labels, ["Yes", 'No'], title="Attrition")
    ax.set_title(str(col1) + " v.s " + str(col2))
    plt.show()
    return plt


def heat_map(df):
    sns.heatmap(df.corr())
    plt.xticks(size=8)
    plt.yticks(size=8)
    plt.show()
    return plt


def main():
    df = pd.read_csv("Attrition_dummy.csv")
    heat_map(df)
    scatter_plot(df, "Age", "JobLevel")
    scatter_plot(df, "Age", "TotalWorkingYears")
    scatter_plot(df, "JobLevel", "MonthlyIncome")
    scatter_plot(df, "JobLevel", "TotalWorkingYears")
    scatter_plot(df, "JobLevel", "YearsAtCompany")
    scatter_plot(df, "MaritalStatus", "StockOptionLevel")
    scatter_plot(df, "MonthlyIncome", "TotalWorkingYears")
    scatter_plot(df, "MonthlyIncome", "YearsAtCompany")
    scatter_plot(df, "PercentSalaryHike", "PerformanceRating")
    scatter_plot(df, "TotalWorkingYears", "YearsAtCompany")
    scatter_plot(df, "YearsAtCompany", "YearsInCurrentRole")
    scatter_plot(df, "YearsAtCompany", "YearsSinceLastPromotion")
    scatter_plot(df, "YearsAtCompany", "YearsWithCurrManager")
    scatter_plot(df, "YearsInCurrentRole", "YearsSinceLastPromotion")
    scatter_plot(df, "YearsInCurrentRole", "YearsWithCurrManager")
    scatter_plot(df, "YearsSinceLastPromotion", "YearsWithCurrManager")


if __name__ == "__main__":
    main()
