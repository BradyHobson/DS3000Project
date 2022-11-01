import pandas as pd


def save_to_txt(col_name, dctn, filename):
    infile = open(filename, 'a')
    infile.write(str(col_name))
    infile.write(str(dctn))
    infile.write("\n")
    infile.close()


def replace_var(df, col_name):
    col_lst = list(df[col_name])

    item_lst = []
    num = 0
    dummy_lst = []

    for i in range(len(col_lst)):
        while True:
            if col_lst[i] not in item_lst:
                item_lst.append(col_lst[i])
                num += 1
                col_lst[i] = num
                dummy_lst.append(col_lst[i])

            break

    return item_lst, dummy_lst


def mk_save_dummy(df, col_lst):
    for col in col_lst:
        item_lst, dummy_lst = replace_var(df, col)
        var_dict = dict(zip(item_lst, dummy_lst))

        df = df.replace({col: var_dict})
        save_to_txt(col, var_dict, "directory.txt")

    return df


def main():
    attrition_df = pd.read_csv("HR-Employee-Attrition.csv")

    col_lst = ["Attrition", "BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus",
               "Over18", "OverTime"]
    attrition_df = mk_save_dummy(attrition_df, col_lst)

    attrition_df.to_csv("Attrition_dummy.csv", index=False)


if __name__ == "__main__":
    main()
