import json
import os.path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def read_file(filename):
    return next(
        pd.read_csv(filename, low_memory=False, chunksize=100_000))


def get_memory_stat(df, filename):
    memory_usage_stat = df.memory_usage(deep=True)
    total_memory_usage = memory_usage_stat.sum()
    print(f"file in memory size ={total_memory_usage // 1024:10} КБ ")
    column_stat = list()
    for key in df.dtypes.keys():
        column_stat.append({
            "column_name": key,
            "memory_abs": int(memory_usage_stat[key] // 1024),
            "memory_per": round(memory_usage_stat[key] / total_memory_usage * 100, 4),
            "dtype": str(df.dtypes[key])
        })
    column_stat.sort(key=lambda x: x['memory_abs'], reverse=True)
    with open(filename, "w") as json_file:
        json.dump(column_stat, json_file, indent=2)

def opt_obj(df):
    converted_obj = pd.DataFrame()
    for col in df.columns:
        num_unique_values = len(df[col].unique())
        num_total_values = len(df[col])
        if num_unique_values / num_total_values < 0.5 and df[col].dtype == 'object':
            converted_obj.loc[:, col] = df[col].astype('category')
        else:
            converted_obj.loc[:, col] = df[col]

    int_columns = df.select_dtypes(include='int').columns
    converted_obj[int_columns] = df[int_columns].apply(pd.to_numeric, downcast='integer')
    float_columns = df.select_dtypes(include='float').columns
    converted_obj[float_columns] = df[float_columns].apply(pd.to_numeric, downcast='float')

    df.to_csv("optimized_file.csv", index=False)
    size = os.path.getsize("optimized_file.csv") // 1024
    print(f"file size = {size} КБ ")
    get_memory_stat(converted_obj, "./output/4_optimized_memory_stat.json")


def get_optimized_dataset():
    selected_columns = ['specializations', 'salary_from', 'salary_to', 'address_lat', 'name', 'employer_name', 'department_name']

    for column in pd.read_csv("[4]vacancies.csv.gz", chunksize=100_000, low_memory=False):
        column['specializations'] = column['specializations'].astype('category')
        column['salary_from'] = column['salary_from'].astype('float32')
        column['salary_to'] = column['salary_to'].astype('float32')
        column['address_lat'] = column['address_lat'].astype('float32')
        column['name'] = column['name'].astype('category')
        column['employer_name'] = column['employer_name'].astype('category')
        column['department_name'] = column['department_name'].astype('category')


        optimized_data = column[selected_columns]
        optimized_data.to_csv("4.csv", index=False)

def plotting():
    df = read_file("4.csv")
    plt.figure(figsize=(30, 15))

    numeric_columns = df.select_dtypes(include=[np.number])

    plt.subplot(2, 3, 1)
    sns.lineplot(x='salary_from', y='salary_to', data=df)
    plt.title('Линейный график')

    plt.subplot(2, 3, 2)
    sns.countplot(x='department_name', data=df)
    plt.title('Столбчатый график')

    plt.subplot(2, 3, 3)
    df['salary_from'].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title('Круговая диаграмма')

    plt.subplot(2, 3, 4)
    sns.heatmap(numeric_columns.corr(), annot=True, cmap='coolwarm')
    plt.title('График корреляции')

    plt.subplot(2, 3, 5)
    sns.histplot(x='salary_from', data=df, bins=2, kde=True)
    plt.title('Гистограмма')

    plt.tight_layout()
    plt.savefig("./output/4.png")


dataset = read_file("[4]vacancies.csv.gz")

size = os.path.getsize("[4]vacancies.csv.gz") // 1024
print(f"file size = {size} КБ ")

get_memory_stat(dataset, "./output/4_memory_stat.json")
opt_obj(dataset)

get_optimized_dataset()
plotting()
