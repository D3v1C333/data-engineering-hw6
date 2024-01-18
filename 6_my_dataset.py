import json
import os.path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


#Файл скачивать по ссылке https://files.consumerfinance.gov/hmda-historic-loan-data/hmda_2017_nationwide_all-records_labels.zip
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
    get_memory_stat(converted_obj, "./output/6_optimized_memory_stat.json")


def get_optimized_dataset():
    selected_columns = ['loan_amount_000s', 'census_tract_number', 'county_name', 'msamd_name', 'state_name', 'lien_status', 'hoepa_status']

    for column in pd.read_csv("hmda_2017_nationwide_all-records_labels.zip", chunksize=300_000, low_memory=False):
        column['loan_amount_000s'] = column['loan_amount_000s'].astype('float32')
        column['census_tract_number'] = column['census_tract_number'].astype('float32')
        column['county_name'] = column['county_name'].astype('category')
        column['msamd_name'] = column['msamd_name'].astype('category')
        column['state_name'] = column['state_name'].astype('category')
        column['lien_status'] = column['lien_status'].astype('int32')
        column['hoepa_status'] = column['hoepa_status'].astype('int32')


        optimized_data = column[selected_columns]
        optimized_data.to_csv("6.csv", index=False)

def plotting():
    df = read_file("6.csv")
    plt.figure(figsize=(30, 15))

    numeric_columns = df.select_dtypes(include=[np.number])

    plt.subplot(2, 3, 1)
    sns.scatterplot(x='census_tract_number', y='loan_amount_000s', data=df)
    plt.title('Линейный график')

    plt.subplot(2, 3, 2)
    sns.countplot(x='hoepa_status', data=df)
    plt.title('Столбчатый график')

    plt.subplot(2, 3, 3)
    df['lien_status'].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title('Круговая диаграмма')

    plt.subplot(2, 3, 4)
    sns.heatmap(numeric_columns.corr(), annot=True, cmap='coolwarm')
    plt.title('График корреляции')

    plt.subplot(2, 3, 5)
    sns.histplot(x='loan_amount_000s', data=df, bins=10, kde=True)
    plt.title('Гистограмма')

    plt.tight_layout()
    plt.savefig("./output/6.png")


dataset = read_file("hmda_2017_nationwide_all-records_labels.zip")

size = os.path.getsize("hmda_2017_nationwide_all-records_labels.zip") // 1024
print(f"file size = {size} КБ ")

get_memory_stat(dataset, "./output/6_memory_stat.json")
opt_obj(dataset)

get_optimized_dataset()
plotting()
