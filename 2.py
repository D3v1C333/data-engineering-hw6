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
    get_memory_stat(converted_obj, "./output/2_optimized_memory_stat.json")


def get_optimized_dataset():
    selected_columns = ['askPrice', 'color', 'mileage', 'brandName', 'modelName', 'stockNum', 'dealerID']

    for column in pd.read_csv("[2]automotive.csv.zip", chunksize=100_000, low_memory=False):
        column['askPrice'] = column['askPrice'].astype('int32')
        column['color'] = column['color'].astype('category')
        column['mileage'] = column['mileage'].astype('int32')
        column['brandName'] = column['brandName'].astype('category')
        column['modelName'] = column['modelName'].astype('category')
        column['stockNum'] = column['stockNum'].astype('category')
        column['dealerID'] = column['dealerID'].astype('int32')


        optimized_data = column[selected_columns]
        optimized_data.to_csv("2.csv", index=False)

def plotting():
    df = read_file("2.csv")
    plt.figure(figsize=(30, 15))

    numeric_columns = df.select_dtypes(include=[np.number])

    plt.subplot(2,3,1)
    sns.boxplot(x='dealerID', y='brandName', data=df)
    plt.title('Ящик с усами: Распределение бренда по дилерам')
    plt.xlabel('Дилер')
    plt.ylabel('Бренд')

    plt.subplot(2, 3, 2)
    sns.barplot(x='brandName', y = 'askPrice',data=df)
    plt.title('Столбчатая диаграмма')

    plt.subplot(2, 3, 3)
    df['brandName'].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title('Круговая диаграмма')

    plt.subplot(2, 3, 4)
    sns.heatmap(numeric_columns.corr(), annot=True, cmap='coolwarm')
    plt.title('График корреляции')

    plt.subplot(2, 3, 5)
    sns.histplot(x='mileage', data=df, bins=10, kde=True)
    plt.title('Гистограмма')

    plt.tight_layout()
    plt.savefig("./output/2.png")


dataset = read_file("[2]automotive.csv.zip")

size = os.path.getsize("[2]automotive.csv.zip") // 1024
print(f"file size = {size} КБ ")

get_memory_stat(dataset, "./output/2_memory_stat.json")
opt_obj(dataset)

get_optimized_dataset()
plotting()
