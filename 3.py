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
    get_memory_stat(converted_obj, "./output/3_optimized_memory_stat.json")


def get_optimized_dataset():
    selected_columns = ['TAIL_NUMBER', 'DEPARTURE_TIME', 'YEAR', 'DAY_OF_WEEK', 'AIRLINE', 'FLIGHT_NUMBER', 'AIR_TIME']

    for column in pd.read_csv("[3]flights.csv", chunksize=100_000, low_memory=False):
        column['TAIL_NUMBER'] = column['TAIL_NUMBER'].astype('category')
        column['DEPARTURE_TIME'] = column['DEPARTURE_TIME'].astype('float32')
        column['YEAR'] = column['YEAR'].astype('int32')
        column['DAY_OF_WEEK'] = column['DAY_OF_WEEK'].astype('int32')
        column['AIRLINE'] = column['AIRLINE'].astype('category')
        column['FLIGHT_NUMBER'] = column['FLIGHT_NUMBER'].astype('int32')
        column['AIR_TIME'] = column['AIR_TIME'].astype('float32')


        optimized_data = column[selected_columns]
        optimized_data.to_csv("3.csv", index=False)

def plotting():
    df = read_file("3.csv")
    plt.figure(figsize=(30, 15))

    numeric_columns = df.select_dtypes(include=[np.number])

    plt.subplot(2, 3, 1)
    sns.lineplot(x='DEPARTURE_TIME', y='AIR_TIME', data=df)
    plt.title('Линейный график')

    plt.subplot(2, 3, 2)
    sns.countplot(x='DAY_OF_WEEK', data=df)
    plt.title('Столбчатый график')

    plt.subplot(2, 3, 3)
    df['AIRLINE'].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title('Круговая диаграмма')

    plt.subplot(2, 3, 4)
    sns.heatmap(numeric_columns.corr(), annot=True, cmap='coolwarm')
    plt.title('График корреляции')

    plt.subplot(2, 3, 5)
    sns.histplot(x='AIR_TIME', data=df, bins=10, kde=True)
    plt.title('Гистограмма')

    plt.tight_layout()
    plt.savefig("./output/3.png")


dataset = read_file("[3]flights.csv")

size = os.path.getsize("[3]flights.csv") // 1024
print(f"file size = {size} КБ ")

get_memory_stat(dataset, "./output/3_memory_stat.json")
opt_obj(dataset)

get_optimized_dataset()
plotting()
