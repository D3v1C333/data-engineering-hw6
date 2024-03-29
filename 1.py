import json
import os.path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


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
    get_memory_stat(converted_obj, "./output/1_optimized_memory_stat.json")


def get_optimized_dataset():
    selected_columns = ['number_of_game', 'length_minutes', 'h_hits', 'v_hits', 'h_errors', 'h_name', 'v_manager_name']

    for column in pd.read_csv("[1]game_logs.csv", chunksize=100_000, low_memory=False):
        column['number_of_game'] = column['number_of_game'].astype('int32')
        column['length_minutes'] = column['length_minutes'].astype('float32')
        column['h_hits'] = column['h_hits'].astype('float32')
        column['v_hits'] = column['v_hits'].astype('float32')
        column['h_errors'] = column['h_errors'].astype('float32')
        column['h_name'] = column['h_name'].astype('category')
        column['v_manager_name'] = column['v_manager_name'].astype('category')

        optimized_data = column[selected_columns]
        optimized_data.to_csv("1.csv", index=False)


def plotting():
    df = read_file("1.csv")
    plt.figure(figsize=(30, 20))

    df_encoded = pd.get_dummies(df, columns=['h_name', 'v_manager_name'])

    plt.subplot(2, 3, 1)
    sns.lineplot(x='number_of_game', y='length_minutes', data=df)
    plt.title('Линейный график')

    plt.subplot(2, 3, 2)
    sns.barplot(x='h_name', y='h_hits', data=df)
    plt.title('Столбчатая диаграмма')

    plt.subplot(2, 3, 3)
    df['number_of_game'].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title('Круговая диаграмма')

    plt.subplot(2, 3, 4)
    sns.heatmap(df_encoded[['number_of_game', 'length_minutes', 'h_hits', 'v_hits', 'h_errors']].corr(), annot=True,
                cmap='coolwarm')
    plt.title('График корреляции')

    plt.subplot(2, 3, 5)
    sns.histplot(x='length_minutes', data=df, bins=10, kde=True)
    plt.title('Гистограмма')

    plt.tight_layout()
    plt.savefig("./output/1.png")


dataset = read_file("[1]game_logs.csv")

size = os.path.getsize("[1]game_logs.csv") // 1024
print(f"file size = {size} КБ ")

get_memory_stat(dataset, "./output/1_memory_stat.json")
opt_obj(dataset) # Оптимизированный файл занимает 79.227 КБ памяти против 132.900 у обычного и 58.486 КБ file in memory size против против 487.556 у неоптимизированного

get_optimized_dataset()
plotting()
