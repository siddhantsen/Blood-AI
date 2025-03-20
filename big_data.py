import pandas as pd
import numpy as np


def create_dataset(original_data, input_list, output_list):
    selected_columns = input_list + output_list
    df = original_data[selected_columns]
    df = df.dropna()
    return df
