import pandas as pd
import numpy as np


def load_objective(
        objective_file, data_dir="/home/kilo/workspace/forest-opt/data/forest_data/", cols=None,
):
    """Loads the table defined in objective_file and returns a pandas frame.

    """
    # name of the file with the relevant data
    full_path = data_dir + objective_file

    df = pd.read_csv(full_path, delimiter="\t")

    if cols is not None:
        return df[cols]
    else:
        return df


def nan_to_value(frame, value=0.0):
    """Replaces nans in a dataframe with value.

    """
    return frame.fillna(value)


if __name__ == "__main__":
    income_file = "INCOME.csv"
    carbon_file = "CARBON.csv"
    
    
    income_raw = load_objective(income_file)
    income = nan_to_value(income_raw)

    carbon_raw = load_objective(carbon_file)
    carbon = nan_to_value(carbon_raw)

