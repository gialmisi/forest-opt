import numpy as np

from .utils import evaluate_with_other
from .boreal import BorealModel
from .process_data import load_objective, nan_to_value


def compute_payoff_table(models):
    """Compute the payoff table using a list of solved models with the same
    type of variables

    """
    po_table = np.zeros((len(models),) * 2)

    for row_i, row_m in enumerate(models):
        for col_i, col_m in enumerate(models):
            po_table[row_i, col_i] = evaluate_with_other(row_m, col_m)

    return po_table


def create_solvers(
    obj_names,
    index=-1,
    management_options=[
        "SA",
        "LRH10",
        "LRH30",
        "LRT10",
        "LRT30",
        "SR5",
        "SRT5",
        "THwoT",
        "THwoTM20",
        "TTN",
        "CCF_2",
        "CCF_4",
        "CCF_2_10",
        "CCF_4_10",
        "CCF_2_20",
        "CCF_4_20",
        "CCF_2_30",
        "CCF_4_30",
        "CCF_2_40",
        "CCF_4_40",
    ],
):
    """Create solvers based on the forest data objectives using the given
    objective names (prefix of the csv file) and mangement refimes (column
    names in the files)

    """
    obj_files = [obj_name + ".csv" for obj_name in obj_names]

    data_raw = [
        load_objective(obj_file, cols=management_options)[:index]
        for obj_file in obj_files
    ]
    data = [nan_to_value(datum_raw) for datum_raw in data_raw]

    # create solvers
    solvers = [BorealModel(datum) for datum in data]

    return solvers
