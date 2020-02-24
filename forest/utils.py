import numpy as np


def evaluate_with_other(model1, model2):
    """Evaluates model1's objective value with the variable values set in
    model2. The models are expected to have the attrbutes n, m and OBJ defined,
    where n is the number of rows in the data, m the number of columns and OBJ
    is the objective. A discrete model with 2D data is expected in other words.

    """
    if model1.n.value != model2.n.value and model1.m.value != model2.m.value:
        print("Models' data dimensions dont't match.")
        return None

    n = model1.n.value
    m = model1.m.value

    tmp = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            # store the first model's original value
            tmp[i, j] = model1.x[i, j].value
            # set the first model's variable value to be equal to the variable
            # value in the second model
            model1.x[i, j].value = model2.x[i, j].value

    res = model1.OBJ.expr()

    # reset the variable values in the first model to the original ones
    for i in range(n):
        for j in range(m):
            model1.x[i, j] = tmp[i, j]

    # return the objective value of the frist model evlauated with the variable
    # values of the second model
    return res
