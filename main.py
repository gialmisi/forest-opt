from forest.optimization import compute_payoff_table, create_solvers
from forest.process_data import load_objective, nan_to_value
from forest.utils import evaluate_with_other
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt


from forest.asf import ASF

from pyomo.opt import SolverFactory

import numpy as np


def main():
    objective_names = ["INCOME", "CARBON", "CHSI"]
    management_options = [
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
    ]

    index = None
    basic_solvers = create_solvers(
        objective_names, management_options=management_options, index=index
    )

    # compute the payoff table
    list(map(lambda x: x.solve_model(tee=False), basic_solvers))
    po_table = compute_payoff_table([s.model for s in basic_solvers])

    raw_data = [
        load_objective(f_name + ".csv", cols=management_options)
        for f_name in objective_names
    ]
    data = [nan_to_value(raw_datum) for raw_datum in raw_data]


    # po_table = np.loadtxt("./data/payoff.dat", delimiter=",")
    ideal = np.diag(po_table)
    nadir = np.min(po_table, axis=1)

    n_rows = 12
    n_objs = 3
    refs = np.zeros((n_rows ** n_objs, n_objs))

    m = 0
    for i_val in np.linspace(nadir[0], ideal[0], n_rows):
        for j_val in np.linspace(nadir[1], ideal[0], n_rows):
            for k_val in np.linspace(nadir[2], ideal[2], n_rows):
                refs[m] = [i_val, j_val, k_val]
                m += 1

    x = np.dstack(list((datum.values[:index] for datum in data)))

    results = np.zeros(refs.shape)
    print("Nadir", nadir)
    print("Ideal", ideal)
    print("Payoff table\n", po_table)
    np.savetxt("./data/payoff.dat", po_table.T, header="Payoff table with cols INCOME, CARBON and AHSI")

    np.savetxt(
        "./data/test_run_extrema.dat",
        np.vstack((nadir, ideal)),
        header=f"Nadir and ideal ponts for objectives INCOME, CARBON, AHSI and {len(refs)} rows",
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(refs[:, 0], refs[:, 1], refs[:, 2])
    ax.set_xlabel("Income")
    ax.set_ylabel("Carbon")
    ax.set_zlabel("CHSI")
    plt.show()
    return

    for ref_i, ref in enumerate(refs):
        print(f"Using ref vector {ref_i+1}/{len(refs)}: {ref}")
        asf = ASF(ideal, nadir, ref, x, sense="maximize", scalarization="STOM")

        opt = SolverFactory("cbc")
        opt.options["threads"] = 4
        opt.solve(asf.model)

        res = np.array(
            [
                evaluate_with_other(basic_solvers[0].model, asf.model),
                evaluate_with_other(basic_solvers[1].model, asf.model),
                evaluate_with_other(basic_solvers[2].model, asf.model),
            ]
        )

        is_between = np.all(np.logical_and(res >= nadir, res <= ideal))
        print(
            f"Done. Result: {res}\nResulting point is feasible: {is_between}"
        )
        results[ref_i] = res

        np.savetxt(
            "./data/test_run.dat",
            results,
            header=f"Test run with objectives INCOME, CARBON, CHSI and {len(refs)} rows",
        )

    # solvers = create_solvers(
    #     objective_names, management_options=management_options
    # )
    # # solve
    # list(map(lambda x: x.solve_model(tee=False), solvers))

    # print(solvers[0].model.OBJ.expr(), solvers[0]._solved)
    # print(solvers[1].model.OBJ.expr(), solvers[1]._solved)
    # print(solvers[2].model.OBJ.expr(), solvers[2]._solved)

    # models = [solver.model for solver in solvers]
    # po_table = compute_payoff_table(models)

    # np.savetxt(
    #     "./data/payoff.dat",
    #     po_table,
    #     delimiter=",",
    #     header="Pay of table for objectives INCOME, CARBON and CHSI",
    # )


if __name__ == "__main__":
    main()
