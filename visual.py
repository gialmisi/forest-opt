from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import numpy as np
import matplotlib.pyplot as plt

pf = np.loadtxt("./data/test_run.dat")
extrema = np.loadtxt("./data/test_run_extrema.dat")
pf = pf[pf != np.zeros(3)].reshape(-1, pf.shape[1])
print(pf)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(extrema[0, 0], extrema[0, 1], extrema[0, 2], label="nadir")
ax.scatter(extrema[1, 0], extrema[1, 1], extrema[1, 2], label="ideal")
ax.scatter(pf[:, 0], pf[:, 1], pf[:, 2])
ax.set_xlabel("Income")
ax.set_ylabel("Carbon")
ax.set_zlabel("CHSI")
plt.legend()
plt.show()

fig, axs = plt.subplots(1, 3)
axs[0].scatter(pf[:, 0], pf[:, 1])
axs[0].scatter(extrema[0, 0], extrema[0, 1], label="nadir")
axs[0].scatter(extrema[1, 0], extrema[1, 1], label="ideal")
axs[0].set_xlabel("Income")
axs[0].set_ylabel("Carbon")
axs[0].set_title("Income and Carbon")
axs[0].legend()

axs[1].scatter(pf[:, 0], pf[:, 2])
axs[1].scatter(extrema[0, 0], extrema[0, 2], label="nadir")
axs[1].scatter(extrema[1, 0], extrema[1, 2], label="ideal")
axs[1].set_xlabel("Income")
axs[1].set_ylabel("CHSI")
axs[1].set_title("Income and CHSI")
axs[1].legend()

axs[2].scatter(pf[:, 1], pf[:, 2])
axs[2].scatter(extrema[0, 1], extrema[0, 2], label="nadir")
axs[2].scatter(extrema[1, 1], extrema[1, 2], label="ideal")
axs[2].set_xlabel("Carbon")
axs[2].set_ylabel("CHSI")
axs[2].set_title("Carbon and CHSI")
axs[2].legend()

plt.show()
