import json
import numpy
from matplotlib import pyplot as plt


fig, axs = plt.subplots(nrows=1, ncols=1, squeeze=False, tight_layout=True)
axs[0][0].set_xlabel("TICA 0th component")
axs[0][0].set_ylabel("TICA 1st component")


with open("points.json", "rb") as f:
    data = numpy.array(json.loads(f.read()))
    print(data.shape)
    axs[0][0].scatter(data[:, 0], data[:, 1], s=2, c="red")
            
    fig.savefig("replot.png", format='png')
    plt.close()
