import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

test = norm.rvs(size=1000)


def test_cum(data):
    x = np.sort(data)
    n = len(data)
    y = np.arange(n)/n
    # plt.plot(x, y)
    plt.boxplot([data, 2*data], labels=["tt2", "steam"])
    plt.show()


test_cum(test)


def plot_cumultive(data: List[List[float]]):
    """Create a cumulitive plot. Each dataset inside the data list results in an 
    individual line in the graph

    Args:
        data (List[List[float]]): List of floating points representing the error
    """
    n_list = list()
    x_list = list()
    y_list = list()
    plot_list = list()
    for total in data:
        n = len(total)
        x = np.sort(total)
        y = np.arange(n)/n
        n_list.append(n)
        x_list.append(x)
        y_list.append(y)
        plot_list.append((x, y))

    acc = round(np.mean(total), 2)
    # std = round(np.std(total), 2)
    # minVal = round(min(total), 2)
    # maxVal = round(max(total), 2)
    # plotting
    plt.figure(dpi=200)
    # popt, pcov = curve_fit(func, x, y)
    plt.xlabel('Fehler [mm]', fontsize=15)
    plt.ylabel('Kumulative Häufigkeit', fontsize=15)

    # Min: {minVal:n}mm Max: {maxVal:n}mm
    # plt.title('Statische Genauigkeit: :\n'+f'{acc:n}mm\u00B1{std:n}mm', fontsize=15)
    plt.title('Wiederholbarkeit', fontsize=15)
    for plotty in plot_list:
        plt.scatter(x=plotty[0], y=plotty[1], marker='o')
    # plt.scatter(relevant_data, y=highlighted_y)
    # plt.plot(x, func(x, *popt), 'r-', label="Fitted Curve")
    plt.grid()
    # ticks
    ticky = list()
    for ti in chunked(x, 13):
        ticky.append(round(np.mean(ti), 0))
    # ticky = [20, 50, 80]
    # plt.xticks(np.linspace(start=min(x), stop=max(x), num=20, dtype=int))
    # accuracy line
    for acc in data:
        plt.vlines(np.mean(acc), ymin=0, ymax=2, colors="r")
    ticky.append(acc)
    # plt.ticklabel_format(useLocale=True)
    # add stuff
    # plt.xticks(ticky)
    plt.ylim(ymin=0, ymax=1.05)
    plt.xlim(xmin=0)
    plt.show()
