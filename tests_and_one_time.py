from utils import parse_file
import matplotlib.pyplot as plt

plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import json

if __name__ == '__main__':
    with open('resources/statistics_result_NLTK.json') as f:
        json_data = json.load(f)
    objects = list(json_data.keys())
    y_pos = np.arange(len(objects))
    performance = list(json_data.values())

    fig, ax = plt.subplots()
    width = 0.75  # the width of the bars
    ind = np.arange(len(performance))  # the x locations for the groups
    ax.barh(ind, performance, width, color="magenta")
    ax.set_yticks(ind + width / 2)
    ax.set_yticklabels(objects, minor=False, color='indigo', fontweight='bold', fontsize=8)
    plt.title('Tokens and Entities', fontweight='bold', color='indigo')
    plt.xlabel('Number of corresponding element', fontweight='bold', color='indigo')
    for i, v in enumerate(performance):
        ax.text(v + 3, i + .25, str(v), color='indigo', fontweight='bold')
    plt.savefig('resources/entities.png', dpi=300, format='png', bbox_inches='tight')
    plt.show()
