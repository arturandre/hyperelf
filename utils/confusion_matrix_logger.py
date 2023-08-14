import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot


def create_and_save_conf_matrix_plot(filename, true, pred):
    labels_order = np.unique(true)[::-1]
    conf_matrix = confusion_matrix(true, pred, labels=labels_order)
    cm_display = ConfusionMatrixDisplay(conf_matrix.T, display_labels=labels_order)

    import matplotlib.pyplot
    matplotlib.pyplot.switch_backend('agg')
    _ = cm_display.plot(cmap="OrRd")

    ax = cm_display.ax_

    ax.set_xlabel("True label")
    ax.set_ylabel("Predicted label")

    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    matplotlib.pyplot.savefig(filename)

def save_conf_matrix_plot(filename, conf_matrix, labels_order):
    cm_display = ConfusionMatrixDisplay(conf_matrix.T, display_labels=labels_order)
    import matplotlib.pyplot
    matplotlib.pyplot.switch_backend('agg')
    _ = cm_display.plot(cmap="OrRd")

    ax = cm_display.ax_

    ax.set_xlabel("True label")
    ax.set_ylabel("Predicted label")

    ax.xaxis.set_label_position('top')

    matplotlib.pyplot.savefig(filename)