import matplotlib.pyplot as pyplot


def eval_metric_plot(eval_results, eval_metric: str):
    for key in eval_results.keys():
        epochs = len(eval_results[str(key)][eval_metric])
        x_axis = range(0, epochs)
        # plot log loss
        fig, ax = pyplot.subplots()
        ax.plot(x_axis, eval_results[str(key)][eval_metric], label=str(key))
        ax.legend()
        pyplot.ylabel(eval_metric)
        pyplot.title("model " + eval_metric + " plot")
    pyplot.show()
    return
