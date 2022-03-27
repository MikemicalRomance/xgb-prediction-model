import matplotlib.pyplot as pyplot
def eval_metric_plot(eval_results,eval_metric:str):
    for key in eval_results.keys():
        epochs = len(eval_results[str(key)][eval_metric])
        x_axis = range(0, epochs)
        # plot log loss
        fig, ax = pyplot.subplots()
        ax.plot(x_axis, eval_results[str(key)][eval_metric], label=str(key))
        # ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
        ax.legend()
        pyplot.ylabel(eval_metric)
        pyplot.title("model " + eval_metric + " plot")
    pyplot.show()
    return 

## optional plots
#results = xgb_cl.evals_result()
#epochs = len(results["validation_0"]["logloss"])
#x_axis = range(0, epochs)
# plot log loss
#fig, ax = pyplot.subplots()
#ax.plot(x_axis, results["validation_0"]["logloss"], label="validation set")
# ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
#ax.legend()
#pyplot.ylabel("Log Loss")
#pyplot.title("XGBoost Log Loss")
#pyplot.show()
# plot classification error
##fig, ax = pyplot.subplots()
#ax.plot(x_axis, results["validation_0"]["logloss"], label="validation set")
# ax.plot(x_axis, results['validation_1']['error'], label='Test')
#ax.legend()
#pyplot.ylabel("Classification Error")
#pyplot.title("XGBoost Classification Error")
#pyplot.show()