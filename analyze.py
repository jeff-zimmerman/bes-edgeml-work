from elm_prediction.analyze import Analysis
from pathlib import Path
import pickle


if __name__ == '__main__':

    run = Analysis('run_dir_regression_log_clustered')
    run.plot_all()
    run.show()