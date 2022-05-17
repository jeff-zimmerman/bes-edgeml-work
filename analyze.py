from elm_prediction import analyze
from elm_prediction.options.test_arguments import TestArguments
from pathlib import Path
import pickle
from elm_prediction import package_dir
import matplotlib.pyplot as plt


if __name__ == '__main__':

    # plt.close('all')
    log = False
    suffix = '_log' if log else ''
    args_file = Path(__file__).parent / f'run_dir_classification_clustered/args.pkl'
    with args_file.open('rb') as f:
        args = pickle.load(f)

    analyze.do_analysis(args_file=args_file, interactive=True, click_through_pages=True, save=True)