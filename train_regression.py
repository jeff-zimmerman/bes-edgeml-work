from elm_prediction.train_regression import train_loop
from elm_prediction import package_dir
from pathlib import Path

if __name__ == '__main__':
    args = {'model_name': 'multi_features_ds',
            'input_data_file': package_dir / 'labeled_elm_events_long_windows_20220419.hdf5',
            'device': 'cuda',
            'batch_size': 64,
            'n_epochs': 20,
            'max_elms': -1,
            'fraction_test': 0.025,
            'fft_num_filters': 20,
            'dwt_num_filters': 20,
            'signal_window_size': 256,
            'output_dir': Path(__file__).parent / 'run_dir_regression_log_long',
            'regression': 'log',
            'weight_decay': 0.00014719712571721345 # from optuna tuning
            }

    train_loop(args)
