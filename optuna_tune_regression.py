import matplotlib.pyplot as plt

from elm_prediction.train_regression import train_loop
from elm_prediction import package_dir
import optuna
from pathlib import Path
from elm_prediction.src.utils import get_logger


def objective(trial):

        args = {'model_name': 'multi_features_ds',
                'input_data_file': package_dir / 'labeled-elm-events.hdf5',
                'device': 'cuda',
                'batch_size': 64,
                'n_epochs': 10,
                'max_elms': 40,
                'fraction_test': 0.025,
                'fft_num_filters': 20,
                'dwt_num_filters': 20,
                'signal_window_size': 256,
                'output_dir': Path(__file__).parent / 'run_dir_log',
                'regression': 'log',
                'dry_run': True,
                'weight_decay': trial.suggest_loguniform('weight_decay', 1e-5, 1e-1)
                }

        outputs, _ = train_loop(args, trial)

        return outputs['r2_scores'][-1]



if __name__ == "__main__":

        run_dir = Path('./run_dir_log/')
        logger = get_logger(script_name=__name__, log_file= run_dir / 'output.log')

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100)

        pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]

        logger.info("Study statistics: ")
        logger.info("  Number of finished trials: ", len(study.trials))
        logger.info("  Number of pruned trials: ", len(pruned_trials))
        logger.info("  Number of complete trials: ", len(complete_trials))

        logger.info("Best trial:")
        trial = study.best_trial
        logger.info("  Value: ", trial.value)

        logger.info("  Params: ")
        for key, value in trial.params.items():
                logger.info("    {}: {}".format(key, value))

        optuna.visualization.matplotlib.plot_intermediate_values(study)
        plt.savefig(run_dir/'plots/optuna_study.png')
        plt.show()
