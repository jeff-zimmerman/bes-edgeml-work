from elm_prediction.analyze import calc_inference
from elm_prediction.options.test_arguments import TestArguments
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import re
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import f1_score
from elm_prediction import package_dir
import matplotlib.pyplot as plt
from elm_prediction.src import utils
#%%

def get_elm_predictions():
    """Returns elm_predictions from elm_prediction.analyze.calc_inference"""

    logger = utils.get_logger(
        script_name=__name__,
        stream_handler=True,
        log_file=None,
    )

    log = False
    suffix = '_log' if log else ''
    args_file = Path(__file__).parent / f'run_dir_classification/args.pkl'
    with args_file.open('rb') as f:
        args = pickle.load(f)
    args = TestArguments().parse(existing_namespace=args)

    device = torch.device('cuda')
    model_cls = utils.create_model_class(args.model_name)
    model = model_cls(args).to(device)

    test_data_file, checkpoint_file, clf_report_dir, plot_dir, roc_dir = \
        utils.create_output_paths(args, infer_mode=True)

    logger.info(f"  Test data file: {test_data_file.as_posix()}")
    with test_data_file.open("rb") as f:
        test_data_dict = pickle.load(f)

    # convert to tuple
    test_data = (
        test_data_dict["signals"],
        test_data_dict["labels"],
        test_data_dict["sample_indices"],
        test_data_dict["window_start"],
        test_data_dict["elm_indices"],
    )

    elm_predictions = calc_inference(args=args,
                                    logger=logger,
                                    model=model,
                                    device=device,
                                    test_data=test_data)

    return elm_predictions

#%%
def id_elms(elm_predictions):

    ids = np.empty((len(elm_predictions), 3))
    for i, elm in enumerate(elm_predictions.values()):
        ch_22 = elm['signals'][:, 2, 5]
        p_elm = ch_22[elm['labels']==0]
        a_elm = ch_22[elm['labels']==1]

        id = [
            np.std(p_elm),
            np.max(ch_22),
            np.min(ch_22)
        ]

        ids[i, :] = id

    # normalize between 0 and 1
    for i in range(ids.shape[-1]):
        if ids[:, i].max() != ids[:, i].min():
            ids[:, i] = (ids[:, i] - ids[:, i].min()) / (ids[:, i].max() - ids[:, i].min())
        else:
            ids[:, i] = 1

    return ids

#%%
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

    return dendrogram

#%%
def make_dendrogram(ids, **kwargs):

    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model = model.fit(ids)

    if not (ax:=kwargs.pop('ax', None)):
        ax = plt.gca()

    ax.set_title("Hierarchical Clustering Dendrogram of ELM Feature Distance")
    # plot the top three levels of the dendrogram
    plot_dendrogram(model, ax=ax, **kwargs)
    ax.set_xlabel("ELM Index")


#%%
if __name__ == '__main__':
    #%%
    with open(Path(__file__).parent/'run_dir_classification/elm_predictions.pkl', 'r+b') as f:
        elm_predictions = pickle.load(f)
    #%%
    ids = id_elms(elm_predictions)
    #%%
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax1 = ax
    ax2 = ax1.twinx()
    fig_fake, ax_fake = plt.subplots(1,1)
    # make dendrogram
    make_dendrogram(ids, ax=ax_fake, distance_sort=True)

    #get f1 score of each elm to plot
    f1 = []
    for elm in elm_predictions.values():
        f1.append(f1_score(elm['labels'], (elm['micro_predictions'] >= 0.4)))
    f1 = np.array(f1)

    #get order of elms in dendrogram
    elm_idxs = [int(re.findall(r'\d+', x.get_text())[0]) for x in ax_fake.get_xticklabels()]
    tick_locations = ax_fake.get_xticks()
    elm_idxs, elm_loc = zip(*sorted(zip(elm_idxs, tick_locations)))

    # add bar graph with f1 scores
    color = 'tab:blue'
    ax2.bar(elm_loc, f1, width=np.diff(tick_locations).min(), color=color, alpha=0.5)
    ax2.tick_params(axis='y', color=color, labelcolor=color)

    # allow context manager for dendrogram
    with plt.rc_context({'lines.linewidth': 2.5}):
        make_dendrogram(ids, ax=ax1)

    # Configure plot
    ax1.grid(False)
    ax2.grid(False)

    ax1.set_ylabel('ELM Feature Distance')
    ax2.set_ylabel('Model F1 Score')

    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    plt.tight_layout()
    fig.savefig(Path(__file__).parent/'run_dir_classification/plots/dendrogram.png')
    plt.close(fig_fake)
    fig.show()
