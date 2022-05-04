from elm_prediction.analyze import calc_inference
from elm_prediction.options.test_arguments import TestArguments
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import re
from scipy.cluster.hierarchy import dendrogram, fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import f1_score
from elm_prediction import package_dir
import matplotlib.pyplot as plt
from elm_prediction.src import utils


# %%

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


# %%
def id_elms(elm_predictions):
    """
    Function to return identifying features of ELM events.
    :param elm_predictions:
    :return: list[std(pre-ELM), max(ELM), min(ELM), len(active-ELM), gradient(first ELM >= 5V)]
    """
    ids = np.empty((len(elm_predictions), 5, 8))
    for i, elm in enumerate(elm_predictions.values()):
        cs = elm['signals'][:, 2, :]
        p_elm = cs[elm['labels'] == 0]
        a_elm = cs[elm['labels'] == 1]
        first_5 = a_elm[(a_elm >= 5).any(axis=1).argmax()]  # cross-section of BES array first time any is > 5V

        id = [
            np.std(p_elm, axis=0),
            np.max(cs, axis=0),
            np.min(cs, axis=0),
            np.full((8,), len(a_elm)),
            np.gradient(first_5),
        ]

        ids[i, ...] = id

    # normalize between 0 and 1
    for i in range(ids.shape[1]):
        if ids[:, i, :].max() != ids[:, i, :].min():
            ids[:, i, :] = (ids[:, i, :] - ids[:, i, :].min()) / (ids[:, i, :].max() - ids[:, i, :].min())
        else:
            ids[:, i, :] = 1

    return ids


# %%
def make_linkage_matrix(ids, distance_sort: bool = False):

    # define model
    model = AgglomerativeClustering(compute_distances=True)
    model = model.fit(ids.reshape(len(ids), -1))


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

    if distance_sort:
        linkage_matrix = linkage_matrix[linkage_matrix[:, 2].argsort()]

    return linkage_matrix


def plot_dendrogram(linkage_matrix, ax=None, thresh=None, **kwargs):

    if not ax:
        ax = plt.gca()

    ax.set_title("Hierarchical Clustering Dendrogram of ELM Feature Distance")
    dendrogram(linkage_matrix, color_threshold=thresh, ax=ax, **kwargs)
    # get clusters calculated in dendrogram
    ax.set_xlabel("ELM Index")


def plot_groups(elm_predictions, thresh: float = None, distance_sort: bool = False):

    """
    Function to plot dendrogram from elm_prediction nested dict
    :param elm_predictions: nested dict of ELMs from analyze.calc_inference
    :param thresh: (optional) threshold value for dendrogram and clustering algorithm
    :param distance_sort: (optional) sort plotted dendrogram by smallest distance first.
    :return: tuple(np.ndarray) ELM indexes (from analyze.calc_inference) grouped by distance below threshold.
    """

    # make linkage_matrix
    linkage_matrix = make_linkage_matrix(ids)

    # get order of elms in dendrogram
    fig_fake, ax_fake = plt.subplots(1, 1)
    plot_dendrogram(linkage_matrix, ax=ax_fake) #used only for this section
    # elm_idxs is the index of the ELM in the array of all elmms, not the index assigned in calc_inference
    elm_idxs = [int(re.findall(r'\d+', x.get_text())[0]) for x in ax_fake.get_xticklabels()]
    tick_locations = ax_fake.get_xticks()
    bar_locations = list(list(zip(*sorted(zip(elm_idxs, tick_locations))))[-1])
    plt.close(fig_fake)

    ### Make real plot
    fig, ax1 = plt.subplots(1, 1, figsize=(16, 9))
    ax2 = ax1.twinx()

    # get f1 score of each elm to plot
    f1 = []
    for elm in elm_predictions.values():
        f1.append(f1_score(elm['labels'], (elm['micro_predictions'] >= 0.4)))
    # add bar graph with f1 scores
    color = 'tab:blue'
    ax2.bar(bar_locations, f1, width=np.diff(tick_locations).min(), color=color, alpha=0.5)
    ax2.tick_params(axis='y', color=color, labelcolor=color)

    # allow context manager for dendrogram
    with plt.rc_context({'lines.linewidth': 2.5}):
        plot_dendrogram(linkage_matrix, thresh=thresh, ax=ax1, distance_sort=distance_sort)
        if thresh:
            ax1.axhline(thresh, color='tab:gray', ls='--', label='Threshold')

    # Configure plot
    ax1.grid(False)
    ax2.grid(False)

    ax1.set_ylabel('ELM Feature Distance')
    ax2.set_ylabel('Model F1 Score')

    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)
    ax1.set_xticklabels(np.array([*elm_predictions.keys()])[elm_idxs])

    plt.tight_layout()
    fig.savefig(Path(__file__).parent / 'run_dir_classification/plots/dendrogram.png')
    fig.show()

    if thresh:
        clusters = fcluster(linkage_matrix, t=thresh, criterion='distance')
        clusters_unique = np.unique(clusters)
        group_elms = tuple(np.array([*elm_predictions.keys()])[clusters == cluster_id] for cluster_id in clusters_unique)
    else:
        group_elms = None

    return group_elms

# %%
if __name__ == '__main__':
    # %%
    with open(Path(__file__).parent / 'run_dir_classification/elm_predictions.pkl', 'r+b') as f:
        elm_predictions = pickle.load(f)
    # %%
    ids = id_elms(elm_predictions)
    thresh = 2
    groups = plot_groups(elm_predictions, thresh)

