import glob
import json
import os
import re
from typing import Dict, List, Tuple, TypedDict
import jsonlines
import numpy as np
from sklearn.metrics import pairwise_distances

class TestResults(TypedDict):
    test_mr1: float
    test_mAP: float

class Postfix(TypedDict):
    Epoch: int
    train_loss: float
    train_loss_step: float
    train_cls_loss: float
    train_cls_loss_step: float
    train_triplet_loss: float
    train_triplet_loss_step: float
    val_loss: float
    mr1: float
    mAP: float

def reduce_func(D_chunk, start, top_size=100):
    nearest_items = np.argsort(D_chunk, axis=1)[:, :top_size + 1]
    return [(i, items[items!=i]) for i, items in enumerate(nearest_items, start)]

def calculate_ranking_metrics(embeddings: np.ndarray, cliques: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    distances = pairwise_distances(embeddings, metric='cosine')
    s_distances = np.argsort(distances, axis=1)
    cliques = np.array(cliques)
    query_cliques = cliques[s_distances[:, 0]]
    search_cliques = cliques[s_distances[:, 1:]]

    query_cliques = np.tile(query_cliques, (search_cliques.shape[-1], 1)).T
    mask = np.equal(search_cliques, query_cliques)

    ranks = 1.0 / (mask.argmax(axis=1) + 1.0)

    cumsum = np.cumsum(mask, axis=1)
    mask2 = mask * cumsum
    mask2 = mask2 / np.arange(1, mask2.shape[-1] + 1)
    average_precisions = np.sum(mask2, axis=1) / np.sum(mask, axis=1)

    # Вычисляем NDCG
    def dcg(mask: np.ndarray) -> np.ndarray:
        # Discounted Cumulative Gain с функцией дисконтирования 1/sqrt(position)
        positions = np.arange(1, mask.shape[-1] + 1)
        discounts = 1 / np.sqrt(positions)
        return np.sum(mask * discounts, axis=1)

    # Ограничиваем маску до top_k (100)
    top_k=100
    mask_top_k = mask[:, :top_k]

    # DCG для фактического ранжирования на top_k
    dcg_vals = dcg(mask_top_k)

    # Ideal DCG (iDCG) для top_k — максимально возможное DCG для каждого примера
    ideal_mask = np.sort(mask, axis=1)[:, ::-1]  # Идеальное ранжирование
    idcg_vals = dcg(ideal_mask[:, :top_k])
    ndcg_vals = np.where(idcg_vals == 0, 0, dcg_vals / idcg_vals)

    return (ranks, average_precisions, ndcg_vals)


# def dir_checker(output_dir: str) -> str:
#     output_dir = re.sub(r"run-[0-9]+/*", "", output_dir)
#     runs = glob.glob(os.path.join(output_dir, "run-*"))
#     if runs != []:
#         max_run = max(map(lambda x: int(x.split("-")[-1]), runs))
#         run = max_run + 1
#     else:
#         run = 0
#     outdir = os.path.join(output_dir, f"run-{run}")
#     return outdir


# def dir_checker(output_dir: str, experiment_number: int) -> str:
#     # Формируем путь для текущего experiment_number
#     target_dir = os.path.join(output_dir, f"run-{experiment_number}")
#
#     # Проверяем, существует ли уже папка с таким номером
#     if os.path.exists(target_dir):
#         raise FileExistsError(f"Run with number {experiment_number} already exist: {target_dir}")
#
#     # Если не существует, возвращаем путь для нового каталога
#     return target_dir

def dir_checker(output_dir: str) -> None:
    # Проверяем, существует ли уже папка с таким номером
    if os.path.exists(output_dir):
        raise FileExistsError(f"Run already exist: {output_dir}")

# def save_test_predictions(predictions: List, output_dir: str) -> None:
#     os.makedirs(output_dir, exist_ok=True)
#     with open(os.path.join(output_dir, 'submission.txt'), 'w') as foutput:
#         for query_item, query_nearest in predictions:
#             foutput.write('{}\t{}\n'.format(query_item, '\t'.join(map(str,query_nearest))))

def save_test_predictions(predictions: List, output_dir: str, experiment_number: int) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f'submission_{experiment_number}.txt'), 'w') as foutput:
        for query_item, query_nearest in predictions:
            foutput.write('{}\t{}\n'.format(query_item, '\t'.join(map(str,query_nearest))))


def save_predictions(outputs: Dict[str, np.ndarray], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for key in outputs:
        if "_ids" in key:
            with jsonlines.open(os.path.join(output_dir, f"{key}.jsonl"), "w") as f:
                if len(outputs[key][0]) == 4:
                    for clique, anchor, pos, neg in outputs[key]:
                        f.write({"clique_id": clique, "anchor_id": anchor, "positive_id": pos, "negative_id": neg})
                else:
                    for clique, anchor in outputs[key]:
                        f.write({"clique_id": clique, "anchor_id": anchor})
        else:
            np.save(os.path.join(output_dir, f"{key}.npy"), outputs[key])


def save_logs(outputs: dict, output_dir: str, name: str = "log") -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"{name}.jsonl")
    with jsonlines.open(log_file, "a") as f:
        f.write(outputs)


def save_best_log(outputs: Postfix, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "best-log.json")
    with open(log_file, "w") as f:
        json.dump(outputs, f, indent=2)