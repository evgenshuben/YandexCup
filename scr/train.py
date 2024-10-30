import os
from copy import deepcopy
import logging
import torch
import mlflow
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from tqdm import tqdm, trange
from typing import Dict, List
import numpy as np
from sklearn.metrics import pairwise_distances_chunked

from scr.early_stopper import EarlyStopper
from scr.utils import (
    TestResults,
    Postfix,
    calculate_ranking_metrics,
    dir_checker,
    reduce_func,
    save_best_log,
    save_logs,
    save_predictions,
    save_test_predictions
)

logger: logging.Logger = logging.getLogger()  # The logger used to log output

class Trainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device: str = cfg.enviroment.device
        self.postfix: Postfix = {}
        self.best_model_path: str = None
        self.state = "initializing"
        self.experiment_number = self.cfg.experiment_number

        # needs for DL pipeline
        self._init_model()
        self.early_stop = EarlyStopper(patience=self.cfg.pipeline.patience) # toDo Early stoper in hidra init
        self.optimizer = instantiate(cfg.optimizer, params=self.model.parameters())
        self._init_data()
        self._init_losses()
        self._init_mlflow()


    def _init_model(self):
        self.model = instantiate(self.cfg.model)
        if self.cfg.pipeline.model_ckpt is not None:
            self.model.load_state_dict(torch.load(self.cfg.pipeline.model_ckpt), strict=False)
            logger.info(f'Model loaded from checkpoint: {self.cfg.pipeline.model_ckpt}')
        self.model.to(self.device)


    def _init_data(self):
        self.train_dataloader = instantiate(self.cfg.dataset.train.dataloader)
        self.val_dataloader = instantiate(self.cfg.dataset.val.dataloader)
        self.test_dataloader = instantiate(self.cfg.dataset.test.dataloader)


    def _init_losses(self):
        '''
        Тут придется вручную прописать, тк некоторые лоссы требуют инициализацию майнеров
        Возможно в дальнейшем будут добавлены новые лоссы (посмотрим)
        '''
        if 'cross_entropy_loss' in self.cfg.losses:
            self.cross_entropy_loss = instantiate(self.cfg.losses.cross_entropy_loss)

        if 'triplet_loss' in self.cfg.losses:
            self.triplet_loss = instantiate(self.cfg.losses.triplet_loss)
            self.triplet_mainer = instantiate(self.cfg.losses.triplet_mainer)


    def _init_mlflow(self):
        mlflow.set_tracking_uri(uri=self.cfg.mlflow.uri)
        mlflow.set_experiment(self.cfg.mlflow.experiment_name)


    def training_step(self, batch, epoch):
        '''
        Тут есть два варианта, либо в батче якорь и класс, либо там еще есть позитив для триплет лосса.
        Это то, что в рамках одного батча происходит
        '''
        anchor = self.model.forward(batch["anchor"].to(self.device))
        anchor_label = batch["anchor_label"].to(self.device)
        if "positive" in batch:
            positive = self.model.forward(batch["positive"].to(self.device))


        l1 = torch.tensor(0.0, device=self.device)  # cross entropy
        l2 = torch.tensor(0.0, device=self.device)  # triplet loss


        # cross_entropy
        cross_entropy_weight = self.cfg.losses_weight.cross_entropy
        if cross_entropy_weight > 0:
            if self.cfg.transforms.collate_fn is None:
                labels = nn.functional.one_hot(anchor_label.long(), num_classes=self.cfg.num_classes)
                l1 = cross_entropy_weight * self.cross_entropy_loss(anchor["cls"], labels.float().to(self.device))
            else:
                l1 = cross_entropy_weight * self.cross_entropy_loss(anchor["cls"], anchor_label.float().to(self.device))

        # triplet loss
        triplet_loss_weight = self.cfg.losses_weight.triplet

        if triplet_loss_weight > 0:
            embeddings = torch.cat([anchor['emb'], positive['emb']], dim=0).to(self.device)
            labels = torch.concat([anchor_label, anchor_label]).long().to(self.device)
            triplets = self.triplet_mainer(embeddings, labels)
            l2 = triplet_loss_weight * self.triplet_loss(embeddings, labels, triplets)

        loss = l1 + l2

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "train_loss_step": loss.item(),
            "train_cls_loss": l1.item(),
            "train_triplet_loss": l2.item(),
        }

    def train_procedure(self, epoch) -> None:
        self.model.train()
        self.max_len = self.train_dataloader.dataset.max_len

        # Словарь для хранения лоссов
        losses = {
            "train_loss_step": [],
            "train_cls_loss": [],
            "train_triplet_loss": [],
            # "train_arcface_loss": [],
            # "train_contrastive_loss": []
        }

        for step, batch in tqdm(
                enumerate(self.train_dataloader),
                total=len(self.train_dataloader),
                disable=(not self.cfg.pipeline.progress_bar),
                position=2,
                leave=False,
        ):
            train_step = self.training_step(batch, epoch)

            # Добавляем текущий шаг для каждого лосса
            for loss_name in losses.keys():
                step_loss = train_step[loss_name]
                self.postfix[f"{loss_name}_step"] = float(f"{step_loss:.3f}")
                losses[loss_name].append(step_loss)

            # Обновляем прогресс-бар с нужными значениями
            self.pbar.set_postfix({k: self.postfix[k] for k in self.postfix.keys() & {"train_loss_step", "mr1", "mAP"}})

            if step % self.cfg.pipeline.log_steps == 0:
                save_logs(
                    {
                        "epoch": self.postfix["Epoch"],
                        "seq_len": self.max_len,
                        "step": step,
                        **{f"{loss_name}_step": f"{train_step[loss_name]:.3f}" for loss_name in losses.keys()}
                    },
                    output_dir=self.cfg.pipeline.output_dir,
                    name="log_steps",
                )

        # Рассчитываем средние значения для всех лоссов и сохраняем их в self.postfix
        for loss_name, loss_list in losses.items():
            self.postfix[loss_name] = torch.tensor(loss_list).mean().item()

        # Вызов методов валидации и проверки переобучения
        self.val_procedure()
        self.overfit_check()
        self.pbar.set_postfix({k: self.postfix[k] for k in self.postfix.keys() & {"train_loss_step", "mr1", "mAP"}})


    def val_procedure(self) -> None:
        self.model.eval()
        embeddings: Dict[int, torch.Tensor] = {}

        for batch in tqdm(self.val_dataloader, disable=(not self.cfg.pipeline.progress_bar), position=1, leave=False):
            val_dict = self.validation_step(batch)
            if val_dict["emb"].ndim == 1:
                val_dict["emb"] = val_dict["emb"].unsqueeze(0)
            for anchor_id, embedding in zip(val_dict["anchor_id"], val_dict["emb"]):
                embeddings[anchor_id] = embedding

        val_outputs = self.validation_epoch_end(embeddings)

        logger.info(
            f"\n{' Validation Results ':=^50}\n"
            + "\n".join([f'"{key}": {value}' for key, value in self.postfix.items()])
            + f"\n{' End of Validation ':=^50}\n"
        )

        if self.cfg.pipeline.save_val_outputs:
            val_outputs["val_embeddings"] = torch.stack(list(embeddings.values())).numpy()
            save_predictions(val_outputs, output_dir=self.cfg.pipeline.output_dir)
            save_logs(self.postfix, output_dir=self.cfg.pipeline.output_dir)
        self.model.train()

    def validation_epoch_end(self, outputs: Dict[int, torch.Tensor]) -> Dict[int, np.ndarray]:
        clique_ids = []
        for k, (anchor_id, embeddings) in enumerate(outputs.items()):
            clique_id = self.val_dataloader.dataset.version2clique.loc[anchor_id, 'clique']
            clique_ids.append(clique_id)

        preds = torch.stack(list(outputs.values()))
        rranks, average_precisions, ndcg = calculate_ranking_metrics(embeddings=preds.numpy(), cliques=clique_ids)
        self.postfix["mrr"] = rranks.mean()
        self.postfix["mAP"] = average_precisions.mean()
        self.postfix["nDCG_100"] = ndcg.mean()
        return {
            "rranks": rranks,
            "average_precisions": average_precisions,
        }


    def validation_step(self, batch):
        anchor_id = batch["anchor_id"]
        features = self.model.forward(batch["anchor"].to(self.device))
        return {
            "anchor_id": anchor_id.numpy(),
            "emb": features["emb"].squeeze(0).detach().cpu(),
        }

    def overfit_check(self) -> None:
        ## Тут поменял в эрли стопере лосс на ндсиджи
        if self.early_stop(self.postfix["nDCG_100"]):
            logger.info(f"\nValidation not improved for {self.early_stop.patience} consecutive epochs. Stopping...")
            self.state = "early_stopped"

        if self.early_stop.counter > 0:
            logger.info("\nValidation nDcg was not improved")
        else:
            logger.info(f"\nMetric improved. New best score: {self.early_stop.max_score:.3f}")
            save_best_log(self.postfix, output_dir=self.cfg.pipeline.output_dir)

            logger.info("Saving model...")
            epoch = self.postfix["Epoch"]
            max_secs = self.max_len
            prev_model = deepcopy(self.best_model_path)
            self.best_model_path = os.path.join(
                self.cfg.pipeline.output_dir, "model", f"best-model-{epoch=}-{max_secs=}.pt"
            )
            os.makedirs(os.path.dirname(self.best_model_path), exist_ok=True)
            torch.save(deepcopy(self.model.state_dict()), self.best_model_path)
            if prev_model is not None:
                os.remove(prev_model)

    def test_procedure(self) -> None:
        self.model.eval()
        # embeddings: Dict[str, torch.Tensor] = {}
        trackids: List[int] = []
        embeddings: List[np.array] = []
        for batch in tqdm(self.test_dataloader, disable=(not self.cfg.pipeline.progress_bar)):
            test_dict = self.validation_step(batch)
            if test_dict["emb"].ndim == 1:
                test_dict["emb"] = test_dict["emb"].unsqueeze(0)
            for anchor_id, embedding in zip(test_dict["anchor_id"], test_dict["emb"]):
                trackids.append(anchor_id)
                embeddings.append(embedding.numpy())
        predictions = []
        for chunk_result in pairwise_distances_chunked(
                embeddings, metric='cosine', reduce_func=reduce_func, working_memory=100
        ):
            for query_indx, query_nearest_items in chunk_result:
                predictions.append((trackids[query_indx], [trackids[nn_indx] for nn_indx in query_nearest_items]))
        save_test_predictions(
            predictions, output_dir=self.cfg.pipeline.test_output_dir, experiment_number=self.experiment_number
        )

    def test(self) -> None:
        self.test_results: TestResults = {}

        if self.best_model_path is not None:
            self.model.load_state_dict(torch.load(self.best_model_path), strict=False)
            logger.info(f"Best model loaded from checkpoint: {self.best_model_path}")
        elif self.state == "initializing":
            if self.cfg.pipeline.model_ckpt is None:
                logger.warning("Warning: Testing with Random weights")
            else:
                logger.warning(f"Warning: Testing from ckpt {self.cfg.pipeline.model_ckpt}")

        self.state = "running"
        self.test_procedure()
        self.state = "finished"

    def pipeline(self) -> None:
        '''
        Тут короче все обучение и валидация
        Сразу с mlflow трекингом.
        '''
        with mlflow.start_run(run_name=str(self.cfg.mlflow.experiment_number)) as run:
            mlflow.log_params(self.cfg)
            # Просто проверяем, что директория не существует иначе ошибка
            dir_checker(self.cfg.pipeline.output_dir)
            self.state = "running"
            self.pbar = trange(
                self.cfg.pipeline.epochs, disable=(not self.cfg.pipeline.progress_bar), position=0, leave=True
            )

            for epoch in self.pbar:
                if self.state in ["early_stopped", "interrupted", "finished"]:
                    return

                self.postfix["Epoch"] = epoch
                self.pbar.set_postfix(self.postfix)

                try:
                    self.train_procedure(epoch)
                except KeyboardInterrupt:
                    logger.warning("\nKeyboard Interrupt detected. Attempting gracefull shutdown...")
                    self.state = "interrupted"
                except Exception as err:
                    raise (err)

                if self.state == "interrupted":
                    self.val_procedure()
                    self.pbar.set_postfix(
                        {k: self.postfix[k] for k in self.postfix.keys() & {"train_loss_step", "mr1", "nDCG_100"}}
                    )

                for key, value in self.postfix.items():
                    if key == 'Epoch':
                        continue
                    mlflow.log_metric(key, value, step=self.postfix['Epoch'])
        self.state = "finished"