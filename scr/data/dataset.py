import os
from typing import Dict, Literal, Tuple, Union, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class UnsqueezeTransform:
    """Пользовательская трансформация для добавления размерности."""
    def __call__(self, tensor):
        return tensor.unsqueeze(0)  #

class DuplicateChannels:
    """Пользовательская трансформация для дублирования каналов."""
    def __call__(self, tensor):
        return tensor.repeat(3, 1, 1)  # Дублируем канал до 3 (3, 224, 224)

class CoverDatasetBase(Dataset):
    def __init__(
            self,
            data_path: str,
            dataset_path: str,
            data_split: Literal["train", "val", "test"],
            file_ext: str = "npy",
            transforms: Union[None, transforms.Compose] = None,
            augmentations: Union[None, Any] = None,
            max_len: int = 50,
            **kwargs,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.file_ext = file_ext
        self.transforms = transforms
        self.augmentations = augmentations
        self.max_len = max_len
        # get kwargs attrs
        for key, value in kwargs.items():
            setattr(self, key, value)

        self._load_data()

    def __len__(self) -> int:
        return len(self.track_ids)

    def _load_data(self) -> None:
        '''
        Тут собираются все таблички, со взаимосвязями кликов и треков
        Кликаи нумеруются индексами от 0 до число кликов

        track_ids - список всез треков которые попали в версии
        versions - не плоская табличка с колонкой клик и колонками из списка версий (айди кликов заменяются на айдишники
            от 0 до кол-во треков в кликах в порядке сортировки по возрастанию)
        version2clique - плоская табличка колонки - версии и клик
        Returns:
        '''
        if self.data_split in ['train', 'val']:
            # нмпай массив с множеством кликов в сплите
            cliques_subset = np.load(os.path.join(self.data_path, "splits", "{}_cliques.npy".format(self.data_split)))

            # айди клика и множество треков каверов
            self.versions = pd.read_csv(
                os.path.join(self.data_path, "cliques2versions.tsv"), sep='\t', converters={"versions": eval}
            )
            self.versions = self.versions[self.versions["clique"].isin(set(cliques_subset))]
            mapping = {}
            # каждой клике ставим айдишник от 0 до количетсва кликов
            for k, clique in enumerate(sorted(cliques_subset)):
                mapping[clique] = k
            self.versions["clique"] = self.versions["clique"].map(lambda x: mapping[x])
            self.versions.set_index("clique", inplace=True)

            # Тут делаем табличку с кликами плоской, будет две колонки версия и клик (в версии ток один айди трека)
            # Верися являетс индексом
            self.version2clique = pd.DataFrame(
                [{'version': version, 'clique': clique} for clique, row in self.versions.iterrows() for version in
                 row['versions']]
            ).set_index('version')
            self.track_ids = self.version2clique.index.to_list()
            # грузим заготовку с негативами откуда будем семплировать их
            # формат заготовки track_id - индекс negatives - список из 200 негативов (это уже айдишники треков)
        else:
            self.track_ids = np.load(os.path.join(self.data_path, "splits", "{}_ids.npy".format(self.data_split)))

    def _make_file_path(self, track_id, file_ext):
        a = track_id % 10
        b = track_id // 10 % 10
        c = track_id // 100 % 10
        return os.path.join(str(c), str(b), str(a), f'{track_id}.{file_ext}')

    def _load_cqt(self, track_id: str) -> torch.Tensor:
        filename = os.path.join(self.dataset_path, self._make_file_path(track_id, self.file_ext))
        cqt_spectrogram = np.load(filename)
        return torch.from_numpy(cqt_spectrogram)


class CoverDatasetClassifier(CoverDatasetBase):
    """
    Возвращает только якорь и его лейбл (нужно для классификатора)
    """
    def __init__(
        self,
        data_path: str,
        dataset_path: str,
        data_split: Literal["train", "val", "test"],
        file_ext: str = "npy",
        transforms: Union[None, transforms.Compose] = None,
        augmentations: Union[None, Any] = None,
        **kwargs,
    ) -> None:
        # Вызываем инициализатор базового класса
        super().__init__(data_path, dataset_path, data_split, file_ext, transforms, augmentations, **kwargs)

    def __getitem__(self, index: int) -> Dict:
        track_id = self.track_ids[index]
        anchor_cqt = self._load_cqt(track_id)

        # Трансформации к якорю
        if self.transforms is not None:
            anchor_cqt = self.transforms(anchor_cqt)

        if self.data_split == "train":
            # Аугментации к якорю только в трейне
            if self.augmentations is not None:
                anchor_cqt = self.augmentations(anchor_cqt)
            clique_id = self.version2clique.loc[track_id, 'clique']
        else:
            clique_id = -1
        return dict(
            anchor_id=track_id,
            anchor=anchor_cqt,
            anchor_label=torch.tensor(clique_id, dtype=torch.float),
        )

class CoverDatasetPair(CoverDatasetBase):
    '''
    Отличие от предыдущего датасета в том, что возвращает к якорю позитив.
    Это нужно ы целом для триплет лосса и для контрастивного лосса.
    Да и в целом для чего угодно пригодиться может.
    '''
    def __init__(
        self,
        data_path: str,
        dataset_path: str,
        data_split: Literal["train", "val", "test"],
        file_ext: str = "npy",
        transforms: Union[None, transforms.Compose] = None,
        augmentations: Union[None, Any] = None,
        **kwargs,
    ) -> None:
        # Вызываем инициализатор базового класса
        super().__init__(data_path, dataset_path, data_split, file_ext, transforms, augmentations, **kwargs)


    def __getitem__(self, index: int) -> Dict:
        track_id = self.track_ids[index]
        anchor_cqt = self._load_cqt(track_id)
        if self.transforms is not None:
            anchor_cqt = self.transforms(anchor_cqt)

        if self.data_split == "train":
            clique_id = self.version2clique.loc[track_id, 'clique']
            pos_id = self._pos_sampling(track_id, clique_id)
            positive_cqt = self._load_cqt(pos_id)

            if self.transforms is not None:
                positive_cqt = self.transforms(positive_cqt)

            if self.augmentations is not None:
                positive_cqt = self.augmentations(positive_cqt)
                anchor_cqt = self.augmentations(anchor_cqt)
        else:
            clique_id = -1
            pos_id = torch.empty(0)
            positive_cqt = torch.empty(0)
        return dict(
            anchor_id=track_id,
            anchor=anchor_cqt,
            anchor_label=torch.tensor(clique_id, dtype=torch.float),
            positive_id=pos_id,
            positive=positive_cqt,
        )

    def _pos_sampling(self, track_id: int, clique_id: int) -> Tuple[int, int]:
        '''
        На вход поется айдишник трека и айдишник клика из которого будет выбран позитив
        Трек берется в качетсве якоря из его клики выбирается любой другой трек в качестве позитива
        '''
        versions = self.versions.loc[clique_id, "versions"]
        pos_list = np.setdiff1d(versions, track_id)
        pos_id = np.random.choice(pos_list, 1)[0]
        return pos_id