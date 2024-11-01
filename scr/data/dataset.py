import os
from typing import Dict, Literal, Tuple, Union, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, default_collate
from torchvision import transforms
import torch.nn.functional as F

from typing import List
import numpy as np

class MixUp1d:
    def __init__(self, num_classes, alpha=1.0):
        self.num_classes = num_classes
        self.alpha = alpha

    def __call__(self, x, y):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]

        # Преобразование y в one-hot векторы
        y_one_hot = torch.zeros(batch_size, self.num_classes, device=x.device)
        y_one_hot[range(batch_size), y] = 1  # Преобразуем в one-hot

        mixed_y = lam * y_one_hot + (1 - lam) * y_one_hot[index]

        return mixed_x, mixed_y


class CutMix1d:
    def __init__(self, num_classes, alpha=1.0):
        self.num_classes = num_classes
        self.alpha = alpha

    def __call__(self, x, y):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size)

        W = x.size(2)
        H = x.size(1)

        r_x = np.random.randint(0, W)
        r_y = np.random.randint(0, H)
        r_w = int(W * np.sqrt(1 - lam))
        r_h = int(H * np.sqrt(1 - lam))

        r_x1 = np.clip(r_x - r_w // 2, 0, W)
        r_x2 = np.clip(r_x + r_w // 2, 0, W)
        r_y1 = np.clip(r_y - r_h // 2, 0, H)
        r_y2 = np.clip(r_y + r_h // 2, 0, H)

        # Создаем смешанные данные
        x[:, r_y1:r_y2, r_x1:r_x2] = x[index, r_y1:r_y2, r_x1:r_x2]

        # Преобразование y в one-hot векторы
        y_one_hot = torch.zeros(batch_size, self.num_classes, device=x.device)
        y_one_hot[range(batch_size), y] = 1  # Преобразуем в one-hot

        # Смешиваем метки
        y_a = y_one_hot
        y_b = y_one_hot[index]
        mixed_y = lam * y_a + (1 - lam) * y_b

        return x, mixed_y



class AugmentationCollator:
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, batch):

        anchor_ids = [item['anchor_id'] for item in batch]  # ID без изменений
        anchors = torch.stack([item['anchor'] for item in batch])  # Тензор изображений
        labels = torch.stack([item['anchor_label'] for item in batch])  # Тензор меток

        augmented_anchors, augmented_labels = self.augmentations(anchors, labels.long())

        return {
            'anchor_id': anchor_ids,
            'anchor': augmented_anchors,
            'anchor_label': augmented_labels
        }


class ResizeSpectrogram:

    def __init__(self, target_size=(80, 50)):
        self.target_size = tuple(target_size)

    def __call__(self, tensor):
        tensor = tensor.unsqueeze(0)  # Добавляем размерность: (1, 84, 50)
        resized_tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(self.target_size[0],
            tensor.shape[2]),
            mode='bicubic',
            align_corners=False
        )

        return resized_tensor.squeeze(0).squeeze(0)

class InterpolateTransform:
    """
    Это нужно для приведения к виду мелл спектрограммы
    """
    def __init__(self, target_size=(96, 64)):
        self.target_size = tuple(target_size)

    def __call__(self, tensor):
        tensor = tensor.unsqueeze(0)  # Размер (1, H, W)
        tensor = F.interpolate(tensor.unsqueeze(0), size=self.target_size, mode='bicubic', align_corners=False)
        return tensor.squeeze(0)  #


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