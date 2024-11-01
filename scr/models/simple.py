import torch.nn as nn


class SimpleNN(nn.Module):
    def __init__(self, input_height=84, input_width=50, emb_size=512, num_classes=8858):
        super(SimpleNN, self).__init__()

        # Расчет входного размера после flatten
        self.input_size = input_height * input_width

        # Добавляем первый полносвязный слой
        self.fc1 = nn.Linear(self.input_size, emb_size)
        self.relu1 = nn.ReLU()

        # Второй полносвязный слой для классификации
        self.fc2 = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        # Изменяем форму входного тензора на (batch_size, input_size)
        x = x.view(x.size(0), -1)  # либо используйте nn.Flatten()

        # Прогон через первый слой и активацию
        emb = self.relu1(self.fc1(x))

        # Прогон через второй слой
        cls = self.fc2(emb)

        # Возвращаем словарь с эмбеддингом и классификацией
        return dict(emb=emb, cls=cls)