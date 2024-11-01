import torch.nn as nn
import torch.nn.functional as F


class DepthwiseConv(nn.Module):
    """Глубинная свертка для Conformer."""

    def __init__(self, dim, kernel_size=31):
        super(DepthwiseConv, self).__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.pointwise = nn.Conv1d(dim, dim, kernel_size=1)
        self.batch_norm = nn.BatchNorm1d(dim)

    def forward(self, x):
        x = self.conv(x)
        x = F.gelu(x)
        x = self.pointwise(x)
        x = self.batch_norm(x)
        return x


class FeedForwardModule(nn.Module):
    """Feedforward модуль с промежуточной размерностью."""

    def __init__(self, dim, expansion_factor=4, dropout=0.1):
        super(FeedForwardModule, self).__init__()
        self.linear1 = nn.Linear(dim, dim * expansion_factor)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim * expansion_factor, dim)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x + residual


class ConformerBlock(nn.Module):
    """Основной блок Conformer."""

    def __init__(self, dim, num_heads=4, expansion_factor=4, kernel_size=31, dropout=0.1):
        super(ConformerBlock, self).__init__()
        self.ff1 = FeedForwardModule(dim, expansion_factor, dropout)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.conv = DepthwiseConv(dim, kernel_size)
        self.ff2 = FeedForwardModule(dim, expansion_factor, dropout)
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Первичный feed-forward модуль
        x = x + 0.5 * self.ff1(x)

        # Механизм внимания
        residual = x
        x = self.layer_norm(x)
        x = x.transpose(0, 1)  # Перемещаем batch и временные размерности для MultiheadAttention
        x, _ = self.self_attn(x, x, x)
        x = self.dropout(x).transpose(0, 1) + residual  # Вернуть обратно размерности

        # Сверточный модуль
        x = x + self.conv(x.transpose(1, 2)).transpose(1, 2)

        # Вторичный feed-forward модуль
        x = x + 0.5 * self.ff2(x)
        return x


class Conformer(nn.Module):
    def __init__(self, input_dim=84, num_blocks=4, dim=144, emb_size=128, num_heads=4, kernel_size=31,
                 expansion_factor=4, dropout=0.1, num_classes=10):
        super(Conformer, self).__init__()

        # Линейная проекция входа в требуемое количество измерений
        self.input_projection = nn.Linear(input_dim, dim)

        # Стек conformer-блоков
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(dim, num_heads, expansion_factor, kernel_size, dropout)
            for _ in range(num_blocks)
        ])

        # Линейные головы для эмбеддингов и классификации
        self.emb_head = nn.Linear(dim, emb_size)
        self.cls_head = nn.Linear(emb_size, num_classes)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (batch_size, 84, 50)
        x = x.transpose(1, 2)  # Преобразовать в (batch_size, 50, 84)
        x = self.input_projection(x)  # Преобразование размерности с 84 на dim

        # Прогон через conformer-блоки
        for block in self.conformer_blocks:
            x = block(x)

        x = self.layer_norm(x)

        # Усреднение по временной оси
        pooled_x = x.mean(dim=1)  # (batch_size, dim)

        # Выходные эмбеддинги и классификационные предсказания
        emb = self.emb_head(pooled_x)
        cls = self.cls_head(emb)

        return {"emb": emb, "cls": cls}