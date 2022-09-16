from io import open
import unicodedata
import string
import re
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        # Как помним hidden_size - размер скрытого состояния
        self.hidden_size = hidden_size
        # Слой Эмбеддингов, который из входного вектора последовательности (либо батча) отдаст представление последовательности для скрытого состояния
        # FYI: в качестве Input_size у нас размер словаря
        self.embedding = nn.Embedding(input_size, hidden_size)
        # И соответственно рекуррентная ячейка GRU которая принимает MxM (hidden на hidden)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        # Приводим эмбеддинг к формату одного предлоежния 1х1 и любая размерность
        embedded = self.embedding(input).view(1, 1, -1)
        # Нужно для следующего шага пока не запутываемся :) просто присвоили наш эмбеддинг
        output = embedded
        # и соответственно подаем все в ГРЮ ячейку (эмбеддинг и скрытые состояния)
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        # Дополнительно сделаем инициализацию скрытого представления (просто заполним нулями)
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    # Будьте внимательны, теперь на вход мы получаем размер скрытого представления
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        # Слой эмбеддингов - рамер словаря, размер скрытого представления
        self.embedding = nn.Embedding(output_size, hidden_size)
        # GRU скрытое состояние на скрытое
        self.gru = nn.GRU(hidden_size, hidden_size)
        # Переводим hidden size в распределение для этого передаем в линейный слов скрытое состояни и размер словаря
        self.out = nn.Linear(hidden_size, output_size)
        # Получаем распределение верояностей
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0])) # берем output по нулевому индексу (одно предложение)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)