'''
Сгенерировать последовательности, которые состоят из цифр (от 0 до 9) и задаются следующим образом:
x - последовательность цифр
y1 = x1
yi = xi + x1
Если
yi >= 10
    то yi = yi - 10
Hаучить модель рекуррентной нейронной сети предсказывать yi по xi Использовать: RNN, LSTM, GRU
6 баллов за правильно выполненное задание.
'''

import torch
def digit_encoder(x):
    y = torch.zeros_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        if x[i]+x[0] < 10:
            y[i] = (x[i]+x[0])
        else:
            y[i] = (x[i]+x[0]-10)
    return y

x = torch.randint(low=0, high=9, size=(100,1))
print(x.view(-1), digit_encoder(x).view(-1), sep='\n')