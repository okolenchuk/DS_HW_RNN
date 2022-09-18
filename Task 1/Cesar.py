# from io import open
import re
import torch
import warnings
import time

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_path = 'Twilight.txt'
string_size = 60
batch_size = 10
num_epoch = 20
lr = 0.01


class Cesar(object):
    def __init__(self, step):
        self.step = step
        self.alphabet = ''
        self.len_alphabet = 0

    def alphabet_from_file(self, file_path):
        with open(file_path) as file:
            while True:
                text = file.read(string_size)
                if not text:
                    break
                for ch in text:
                    if ch not in self.alphabet:
                        self.alphabet += ch
        self.alphabet = re.sub(r'[^a-zA-Z.!? ]+', r'', ''.join(sorted(self.alphabet)))
        self.len_alphabet = len(self.alphabet)

    def encode(self, text):
        res = ''
        for c in text:
            if c in self.alphabet:
                res += self.alphabet[(self.alphabet.index(c) + self.step) % len(self.alphabet)]
        return res

    def decode(self, text):
        res = ''
        for c in text:
            res += self.alphabet[(self.alphabet.index(c) - self.step % len(self.alphabet))]
        return res


coder = Cesar(2)
coder.alphabet_from_file(file_path)
alpha = coder.alphabet


def sent_to_index(sentence):
    return [alpha.find(y) for y in sentence]


def make_dataset(file_path, step):
    text_array = []
    with open(file_path) as file:
        while True:
            text = file.read(step)
            if not text:
                break
            text_array.append(re.sub(r'[^a-zA-Z.!? ]', r' ', text))
    del text_array[-1]
    y_train = torch.tensor([sent_to_index(lines) for lines in text_array[len(text_array) // 5:]])
    x_train = torch.tensor([sent_to_index(coder.encode(lines)) for lines in text_array[len(text_array) // 5:]])

    y_test = torch.tensor([sent_to_index(lines) for lines in text_array[:len(text_array) // 5]])
    x_test = torch.tensor([sent_to_index(coder.encode(lines)) for lines in text_array[:len(text_array) // 5]])

    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = make_dataset(file_path, string_size)


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, x, y):
        super().__init__()
        self._len = len(x)
        self.y = y.to(DEVICE)
        self.x = x.to(DEVICE)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


train_ds = torch.utils.data.DataLoader(MyDataset(x_train, y_train),
                                       batch_size=batch_size,
                                       shuffle=True)
test_ds = torch.utils.data.DataLoader(MyDataset(x_test, y_test),
                                      batch_size=batch_size,
                                      shuffle=True)


class RNNModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(60, 32)
        self.rnn = torch.nn.RNN(32, 128, batch_first=True)
        self.linear = torch.nn.Linear(128, len(alpha))

    def forward(self, sentence, state=None):
        embed = self.embed(sentence)
        o, h = self.rnn(embed)
        return self.linear(o)


model = RNNModel().to(DEVICE)
loss = torch.nn.CrossEntropyLoss().to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

for epoch in range(num_epoch):
    train_loss, train_acc, iter_num = .0, .0, .0
    start_epoch_time = time.time()
    model.train()
    for x_in, y_in in train_ds:
        x_in = x_in.to(DEVICE)
        y_in = y_in.view(1, -1).squeeze().to(DEVICE)
        optimizer.zero_grad()
        out = model.forward(x_in).view(-1, len(alpha))
        l = loss(out, y_in)
        train_loss += l.item()
        batch_acc = (out.argmax(dim=1) == y_in)
        train_acc += batch_acc.sum().item() / batch_acc.shape[0]
        l.backward()
        optimizer.step()
        iter_num += 1
    print(
        f"Epoch: {epoch + 1}, loss: {train_loss:.4f}, acc: "
        f"{train_acc / iter_num:.4f}",
        end=" | "
    )
    test_loss, test_acc, iter_num = .0, .0, .0
    model.eval()
    for x_in, y_in in test_ds:
        x_in = x_in.to(DEVICE)
        y_in = y_in.view(1, -1).squeeze()
        out = model.forward(x_in).view(-1, len(alpha)).to(DEVICE)
        l = loss(out, y_in)
        test_loss += l.item()
        batch_acc = (out.argmax(dim=1) == y_in)
        test_acc += batch_acc.sum().item() / batch_acc.shape[0]
        iter_num += 1
    print(f"test loss: {test_loss:.4f}, test acc: {test_acc / iter_num:.4f} | "
          f"{time.time() - start_epoch_time:.2f} sec.")
