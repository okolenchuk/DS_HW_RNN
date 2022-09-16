class Cesar(object):
    def __init__(self, step):
        self.step = step
        self.alpha = 'abcdefghijklmnopqrstuvwxyz .,-;:'

    def encode(self, text):
        res = ''
        for c in text.lower():
            if c in self.alpha:
                res += self.alpha[(self.alpha.index(c) + self.step) % len(self.alpha)]
        return res

    def decode(self, text):
        res = ''
        for c in text.lower():
            res += self.alpha[(self.alpha.index(c) - self.step% len(self.alpha))]
        return res

coder = Cesar(2)

a = 'Gensim is billed as a Natural Language Processing package that does ‘Topic Modeling for Humans’. But it is practically much more than that. It is a leading and a state-of-the-art package for processing texts, working with word vector models (such as Word2Vec, FastText etc) and for building topic models'
b = coder.encode(a)

print(b)
print(coder.decode(b))