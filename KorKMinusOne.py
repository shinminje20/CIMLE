from random import sample

class KorKMinusOne:
    def __init__(self, idxs, shuffle=False):
        self.counter = 0
        self.shuffle = shuffle
        self.idxs = idxs
    def pop(self):
        if self.counter == len(self.idxs):
            self.counter = 0
            self.idxs = sample(self.idxs, k=len(self.idxs)) if self.shuffle else self.idxs
        
        result = self.idxs[self.counter]
        self.counter += 1
        return result