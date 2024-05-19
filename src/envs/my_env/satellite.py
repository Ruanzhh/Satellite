import random


class Satellite:
    def __init__(self, id, neighbor_ids, neighbor_bandwidths, type):
        self.id = id
        self.neighbor = neighbor_ids
        self.neighbor_bandwidths = neighbor_bandwidths
        self.idToBand = {}
        for i in range(4):
            self.idToBand[self.neighbor[i]] = self.neighbor_bandwidths[i]
        self.type = type

    def changeBandwiths(self):
        for i in range(4):
            self.neighbor_bandwidths[i] = max(self.neighbor_bandwidths[i]+random.randint(-3, 3)*0.1, 0)






