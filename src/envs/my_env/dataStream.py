import math

class DataStream:
    def __init__(self, id, satellite_id, data_amount):
        self.id = id
        self.curr_satellite_id = satellite_id
        self.arrive_time = None
        self.next_satellite_id = None
        self.data_amount = data_amount
        self.min_data_amount = data_amount * 0.5
        self.original_data_amount = data_amount
        self.isTransmitting = False
        self.alpha = 100
    def compute_data_loss_penalty(self):
        loss = (self.original_data_amount-self.data_amount) / self.original_data_amount
        penalty = math.exp(loss*10) - 1
        return -penalty

