class DataStream:
    def __init__(self, id, satellite_id, data_amount):
        self.id = id
        self.curr_satellite_id = satellite_id
        self.arrive_time = None
        self.next_satellite_id = None
        self.data_amount = data_amount
        self.isTransmitting = False

