
class Variant:
    def __init__(self,
                message_channel_count=16,  
                precision_count=1,
                aggregation_buffer_slots=4
    ):
        self.message_channel_count    = message_channel_count 
        self.weight_channel_count     = message_channel_count #Need to change when mesh multiplier changes ?
        self.precision_count          = precision_count
        self.aggregation_buffer_slots = aggregation_buffer_slots
