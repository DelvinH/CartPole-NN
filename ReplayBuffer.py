from collections import deque
import random

class ReplayBuffer():
    def __init__(self, length):
        self.buffer = deque(maxlen=length)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        sample_size = min(len(self.buffer), batch_size)
        samples = random.choices(self.buffer, k=sample_size)
        return map(list, zip(*samples))