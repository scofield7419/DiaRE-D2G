
class Node:
    def __init__(self, id, v_id, v_no, sent_id, pos_start, pos_end):
        self.id = id
        self.v_id = v_id
        self.v_no = v_no
        self.sent_id = sent_id
        self.pos_start = pos_start
        self.pos_end = pos_end


class Accuracy(object):
    def __init__(self):
        self.correct = 0
        self.total = 0

    def add(self, is_correct):
        self.total += 1
        if is_correct:
            self.correct += 1

    def get(self):
        if self.total == 0:
            return 0.0
        else:
            return float(self.correct) / self.total

    def clear(self):
        self.correct = 0
        self.total = 0
