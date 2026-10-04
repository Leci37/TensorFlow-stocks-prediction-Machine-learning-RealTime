import time
from collections import defaultdict


# https://stackoverflow.com/questions/5888734/dictionaryqueue-data-structure-with-active-removal-of-old-messages


class QueueMap(object):

    def __init__(self):
        self._expire = defaultdict(lambda *n: defaultdict(int))
        self._store = defaultdict(list)
        self._oldest_key = int(time.time())

    def get_queue(self, name):
        return self._store.get(name, [])

    def pop(self, name):
        queue = self.get_queue(name)
        if queue:
            key, msg = queue.pop()
            self._expire[key][name] -= 1
            return msg
        return None

    def set(self, name, message):
        key = int(time.time())
        # increment count of messages in this bucket/queue
        self._expire[key][name] += 1
        self._store[name].insert(0, (key, message))

    def reap(self, age):
        now = time.time()
        threshold = int(now - age)
        oldest = self._oldest_key

        # iterate over buckets we need to check
        for key in range(oldest, threshold + 1):
            # for each queue with items, expire the oldest ones
            for name, count in self._expire[key].iteritems():
                if count <= 0:
                    continue

                queue = self.get_queue(name)
                while queue:
                    if queue[-1][0] > threshold:
                        break
                    queue.pop()
            del self._expire[key]

        # set oldest_key for next pass
        self._oldest_key = threshold