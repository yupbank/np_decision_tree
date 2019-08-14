import heapq


def push_large(l, x):
    heapq.heappush(l, x)


def pop_large(l):
    return heapq.heappop(l)


def top_large(l):
    return l[0]


def push_small(l, x):
    heapq.heappush(l, -x)


def pop_small(l):
    return -heapq.heappop(l)


def top_small(l):
    return -l[0]


class StreamMedian(object):
    def __init__(self):
        self.small = []
        self.large = []

    def add(self, num):
        if len(self.small) == 0:
            return push_small(self.small, num)

        if num <= top_small(self.small):
            push_small(self.small, num)
        else:
            push_large(self.large, num)

        if len(self.small) > len(self.large)+1:
            push_large(self.large, pop_small(self.small))

        if len(self.large) > len(self.small)+1:
            push_small(self.small, pop_large(self.large))

    def median(self):
        if len(self.large) == len(self.small):
            return (top_large(self.large)+top_small(self.small))/2
        if len(self.large) > len(self.small):
            return top_large(self.large)
        else:
            return top_small(self.small)

    def double_median(self):
        if len(self.large) == len(self.small):
            return top_small(self.small), top_large(self.large)

        if len(self.large) > len(self.small):
            median = top_large(self.large)
        else:
            median = top_small(self.small)
        return median, median

    def __call__(self, num):
        self.add(num)
        return self.double_median()
