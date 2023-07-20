class DirecNode:
    def __init__(self, key, val, pre=None, next_=None):
        self.key = key
        self.val = val
        self.pre = pre
        self.next = next_


class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._log = dict()
        self._head = DirecNode(None, None)
        self._tail = DirecNode(None, None)
        self._head.next = self._tail
        self._tail.pre = self._head

    def get(self, key: int) -> int:
        if key not in self._log:
            return -1
        node = self._log[key]
        self._move_to_head(node)
        return self._log[key].val

    def _move_to_head(self, node):
        node.pre.next, node.next.pre = node.next, node.pre
        old = self._head.next
        self._head.next = node
        node.pre = self._head
        node.next = old
        old.pre = node

    def _add_node(self, node):
        head_next = self._head.next
        self._head.next, node.pre = node, self._head
        node.next, head_next.pre = head_next, node

    def put(self, key: int, value: int) -> None:
        if key in self._log:
            self._log[key].val = value
            self._move_to_head(self._log[key])
        else:
            node = DirecNode(key, value)
            self._add_node(node)
            self._log[key] = node
            if len(self._log) > self.capacity:
                self._log.pop(self._tail.pre.key)
                self._tail.pre.pre.next = self._tail
                self._tail.pre = self._tail.pre.pre




lru = LRUCache(2)
lru.put(1, 0)
lru.put(2, 2)
print(lru.get(1))
lru.put(3, 3)
print(lru.get(2))
lru.put(4, 4)
print(lru.get(1))
print(lru.get(3))
print(lru.get(4))


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)