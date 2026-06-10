"""
Assignment : Data Structures & Systems Design(SE Intern)
Role  : AI/ML Engineering Intern — Agentic AI Systems
Author: Yagyaraj Pandey
Date  : June 2026
"""

import heapq
import threading
import unittest


# ══════════════════════════════════════════════════════════════════
# PROBLEM 1 — LRU Cache Implementation
# ══════════════════════════════════════════════════════════════════

"""
My Approach
-----------
I needed O(1) for both get and put, which means I can't afford any
kind of scanning or traversal. After thinking it through, I realised
two data structures need to work together:

The first is a HashMap. It lets me jump directly to any node by key
in O(1) — no searching involved.

The second is a Doubly Linked List. This is what maintains the usage
order. I keep the most recently used item near the head and the least
recently used item near the tail. The reason I chose doubly linked
over singly linked is that when I want to remove a node from the
middle of the list, I need to update both its previous and next
neighbours. With a singly linked list I'd have to traverse to find
the previous node, which breaks the O(1) requirement. With doubly
linked, every node already knows its prev and next, so removal and
insertion are both O(1) no matter where in the list we are.

I also added two dummy sentinel nodes at the head and tail. This
removes all the awkward edge-case checks you'd normally need when
the list is empty or has only one element.

On get: I find the node via the hashmap, unlink it from its current
position, and reinsert it right after the head. This marks it as
most recently used.

On put: If the key already exists I just update the value and move
it to the head. If it's a new key and the cache is full, I remove
the node just before the tail sentinel (that's the LRU item), delete
it from the hashmap too, then insert the new node at the head.

I also added a stats() method that tracks hits, misses, and evictions
in real time. In a production system you'd want this to monitor
whether the cache size is tuned correctly.
"""


class Node:
    """
    A single node in the doubly linked list.

    Each node stores a key-value pair along with pointers to its
    previous and next neighbours in the list.
    """

    def __init__(self, key: int = 0, value: int = 0) -> None:
        self.key   = key
        self.value = value
        self.prev: "Node | None" = None
        self.next: "Node | None" = None


class LRUCache:
    """
    Least-Recently-Used cache with O(1) get and put operations.

    Internally uses a HashMap for O(1) key lookup and a doubly
    linked list to maintain usage order. The head sentinel points
    toward the most recently used item; the tail sentinel's previous
    neighbour is always the least recently used item and the first
    candidate for eviction.

    Parameters
    ----------
    capacity : int
        Maximum number of key-value pairs the cache can hold.

    Examples
    --------
    >>> cache = LRUCache(2)
    >>> cache.put(1, 1)
    >>> cache.put(2, 2)
    >>> cache.get(1)
    1
    >>> cache.put(3, 3)   # evicts key 2
    >>> cache.get(2)
    -1
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.cache: dict[int, Node] = {}

        # Performance counters
        self._hits      = 0
        self._misses    = 0
        self._evictions = 0

        # Dummy sentinels — they never hold real data
        self._head = Node()   # most-recent side
        self._tail = Node()   # least-recent side
        self._head.next = self._tail
        self._tail.prev = self._head

    def _remove(self, node: Node) -> None:
        """Unlink a node from its current position in the list. O(1)."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _insert_at_head(self, node: Node) -> None:
        """Insert a node right after the head sentinel. O(1)."""
        node.next            = self._head.next
        node.prev            = self._head
        self._head.next.prev = node
        self._head.next      = node

    def get(self, key: int) -> int:
        """
        Return the value for key, or -1 if not present.

        Also moves the key to most-recently-used position so it is
        protected from eviction.

        Time  : O(1)
        Space : O(1)
        """
        if key not in self.cache:
            self._misses += 1
            return -1
        node = self.cache[key]
        self._remove(node)
        self._insert_at_head(node)
        self._hits += 1
        return node.value

    def put(self, key: int, value: int) -> None:
        """
        Insert or update key with value.

        If the cache is at capacity, the least-recently-used entry
        is evicted first.

        Time  : O(1)
        Space : O(n) where n = capacity
        """
        if key in self.cache:
            node       = self.cache[key]
            node.value = value
            self._remove(node)
            self._insert_at_head(node)
        else:
            if len(self.cache) == self.capacity:
                lru = self._tail.prev
                self._remove(lru)
                del self.cache[lru.key]
                self._evictions += 1
            new_node        = Node(key, value)
            self.cache[key] = new_node
            self._insert_at_head(new_node)

    def stats(self) -> dict:
        """
        Return a snapshot of cache performance counters.

        Useful for monitoring whether the cache capacity is
        appropriately sized for the workload.

        Returns
        -------
        dict with keys: hits, misses, evictions, hit_rate, size

        Time  : O(1)
        Space : O(1)
        """
        total    = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total else 0.0
        return {
            "hits"      : self._hits,
            "misses"    : self._misses,
            "evictions" : self._evictions,
            "hit_rate"  : f"{hit_rate:.1f}%",
            "size"      : len(self.cache),
        }

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"LRUCache(capacity={self.capacity}, "
            f"size={s['size']}, "
            f"hit_rate={s['hit_rate']})"
        )


class ThreadSafeLRUCache(LRUCache):
    """
    A thread-safe extension of LRUCache.

    Both get and put modify shared state across multiple steps. If
    two threads call these methods simultaneously without any
    synchronisation, one can corrupt the linked list while the other
    is mid-operation. Wrapping every method with a threading.Lock
    ensures only one thread accesses the shared state at a time.

    For workloads that are read-heavy, a readers-writer lock would
    allow multiple threads to read simultaneously while still
    serialising writes, which would improve throughput further.

    Time  : O(1) per operation (same as base class, plus lock overhead)
    Space : O(n) + O(1) for the lock object
    """

    def __init__(self, capacity: int) -> None:
        super().__init__(capacity)
        self._lock = threading.Lock()

    def get(self, key: int) -> int:
        with self._lock:
            return super().get(key)

    def put(self, key: int, value: int) -> None:
        with self._lock:
            super().put(key, value)

    def stats(self) -> dict:
        with self._lock:
            return super().stats()


# ══════════════════════════════════════════════════════════════════
# PROBLEM 2 — Event Scheduler
# ══════════════════════════════════════════════════════════════════

"""
My Approach
-----------
For can_attend_all, I sort the events by start time first. Then I
walk through them one by one. If the next event starts strictly
before the current one ends, that's an overlap and I return False
immediately. If an event's start time is exactly equal to the
previous event's end time, that's fine — they share a boundary but
don't actually conflict, so I treat that as no overlap, as stated
in the constraints.

For min_rooms_required, the key insight is that I don't need to
track which specific events are running at any moment — I just need
to know how many are running simultaneously at the peak.

I separate all start times into one sorted list and all end times
into another sorted list. Then I use two pointers, one moving
through starts and one through ends, and simulate a running count
of active events.

When the next start time comes before the earliest end time, a new
event is beginning while an existing one is still ongoing, so I
need one more room. When the earliest end time comes first, a room
has just been freed up. I track the highest count reached — that's
the minimum number of rooms needed.

This avoids using a heap entirely and runs in O(1) extra space
beyond the two sorted arrays, which I find more elegant.

I also implemented assign_rooms() as an extension for the future
proofing question — it assigns a named room (Room A, Room B, ...) 
to each event using a min-heap.
"""


def can_attend_all(events: list[tuple]) -> bool:
    """
    Return True if one person can attend every event without overlap.

    Events that share only a boundary (end_i == start_{i+1}) are
    not considered overlaps per the problem constraints.

    Parameters
    ----------
    events : list of (start, end) tuples

    Returns
    -------
    bool

    Time  : O(n log n)
    Space : O(n)

    Examples
    --------
    >>> can_attend_all([(9, 10), (10, 11), (11, 12)])
    True
    >>> can_attend_all([(9, 11), (10, 12)])
    False
    """
    if not events:
        return True

    sorted_events = sorted(events, key=lambda e: e[0])

    for i in range(1, len(sorted_events)):
        if sorted_events[i][0] < sorted_events[i - 1][1]:
            return False

    return True


def min_rooms_required(events: list[tuple]) -> int:
    """
    Return the minimum number of meeting rooms needed to hold all
    events simultaneously without conflicts.

    Uses a two-pointer sweep on separately sorted start and end
    arrays. This avoids a heap and achieves O(1) extra space beyond
    the two auxiliary arrays.

    Parameters
    ----------
    events : list of (start, end) tuples

    Returns
    -------
    int

    Time  : O(n log n)
    Space : O(n)

    Examples
    --------
    >>> min_rooms_required([(9, 10), (10, 11), (11, 12)])
    1
    >>> min_rooms_required([(9, 11), (10, 12), (11, 13)])
    2
    """
    if not events:
        return 0

    starts = sorted(e[0] for e in events)
    ends   = sorted(e[1] for e in events)

    rooms = max_rooms = 0
    s = e = 0

    while s < len(starts):
        if starts[s] < ends[e]:
            rooms += 1
            s     += 1
        else:
            rooms -= 1
            e     += 1
        max_rooms = max(max_rooms, rooms)

    return max_rooms


def assign_rooms(events: list[tuple]) -> dict:
    """
    Assign a named room (Room A, Room B, ...) to each event.

    Uses a min-heap to always reuse the room whose current event
    ends earliest, minimising the total number of rooms allocated.

    Parameters
    ----------
    events : list of (start, end) tuples

    Returns
    -------
    dict mapping each event tuple to its assigned room name

    Time  : O(n log n)
    Space : O(n)

    Examples
    --------
    >>> assign_rooms([(9, 11), (10, 12)])
    {(9, 11): 'Room A', (10, 12): 'Room B'}
    """
    if not events:
        return {}

    heap    = []
    counter = 0
    assignments = {}

    indexed = sorted(enumerate(events), key=lambda x: x[1][0])

    for orig_idx, (start, end) in indexed:
        if heap and heap[0][0] <= start:
            _, room = heapq.heapreplace(heap, (end, heap[0][1]))
        else:
            room = f"Room {chr(65 + counter)}"
            counter += 1
            heapq.heappush(heap, (end, room))
        assignments[orig_idx] = room

    return {events[i]: r for i, r in assignments.items()}


# ══════════════════════════════════════════════════════════════════
# UNIT TESTS
# ══════════════════════════════════════════════════════════════════

class TestLRUCache(unittest.TestCase):

    def test_basic_get_put(self):
        cache = LRUCache(2)
        cache.put(1, 10)
        self.assertEqual(cache.get(1), 10)

    def test_missing_key_returns_minus_one(self):
        cache = LRUCache(2)
        self.assertEqual(cache.get(99), -1)

    def test_evicts_lru_on_overflow(self):
        cache = LRUCache(2)
        cache.put(1, 1)
        cache.put(2, 2)
        cache.put(3, 3)
        self.assertEqual(cache.get(1), -1)
        self.assertEqual(cache.get(2), 2)
        self.assertEqual(cache.get(3), 3)

    def test_get_refreshes_recency(self):
        cache = LRUCache(2)
        cache.put(1, 1)
        cache.put(2, 2)
        cache.get(1)
        cache.put(3, 3)
        self.assertEqual(cache.get(1), 1)
        self.assertEqual(cache.get(2), -1)

    def test_update_existing_key(self):
        cache = LRUCache(2)
        cache.put(1, 1)
        cache.put(1, 100)
        self.assertEqual(cache.get(1), 100)
        self.assertEqual(len(cache.cache), 1)

    def test_capacity_one(self):
        cache = LRUCache(1)
        cache.put(1, 1)
        cache.put(2, 2)
        self.assertEqual(cache.get(1), -1)
        self.assertEqual(cache.get(2), 2)

    def test_stats_accuracy(self):
        cache = LRUCache(2)
        cache.put(1, 1)
        cache.get(1)
        cache.get(2)
        cache.put(2, 2)
        cache.put(3, 3)
        s = cache.stats()
        self.assertEqual(s["hits"],      1)
        self.assertEqual(s["misses"],    1)
        self.assertEqual(s["evictions"], 1)
        self.assertEqual(s["hit_rate"],  "50.0%")

    def test_thread_safe_no_corruption(self):
        cache  = ThreadSafeLRUCache(100)
        errors = []

        def worker(start):
            try:
                for i in range(start, start + 50):
                    cache.put(i, i * 10)
                    cache.get(i)
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=worker, args=(i * 50,))
                   for i in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(errors, [])


class TestEventScheduler(unittest.TestCase):

    def test_no_overlap_adjacent(self):
        events = [(9, 10), (10, 11), (11, 12)]
        self.assertTrue(can_attend_all(events))
        self.assertEqual(min_rooms_required(events), 1)

    def test_overlap_detected(self):
        events = [(9, 11), (10, 12)]
        self.assertFalse(can_attend_all(events))
        self.assertEqual(min_rooms_required(events), 2)

    def test_three_simultaneous(self):
        events = [(9, 12), (9, 12), (9, 12)]
        self.assertFalse(can_attend_all(events))
        self.assertEqual(min_rooms_required(events), 3)

    def test_empty_list(self):
        self.assertTrue(can_attend_all([]))
        self.assertEqual(min_rooms_required([]), 0)
        self.assertEqual(assign_rooms([]), {})

    def test_single_event(self):
        self.assertTrue(can_attend_all([(9, 10)]))
        self.assertEqual(min_rooms_required([(9, 10)]), 1)

    def test_mixed_schedule(self):
        events = [(9, 11), (10, 12), (11, 13)]
        self.assertFalse(can_attend_all(events))
        self.assertEqual(min_rooms_required(events), 2)

    def test_assign_rooms_names(self):
        events = [(9, 11), (10, 12)]
        result = assign_rooms(events)
        self.assertEqual(len(set(result.values())), 2)
        self.assertTrue(all(r.startswith("Room ") for r in result.values()))


# ══════════════════════════════════════════════════════════════════
# QUICK DEMO
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 55)
    print("Problem 1 — LRU Cache")
    print("=" * 55)
    cache = LRUCache(3)
    cache.put(1, 10)
    cache.put(2, 20)
    cache.put(3, 30)
    print(f"get(1) = {cache.get(1)}")
    cache.put(4, 40)                   # evicts key 2
    print(f"get(2) = {cache.get(2)}")  # -1
    print(f"get(3) = {cache.get(3)}")
    print(f"get(4) = {cache.get(4)}")
    print("Stats :", cache.stats())

    print()
    print("=" * 55)
    print("Problem 2 — Event Scheduler")
    print("=" * 55)
    e1 = [(9, 10), (10, 11), (11, 12)]
    e2 = [(9, 11), (10, 12), (11, 13)]

    print(f"Events: {e1}")
    print(f"  can_attend_all     -> {can_attend_all(e1)}")
    print(f"  min_rooms_required -> {min_rooms_required(e1)}")
    print(f"  assign_rooms       -> {assign_rooms(e1)}")

    print(f"\nEvents: {e2}")
    print(f"  can_attend_all     -> {can_attend_all(e2)}")
    print(f"  min_rooms_required -> {min_rooms_required(e2)}")
    print(f"  assign_rooms       -> {assign_rooms(e2)}")

    print()
    print("=" * 55)
    print("Unit Tests")
    print("=" * 55)
    unittest.main(verbosity=2, exit=False)
