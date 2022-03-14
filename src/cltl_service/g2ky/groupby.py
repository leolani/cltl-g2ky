import abc
import logging
from collections import OrderedDict

from cltl.combot.infra.event import Event
from cltl.combot.infra.time_util import timestamp_now
from typing import Callable, List

logger = logging.getLogger(__name__)


class Group(abc.ABC):
    def __init__(self):
        self._timestamp = timestamp_now()

    @property
    def timestamp(self) -> int:
        return self._timestamp

    @property
    def complete(self) -> bool:
        raise NotImplementedError()

    def add(self, event: Event):
        raise NotImplementedError()


class GroupProcessor(abc.ABC):
    def new_group(self, key):
        raise NotImplementedError()

    def process_group(self, group):
        raise NotImplementedError()


class GroupByProcessor:
    def __init__(self, group_processor: GroupProcessor, key: Callable[[Event], str] = None,
                 max_size: int = 1):
        self._group_processor = group_processor
        self._groups = OrderedDict()
        self._completed = OrderedDict()
        self._dropped = OrderedDict()
        self._key = key
        self._max_size = max_size
        self._completion_buffer = 10 * self._max_size
        self._timeout = 10_000

    def process(self, event: Event):
        key = self.get_key(event)

        current = timestamp_now()
        expired = [key for key, group in self._groups.items() if current - group.timestamp > self._timeout]
        for key in expired:
            logger.debug("Group %s timed out", key)
            self._groups.popitem(last=False)

        if key in self._completed:
            logger.exception("Received event for completed group %s: %s", key, event)
            return
        if key in self._dropped:
            return

        if len(self._groups) == self._max_size and key not in self._groups:
            self._dropped[key] = None
            return

        if key not in self._groups:
            self._groups[key] = self._group_processor.new_group(key)

        self._groups[key].add(event)

        if self._groups[key].complete:
            self._group_processor.process_group(self._groups[key])
            self._completed[key] = None
            del self._groups[key]

        if len(self._completed) > self._completion_buffer:
            self._completed.popitem(last=False)
        if len(self._dropped) > self._completion_buffer:
            self._dropped.popitem(last=False)

    def get_key(self, event: Event) -> str:
        if self._key:
            return self._key(event)

        raise ValueError("No key function")


class SizeGroup(Group):
    def __init__(self, key, size):
        super().__init__()
        self.size = size
        self.events = []

    @property
    def complete(self):
        return len(self.events) == self.size

    def add(self, event: Event):
        self.events.append(event)


class SizeGroupProcessor(GroupProcessor):
    def __init__(self, size, processor: Callable[[List[Event]], None]):
        self._processor = processor
        self._size = size

    def new_group(self, key):
        return SizeGroup(key, self._size)

    def process_group(self, group):
        self._processor(group.events)

