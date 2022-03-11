import enum
import logging
from collections import Counter
from typing import Optional, Collection, Tuple, Mapping, Iterable

from cltl.face_recognition.api import Face

from cltl.g2ky.api import GetToKnowYou

logger = logging.getLogger(__name__)


class MemoryGetToKnowYou(GetToKnowYou):
    class State(enum.Enum):
        START = 1
        DETECTED = 2
        GAZE = 3
        QUERY = 4
        CONFIRM = 5
        KNOWN = 6

    def __init__(self, friends: Mapping[str, str] = None):
        self._friends = dict(friends) if friends else dict()
        self._id = None
        self._name = None
        self._faces = []
        self._state = self.State.START

    @property
    def speaker(self) -> Tuple[str, str]:
        return self._id, self._name

    @property
    def state(self):
        return self._state

    def utterance_detected(self, utterance: str) -> Optional[str]:
        logger.debug("Received utterance %s in state %s", utterance, self._state.name)

        if self._state == self.State.START:
            return "Hi, I can't see you.."
        elif self._state == self.State.GAZE:
            return "One more second, stranger, I'm memorizing your face."
        if self._state == self.State.QUERY:
            self._name = " ".join([foo.title() for foo in utterance.strip().split()])
            self._state = self.State.CONFIRM
            return f"So your name is {self._name}?"
        if self._state == self.State.CONFIRM:
            if "yes" in utterance.strip().lower():
                self._state = self.State.KNOWN
                self._friends[self._id] = self._name
                return f"Nice to meet you, {self._name}!"
            else:
                self._state = self.State.QUERY
                return "Can you please repeat and only say your name!"

        return None

    def persons_detected(self, persons: Iterable[Tuple[str, Face]]) -> Optional[str]:
        if self._state == self.State.KNOWN:
            logger.debug("Received persons in state %s", self._state.name)
            if self._id not in list(zip(*persons))[0]:
                self._state = self.State.START
                return self.persons_detected(persons)

        persons = list(persons)
        logger.debug("Received %s persons in state %s", len(persons), self._state.name)

        if len(persons) == 0:
            return "Hi, anyone there? I can't see you.."

        if len(persons) > 1:
            return "Hi there! Apologizes, but I will only talk to one of you at a time.."

        identifier, face = next(iter(persons))
        if self._state == self.State.START and identifier in self._friends:
            self._state = self.State.KNOWN
            return f"Nice to meet you again {self._friends[identifier]}!"
        elif self._state == self.State.START:
            self._state = self.State.GAZE
            return f"Hi Stranger! We haven't met, let me look at your face!"
        elif self._state == self.State.GAZE:
            self._faces.append((identifier, face))
            ids = list(zip(*self._faces))[0]
            self._id = Counter(ids).most_common()[0][0]
            if len(set(ids)) > 1:
                logger.debug("Filter multiple faces for %s", self._id)
                self._faces = [(id, face) for id, face in self._faces if id == self._id]

            if len(self._faces) == 5:
                logger.debug("Memorized face for id %s", self._id)
                self._state = self.State.QUERY
                return f"What is your name, stranger?"

        return None

    def response(self) -> Optional[str]:
        # if self._state == self.State.START:
        #     return "Hi, anyone there? I can't see you.."
        return None