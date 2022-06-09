import dataclasses
import enum
import logging
from collections import Counter

from cltl.face_recognition.api import Face
from typing import Optional, Tuple, Mapping, Iterable, List

from cltl.g2ky.api import GetToKnowYou

logger = logging.getLogger(__name__)


class ConvState(enum.Enum):
    START = 1
    GAZE = 2
    QUERY = 3
    CONFIRM = 4
    KNOWN = 5

    def transitions(self):
        return self._allowed[self]

    @property
    def _allowed(self):
        return {
            ConvState.START: [ConvState.GAZE, ConvState.KNOWN],
            ConvState.GAZE: [ConvState.QUERY, ConvState.START],
            ConvState.QUERY: [ConvState.CONFIRM],
            ConvState.CONFIRM: [ConvState.KNOWN, ConvState.QUERY],
            ConvState.KNOWN: [ConvState.START]
        }


@dataclasses.dataclass
class State:
    face_id: Optional[str]
    name: Optional[str]
    conv_state: Optional[ConvState]
    faces: List
    state_count: int

    def transition(self, conv_state: ConvState, **kwargs):
        if not conv_state in self.conv_state.transitions():
            raise ValueError(f"Cannot change state from {self.conv_state} to {conv_state}")

        logger.debug("Transition from conversation state %s to %s", self.conv_state, conv_state)

        return self._transition(conv_state, state_count=0, **kwargs)

    def stay(self, **kwargs):
        logger.debug("Reenter conversation state %s to %s", self.conv_state, self.state_count + 1)
        return self._transition(self.conv_state, state_count=self.state_count + 1, **kwargs)

    def _transition(self, conv_state: ConvState, **kwargs):
        new_state = vars(State(None, None, None, [], 0)) if conv_state == ConvState.START else vars(self)
        new_state.update(**kwargs)
        new_state["conv_state"] = conv_state

        return State(**new_state)


class MemoryGetToKnowYou(GetToKnowYou):
    def __init__(self, gaze_images: int = 5, friends: Mapping[str, str] = None):
        self._gaze_images = gaze_images
        self._friends = dict(friends) if friends else dict()
        self._state = State(None, None, ConvState.START, [], 0)

    @property
    def speaker(self) -> Tuple[str, str]:
        return self.state.face_id, self.state.name

    @property
    def state(self) -> State:
        return self._state

    def utterance_detected(self, utterance: str) -> Optional[str]:
        logger.debug("Received utterance %s in state %s", utterance, self.state.conv_state.name)

        response = None
        if self.state.conv_state == ConvState.START:
            if self.state.state_count == 0:
                response = "Hi, I can't see you.."
            else:
                response = "Sorry, I still can't see you.."
            self._state = self.state.stay()
        elif self.state.conv_state == ConvState.GAZE:
            response = "One more second, stranger, I'm memorizing your face."
            self._state = self.state.stay()
        elif self.state.conv_state == ConvState.QUERY:
            name = " ".join([foo.title() for foo in utterance.strip().split()])
            response = f"So your name is {name}?"
            self._state = self.state.transition(ConvState.CONFIRM, name= name)
        elif self.state.conv_state == ConvState.CONFIRM:
            if "yes" in utterance.strip().lower():
                self._friends[self.state.face_id] = self.state.name
                response = f"Nice to meet you, {self.state.name}!"
                self._state = self.state.transition(ConvState.KNOWN)
            else:
                response = "Can you please repeat and only say your name!"
                self._state = self.state.transition(ConvState.QUERY)

        return response

    def persons_detected(self, persons: Iterable[Tuple[str, Face]]) -> Optional[str]:
        persons = list(persons)
        logger.debug("Received %s persons in state %s", len(persons), self.state.conv_state)

        response = None
        if len(persons) == 0:
            if self.state.conv_state == ConvState.START:
                response = "Hi, anyone there? I can't see anyone.." if self.state.state_count % 10 == 0 else None
                self._state = self.state.stay()
            elif self.state.state_count % 10 and ConvState.START in self.state.conv_state.transitions():
                self._state = self.state.transition(ConvState.START)
            else:
                self._state = self.state.stay()
        elif len(persons) > 1:
            if self.state.state_count % 3 == 2:
                response = "Hi there! Apologizes, but I will only talk to one of you at a time.."
            self._state = self.state.stay()
        else:
            identifier, face = next(iter(persons))
            if self.state.conv_state == ConvState.KNOWN:
                if identifier != self.state.face_id and self.state.state_count > 2:
                    self._state = self.state.transition(ConvState.START)
                else:
                    self._state = self.state.stay()
            elif self.state.conv_state == ConvState.START:
                if identifier in self._friends:
                    name = self._friends[identifier]
                    response = f"Nice to meet you again {name}!"
                    self._state = self.state.transition(ConvState.KNOWN, face_id=identifier, name=name)
                else:
                    response = f"Hi Stranger! We haven't met, let me look at your face!"
                    self._state = self.state.transition(ConvState.GAZE)
            elif self.state.conv_state == ConvState.GAZE:
                self.state.faces.append((identifier, face))
                if len(self.state.faces) == self._gaze_images:
                    ids = list(zip(*self.state.faces))[0]
                    identifier = Counter(ids).most_common()[0][0]
                    if len(set(ids)) > 1:
                        logger.debug("Filter multiple faces for %s", identifier)
                        faces = [(id, face) for id, face in self.state.faces if id == identifier]
                    else:
                        faces = self.state.faces
                    logger.debug("Memorized face for id %s", identifier)
                    response = f"What is your name, stranger?"
                    self._state = self.state.transition(ConvState.QUERY, face_id=identifier, faces=faces)

        return response

    def response(self) -> Optional[str]:
        return None