import dataclasses
import enum
import logging
import uuid
from typing import Optional, Tuple, Iterable, Mapping

from cltl.face_recognition.api import Face

from cltl.g2ky.api import GetToKnowYou

logger = logging.getLogger(__name__)


class ConvState(enum.Enum):
    START = 1
    QUERY = 2
    CONFIRM = 3
    KNOWN = 4

    def transitions(self):
        return self._allowed[self]

    @property
    def _allowed(self):
        return {
            ConvState.START: [ConvState.QUERY],
            ConvState.QUERY: [ConvState.CONFIRM],
            ConvState.CONFIRM: [ConvState.KNOWN, ConvState.QUERY],
            ConvState.KNOWN: [ConvState.START]
        }


@dataclasses.dataclass
class State:
    face_id: Optional[str]
    name: Optional[str]
    conv_state: Optional[ConvState]

    def transition(self, conv_state: ConvState, **kwargs):
        if conv_state not in self.conv_state.transitions():
            raise ValueError(f"Cannot change state from {self.conv_state} to {conv_state}")

        logger.debug("Transition from conversation state %s to %s", self.conv_state, conv_state)

        return self._transition(conv_state, **kwargs)

    def stay(self, **kwargs):
        logger.debug("Reenter conversation state %s to %s", self.conv_state)
        return self._transition(self.conv_state, **kwargs)

    def _transition(self, conv_state: ConvState, **kwargs):
        new_state = vars(State(None, None, None)) if conv_state == ConvState.START else vars(self)
        new_state.update(**kwargs)
        new_state["conv_state"] = conv_state

        return State(**new_state)


class VerbalGetToKnowYou(GetToKnowYou):
    def __init__(self, friends: Mapping[str, str] = None):
        self._friends = {name: id for id, name in friends.items()} if friends else dict()
        self._state = State(None, None, ConvState.START)

    @property
    def speaker(self) -> Tuple[str, str]:
        return (self.state.face_id, self.state.name) if self._state.conv_state == ConvState.KNOWN else (None, None)

    @property
    def state(self) -> State:
        return self._state

    def utterance_detected(self, utterance: str) -> Optional[str]:
        logger.debug("Received utterance %s in state %s", utterance, self.state.conv_state.name)

        response = None
        if self.state.conv_state == ConvState.START:
            response = "Hi, nice to meet you! What is your name?"
            self._state = self.state.transition(ConvState.QUERY)
        elif self.state.conv_state == ConvState.QUERY:
            name = " ".join([foo.title() for foo in utterance.strip().split()])
            response = f"So your name is {name}?"
            if name not in self._friends:
                self._friends[name] = str(uuid.uuid4())
            self._state = self.state.transition(ConvState.CONFIRM, name=name, face_id=self._friends[name])
        elif self.state.conv_state == ConvState.CONFIRM:
            if "yes" in utterance.strip().lower():
                response = f"Nice to meet you, {self.state.name}!"
                self._state = self.state.transition(ConvState.KNOWN)
            else:
                response = "Can you please repeat and only say your name!"
                self._state = self.state.transition(ConvState.QUERY)

        return response

    def persons_detected(self, persons: Iterable[Tuple[str, Face]]) -> Optional[str]:
        pass

    def response(self) -> Optional[str]:
        if self.state.conv_state == ConvState.START:
            self._state = self.state.transition(ConvState.QUERY)

            return "Hi, nice to meet you! What is your name?"

        return None

    def clear(self):
        self._state = self._state.transition(ConvState.START)
