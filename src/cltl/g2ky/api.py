import abc
from typing import Optional, Collection, Tuple

from cltl.face_recognition.api import Face

class GetToKnowYou(abc.ABC):
    """
    Abstract class representing the interface of the component.
    """
    def utterance_detected(self, utterance: str) -> Optional[str]:
        raise NotImplementedError()

    def persons_detected(self, persons: Collection[Tuple[str, Face]]) -> Optional[str]:
        raise NotImplementedError()

    def response(self) -> Optional[str]:
        raise NotImplementedError()

    @property
    def speaker(self) -> Tuple[str, str]:
        raise NotImplementedError()