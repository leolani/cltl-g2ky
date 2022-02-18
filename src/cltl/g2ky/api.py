import abc
from typing import Optional, Collection, Tuple, Iterable

from cltl.face_recognition.api import Face

class GetToKnowYou(abc.ABC):
    """
    Abstract class representing the interface of the component.
    """
    def utterance_detected(self, utterance: str) -> Optional[str]:
        raise NotImplementedError()

    def persons_detected(self, persons: Iterable[Tuple[str, Face]]) -> Optional[str]:
        raise NotImplementedError()

    def response(self) -> Optional[str]:
        raise NotImplementedError()

    @property
    def speaker(self) -> Tuple[str, str]:
        raise NotImplementedError()