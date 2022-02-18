import dataclasses
import logging
from typing import Union, Tuple, List, Iterable

from cltl.combot.event.emissor import TextSignalEvent, AnnotationEvent
from cltl.combot.infra.config import ConfigurationManager
from cltl.combot.infra.event import Event, EventBus
from cltl.combot.infra.resource import ResourceManager
from cltl.combot.infra.time_util import timestamp_now
from cltl.combot.infra.topic_worker import TopicWorker
from cltl.face_recognition.api import Face
from emissor.representation.annotation import AnnotationType
from emissor.representation.scenario import TextSignal, Mention, Annotation

from cltl.g2ky.api import GetToKnowYou

logger = logging.getLogger(__name__)


class _FaceCache:
    def __init__(self):
        self._img_id = None
        self._faces = None
        self._ids = None
        self._complete = False
        self._ignored = set()

    @property
    def complete(self):
        return self._complete or (self._img_id is not None
                                  and self._faces is not None
                                  and self._ids is not None)

    def set_image(self, id: str) -> bool:
        if id in self._ignored:
            logger.warning("Image signal event received after annotation: {}", id)
            return False

        if not self.complete:
            self._ignored.add(id)
            return False

        self.__init__()
        self._img_id = id

        return True

    def set_faces(self, mentions: List[Mention]) -> bool:
        image_id = self._get_image_id(mentions)
        if image_id != self._img_id:
            self._ignored.add(id)
            return False

        if len(mentions) == 1 and len(mentions[0].annotation) == 0:
            self._complete = True
        else:
            self._faces = {self._to_segment_key(mention.segment): self._to_annotation_value(mention.annotation)
                           for mention in mentions}
        return True

    def set_ids(self, mentions: Iterable[Mention]) -> bool:
        image_id = self._get_image_id(mentions)
        if image_id != self._img_id:
            self._ignored.add(id)
            return False

        self._ids = {self._to_segment_key(mention.segment): self._to_annotation_value(mention.annotation)
                     for mention in mentions}

        return True

    def _get_image_id(self, mentions: Iterable[Mention]) -> str:
        image_ids = {segment.container_id for mention in mentions for segment in mention.segment}
        if len(image_ids) != 1:
            raise ValueError("Expected exactly on image container")
        image_id = next(iter(image_ids))

        return image_id

    def get_persons(self) -> Iterable[Tuple[str, Face]]:
        if not self.complete:
            raise ValueError("Not all annotations received")

        if self._ids is None:
            return []

        return [(self._ids[key], face) for key, face in self._faces.items()]


    def _to_segment_key(self, segments):
        if len(segments) == 0:
            raise ValueError("Got Mention without segment")

        if len(segments) > 1:
            logger.warning("Got mention with more than one segment: {}", segments)

        return dataclasses.astuple(segments[0])

    def _to_annotation_value(self, annotations):
        if len(annotations) == 0:
            raise ValueError("Got Mention without annotation")

        if len(annotations) > 1:
            logger.warning("Got mention with more than one annotation: {}", annotations)

        return annotations[0].value

class GetToKnowYouService:
    """
    Service used to integrate the component into applications.
    """
    @classmethod
    def from_config(cls, g2ky: GetToKnowYou, event_bus: EventBus, resource_manager: ResourceManager,
                    config_manager: ConfigurationManager):
        config = config_manager.get_config("cltl.g2ky.events")

        return cls(config.get("utterance_topic"), config.get("image_topic"), config.get("face_topic"), config.get("id_topic"),
                   config.get("response_topic"), g2ky, event_bus, resource_manager)

    def __init__(self, utterance_topic: str, image_topic: str, face_topic: str, id_topic: str, response_topic: str,
                 g2ky: GetToKnowYou, event_bus: EventBus, resource_manager: ResourceManager):
        self._g2ky = g2ky

        self._event_bus = event_bus
        self._resource_manager = resource_manager

        self._utterance_topic = utterance_topic
        self._image_topic = image_topic
        self._face_topic = face_topic
        self._id_topic = id_topic
        self._response_topic = response_topic

        self._topic_worker = None
        self._app = None

        self._face_cache = _FaceCache()

    def start(self, timeout=30):
        self._topic_worker = TopicWorker([self._utterance_topic, self._face_topic, self._id_topic], self._event_bus,
                                         provides=[self._response_topic],
                                         resource_manager=self._resource_manager,
                                         scheduled=0.1,
                                         processor=self._process)
        self._topic_worker.start().wait()

    def stop(self):
        if not self._topic_worker:
            pass

        self._topic_worker.stop()
        self._topic_worker.await_stop()
        self._topic_worker = None

    def _process(self, event: Event[Union[TextSignalEvent, AnnotationEvent]]):
        response = None
        if event == None:
            response = self._g2ky.response()
        if event.metadata.topic == self._utterance_topic:
            response = self._g2ky.utterance_detected(event.payload.signal.text)
            id, name = self._g2ky.speaker
            if id and name:
                speaker_event = self._create_speaker_payload(event.payload.signal, id, name)
                self._event_bus.publish(self._speaker_topic, Event.for_payload(speaker_event))
        elif event.metadata.topic == self._image_topic:
            if not self._face_cache.set_image(event.payload.signal.id):
                return
        elif event.metadata.topic == self._id_topic:
            if not self._face_cache.set_ids(event.payload.mentions):
                return
        elif event.metadata.topic == self._face_topic:
            if not self._face_cache.set_ids(event.payload.mentions):
                return

        if self._face_cache.complete:
            response = self._g2ky.persons_detected(self._face_cache.get_persons())

        if response:
            response_payload = self._create_payload(response)
            self._event_bus.publish(self._response_topic, Event.for_payload(response_payload))

    def _create_payload(self, response):
        signal = TextSignal.for_scenario(None, timestamp_now(), timestamp_now(), None, response)

        return TextSignalEvent.create(signal)

    def _create_speaker_payload(self, signal: TextSignal, id: str, name: str):
        speaker_annotation = Annotation(AnnotationType.UTTERANCE, (id, name), self.__name__, timestamp_now())
        return Mention([signal.ruler], [speaker_annotation])