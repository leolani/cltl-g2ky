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
from cltl.combot.infra.groupby_processor import GroupProcessor, Group, GroupByProcessor

logger = logging.getLogger(__name__)


class FaceGroup(Group):
    def __init__(self, image_id, face_topic, id_topic):
        super().__init__()
        self._face_topic = face_topic
        self._id_topic = id_topic

        self._img_id = image_id
        self._faces = None
        self._ids = None

    @property
    def key(self) -> str:
        return self._img_id

    @property
    def complete(self) -> bool:
        return self._faces is not None and self._ids is not None

    def get_persons(self) -> Iterable[Tuple[str, Face]]:
        if not self.complete:
            raise ValueError("Not all annotations received")

        if self._ids is None:
            return []

        return [(self._ids[key], face) for key, face in self._faces.items()]

    def add(self, event: Event):
        if event.metadata.topic == self._id_topic:
            self._set_ids(event.payload.mentions)
        elif event.metadata.topic == self._face_topic:
            self._set_faces(event.payload.mentions)

    def _set_faces(self, mentions: List[Mention]):
        has_faces = len(mentions) == 1 and mentions[0].annotations and mentions[0].annotations[0].value
        logger.debug("Received %sface for image %s", "" if has_faces else "no ", self._img_id)

        if not has_faces:
            self._faces = {}
            self._ids = []
        else:
            self._faces = {self._to_segment_key(mention.segment): self._to_annotation_value(mention.annotations)
                           for mention in mentions}

    def _set_ids(self, mentions: Iterable[Mention]):
        logger.debug("Received ids for image %s", self._img_id)

        self._ids = {self._to_segment_key(mention.segment): self._to_annotation_value(mention.annotations)
                     for mention in mentions}

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


class GetToKnowYouService(GroupProcessor):
    """
    Service used to integrate the component into applications.
    """
    @classmethod
    def from_config(cls, g2ky: GetToKnowYou, event_bus: EventBus, resource_manager: ResourceManager,
                    config_manager: ConfigurationManager):
        config = config_manager.get_config("cltl.g2ky.events")

        intention_topic = config.get("intention_topic") if "intention_topic" in config else None
        intentions = config.get("intentions", multi=True) if "intentions" in config else []

        return cls(config.get("utterance_topic"), config.get("image_topic"), config.get("face_topic"),
                   config.get("id_topic"), config.get("response_topic"), intention_topic, intentions,
                   g2ky, event_bus, resource_manager)

    def __init__(self, utterance_topic: str, image_topic: str, face_topic: str, id_topic: str, response_topic: str,
                 intention_topic: str, intentions: List[str],
                 g2ky: GetToKnowYou, event_bus: EventBus, resource_manager: ResourceManager):
        self._g2ky = g2ky

        self._event_bus = event_bus
        self._resource_manager = resource_manager

        self._utterance_topic = utterance_topic
        self._image_topic = image_topic
        self._face_topic = face_topic
        self._id_topic = id_topic
        self._response_topic = response_topic
        self._intention_topic = intention_topic
        self._intentions = intentions

        self._topic_worker = None
        self._app = None

        self._face_processor = GroupByProcessor(self)

    def start(self, timeout=30):
        self._topic_worker = TopicWorker([self._utterance_topic, self._image_topic, self._face_topic, self._id_topic],
                                         self._event_bus,
                                         provides=[self._response_topic],
                                         resource_manager=self._resource_manager,
                                         intention_topic=self._intention_topic, intentions=self._intentions,
                                         scheduled=0.1,
                                         processor=self._process,
                                         name=self.__class__.__name__)
        self._topic_worker.start().wait()

        # TODO for now start the intention here
        if self._intentions:
            self._event_bus.publish(self._intention_topic, self._intentions[0])

    def stop(self):
        if not self._topic_worker:
            pass

        self._topic_worker.stop()
        self._topic_worker.await_stop()
        self._topic_worker = None

    def _process(self, event: Event[Union[TextSignalEvent, AnnotationEvent]]):
        response = None
        if event is None:
            response = self._g2ky.response()
        elif event.metadata.topic == self._utterance_topic:
            response = self._g2ky.utterance_detected(event.payload.signal.text)
            id, name = self._g2ky.speaker
            # TODO
            # if id and name:
            #     speaker_event = self._create_speaker_payload(event.payload.signal, id, name)
            #     self._event_bus.publish(self._response_topic, Event.for_payload(speaker_event))
        elif event.metadata.topic in [self._image_topic, self._id_topic, self._face_topic]:
            self._face_processor.process(event)

        if response:
            response_payload = self._create_payload(response)
            self._event_bus.publish(self._response_topic, Event.for_payload(response_payload))

    def _create_payload(self, response):
        signal = TextSignal.for_scenario(None, timestamp_now(), timestamp_now(), None, response)

        return TextSignalEvent.create(signal)

    def _create_speaker_payload(self, signal: TextSignal, id: str, name: str):
        speaker_annotation = Annotation(AnnotationType.UTTERANCE, (id, name), __name__, timestamp_now())
        return Mention([signal.ruler], [speaker_annotation])

    def new_group(self, image_id) -> FaceGroup:
        return FaceGroup(image_id, self._face_topic, self._id_topic)

    def process_group(self, group: FaceGroup):
        logger.debug("Processing faces for image %s", group.key)
        response = self._g2ky.persons_detected(group.get_persons())
        if response:
            response_payload = self._create_payload(response)
            self._event_bus.publish(self._response_topic, Event.for_payload(response_payload))

    def get_key(self, event: Event) -> str:
        if event.metadata.topic == self._image_topic:
            return event.payload.signal.id
        elif event.metadata.topic in [self._id_topic, self._face_topic]:
            return self._get_image_id(event.payload.mentions)

    def _get_image_id(self, mentions: Iterable[Mention]) -> str:
        image_ids = {segment.container_id
                     for mention in mentions
                     for segment in mention.segment}

        if len(image_ids) != 1:
            raise ValueError("Expected exactly on image container, was %s", len(image_ids))

        return next(iter(image_ids))
