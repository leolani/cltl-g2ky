import dataclasses
import logging
import uuid
from typing import Union, Tuple, List, Iterable

from cltl.combot.event.bdi import DesireEvent
from cltl.combot.event.emissor import TextSignalEvent, AnnotationEvent
from cltl.combot.infra.config import ConfigurationManager
from cltl.combot.infra.event import Event, EventBus
from cltl.combot.infra.groupby_processor import GroupProcessor, Group, GroupByProcessor
from cltl.combot.infra.resource import ResourceManager
from cltl.combot.infra.time_util import timestamp_now
from cltl.combot.infra.topic_worker import TopicWorker
from cltl.face_recognition.api import Face
from cltl.nlp.api import Entity, EntityType
from cltl.vector_id.api import VectorIdentity
from cltl_service.emissordata.client import EmissorDataClient
from emissor.representation.scenario import TextSignal, Mention, Annotation, Signal, MultiIndex

from cltl.g2ky.api import GetToKnowYou

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
    def from_config(cls, g2ky: GetToKnowYou, emissor_client: EmissorDataClient,
                    event_bus: EventBus, resource_manager: ResourceManager,
                    config_manager: ConfigurationManager):
        config = config_manager.get_config("cltl.g2ky.events")

        intention_topic = config.get("topic_intention") if "topic_intention" in config else None
        desire_topic = config.get("topic_desire") if "topic_desire" in config else None
        intentions = config.get("intentions", multi=True) if "intentions" in config else []

        return cls(config.get("topic_utterance"), config.get("topic_image"), config.get("topic_face"),
                   config.get("topic_id"), config.get("topic_response"), config.get("topic_speaker"),
                   intention_topic, desire_topic, intentions,
                   g2ky, emissor_client, event_bus, resource_manager)

    def __init__(self, utterance_topic: str, image_topic: str, face_topic: str, id_topic: str, response_topic: str,
                 speaker_topic: str, intention_topic: str, desire_topic: str, intentions: List[str],
                 g2ky: GetToKnowYou, emissor_client: EmissorDataClient,
                 event_bus: EventBus, resource_manager: ResourceManager):
        self._g2ky = g2ky

        self._emissor_client = emissor_client
        self._event_bus = event_bus
        self._resource_manager = resource_manager

        self._utterance_topic = utterance_topic
        self._image_topic = image_topic
        self._face_topic = face_topic
        self._id_topic = id_topic
        self._response_topic = response_topic
        self._speaker_topic = speaker_topic
        self._intention_topic = intention_topic
        self._desire_topic = desire_topic
        self._intentions = intentions

        self._topic_worker = None
        self._app = None

        self._face_processor = GroupByProcessor(self, max_size=4, buffer_size=16)

    def start(self, timeout=30):
        topics = [self._utterance_topic, self._image_topic, self._face_topic, self._id_topic, self._intention_topic]
        self._topic_worker = TopicWorker(topics, self._event_bus,
                                         provides=[self._speaker_topic, self._response_topic],
                                         resource_manager=self._resource_manager,
                                         intention_topic=self._intention_topic, intentions=self._intentions,
                                         scheduled=1, buffer_size=16,
                                         processor=self._process, name=self.__class__.__name__)
        self._topic_worker.start().wait()

    def stop(self):
        if not self._topic_worker:
            pass

        self._topic_worker.stop()
        self._topic_worker.await_stop()
        self._topic_worker = None

    def _process(self, event: Event[Union[TextSignalEvent, AnnotationEvent]]):
        response = None
        if event is None or self._is_g2ky_intention(event):
            response = self._g2ky.response()
        elif event.metadata.topic == self._utterance_topic:
            response = self._g2ky.utterance_detected(event.payload.signal.text)
        elif event.metadata.topic in [self._image_topic, self._id_topic, self._face_topic]:
            self._face_processor.process(event)

        if response:
            response_payload = self._create_payload(response)
            self._event_bus.publish(self._response_topic, Event.for_payload(response_payload))

        id, name = self._g2ky.speaker
        # TODO remember the right utterance
        if id and name and event and event.metadata.topic in [self._utterance_topic]:
            speaker_event = self._create_speaker_payload(event.payload.signal, id, name)
            self._event_bus.publish(self._speaker_topic, Event.for_payload(speaker_event))
            if self._desire_topic:
                self._event_bus.publish(self._desire_topic, Event.for_payload(DesireEvent(["resolved"])))

            self._g2ky.clear()
            self._topic_worker.clear()

        if event:
            logger.debug("Found %s, %s, response: %s", id, name, response)

    def _is_g2ky_intention(self, event):
        return (event.metadata.topic == self._intention_topic
                and hasattr(event.payload, "intentions")
                and any('g2ky' == intention.label for intention in event.payload.intentions))

    def _create_payload(self, response):
        scenario_id = self._emissor_client.get_current_scenario_id()
        signal = TextSignal.for_scenario(scenario_id, timestamp_now(), timestamp_now(), None, response)

        return TextSignalEvent.for_agent(signal)

    def _create_speaker_payload(self, signal: Signal, id, name):
        offset = signal.ruler
        if hasattr(signal, 'text'):
            segment_start = signal.text.find(name)
            if segment_start >= 0:
                offset = signal.ruler.get_offset(segment_start, segment_start + len(name))

        ts = timestamp_now()

        id_annotations = [Annotation(VectorIdentity.__name__, id, __name__, ts),
                          Annotation(Entity.__name__, Entity(name, EntityType.SPEAKER, offset), __name__, ts)]

        return AnnotationEvent.create([Mention(str(uuid.uuid4()), [offset], id_annotations)])

    def new_group(self, image_id) -> FaceGroup:
        return FaceGroup(image_id, self._face_topic, self._id_topic)

    def process_group(self, group: FaceGroup):
        logger.debug("Processing faces for image %s", group.key)
        response = self._g2ky.persons_detected(group.get_persons())
        if response:
            response_payload = self._create_payload(response)
            self._event_bus.publish(self._response_topic, Event.for_payload(response_payload))

        id, name = self._g2ky.speaker
        logger.debug("Found %s, %s, response: %s", id, name, response)
        if id and name:
            assert len(group._faces) > 0
            face_key = next(iter(group._faces))
            speaker_event = self._create_speaker_payload_for_img(face_key[0], face_key[1], id, name)
            self._event_bus.publish(self._speaker_topic, Event.for_payload(speaker_event))
            if self._desire_topic:
                self._event_bus.publish(self._desire_topic, Event.for_payload(DesireEvent(["resolved"])))

            self._g2ky.clear()
            self._topic_worker.clear()

    def _create_speaker_payload_for_img(self, img_id, bbox, id, name):
        offset = MultiIndex(img_id, bbox)

        ts = timestamp_now()

        id_annotations = [Annotation(VectorIdentity.__name__, id, __name__, ts),
                          Annotation(Entity.__name__, Entity(name, EntityType.SPEAKER, offset), __name__, ts)]

        return AnnotationEvent.create([Mention(str(uuid.uuid4()), [offset], id_annotations)])

    def get_key(self, event: Event) -> str:
        if event.metadata.topic == self._image_topic:
            return event.payload.signal.id
        elif event.metadata.topic in [self._id_topic, self._face_topic]:
            return self._get_image_id(event.payload.mentions)

        return None

    def _get_image_id(self, mentions: Iterable[Mention]) -> str:
        image_ids = {segment.container_id
                     for mention in mentions
                     for segment in mention.segment}

        if len(image_ids) != 1:
            raise ValueError("Expected exactly on image container, was %s", len(image_ids))

        return next(iter(image_ids))
