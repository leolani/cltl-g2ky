import unittest

import numpy as np
from cltl.backend.api.camera import Bounds
from cltl.face_recognition.api import Face
from emissor.representation.entity import Gender

from cltl.g2ky.memory import MemoryGetToKnowYou, ConvState


EMPTY_ARRAY = np.empty((0,))

class TestMemoryG2KY(unittest.TestCase):
    def setUp(self) -> None:
        self.g2ky = MemoryGetToKnowYou()

    def test_regular_flow(self):
        self.assertEqual(ConvState.START, self.g2ky.state.conv_state)

        response = self.g2ky.persons_detected([("id1", Face(EMPTY_ARRAY, Gender.FEMALE, 1))])
        self.assertEqual("Hi Stranger! We haven't met, let me look at your face!", response)
        self.assertEqual(ConvState.GAZE, self.g2ky.state.conv_state)

        for i in range(4):
            response = self.g2ky.persons_detected([("id1", Face(EMPTY_ARRAY, Gender.FEMALE, 1))])
            self.assertIsNone(response)
            self.assertEqual(ConvState.GAZE, self.g2ky.state.conv_state)

        response = self.g2ky.persons_detected([("id1", Face(EMPTY_ARRAY, Gender.FEMALE, 1))])
        self.assertEqual("What is your name, stranger?", response)
        self.assertEqual(ConvState.QUERY, self.g2ky.state.conv_state)

        response = self.g2ky.persons_detected([("id1", Face(EMPTY_ARRAY, Gender.FEMALE, 1))])
        self.assertIsNone(response)
        self.assertEqual(ConvState.QUERY, self.g2ky.state.conv_state)

        response = self.g2ky.utterance_detected("Thomas")
        self.assertEqual("So your name is Thomas?", response)
        self.assertEqual(ConvState.CONFIRM, self.g2ky.state.conv_state)

        response = self.g2ky.utterance_detected("Yes, it is!")
        self.assertEqual("Nice to meet you, Thomas!", response)
        self.assertEqual(ConvState.KNOWN, self.g2ky.state.conv_state)

        response = self.g2ky.utterance_detected("Bla")
        self.assertIsNone(response)
        self.assertEqual(ConvState.KNOWN, self.g2ky.state.conv_state)

        response = self.g2ky.persons_detected([("id1", Face(EMPTY_ARRAY, Gender.FEMALE, 1))])
        self.assertIsNone(response)
        self.assertEqual(ConvState.KNOWN, self.g2ky.state.conv_state)

        # Change person only after two new detections
        for i in range(2):
            response = self.g2ky.persons_detected([("id2", Face(EMPTY_ARRAY, Gender.FEMALE, 1))])
            self.assertIsNone(response)
            self.assertEqual(ConvState.KNOWN, self.g2ky.state.conv_state)

        response = self.g2ky.persons_detected([("id2", Face(EMPTY_ARRAY, Gender.FEMALE, 1))])
        self.assertIsNone(response)
        self.assertEqual(ConvState.START, self.g2ky.state.conv_state)
        response = self.g2ky.persons_detected([("id2", Face(EMPTY_ARRAY, Gender.FEMALE, 1))])
        self.assertEqual("Hi Stranger! We haven't met, let me look at your face!", response)
        self.assertEqual(ConvState.GAZE, self.g2ky.state.conv_state)

    def test_utterance_while_gaze(self):
        response = self.g2ky.persons_detected([("id1", Face(EMPTY_ARRAY, Gender.FEMALE, 1))])

        response = self.g2ky.utterance_detected("Hallo!")
        self.assertEqual("One more second, stranger, I'm memorizing your face.", response)
        self.assertEqual(ConvState.GAZE, self.g2ky.state.conv_state)

    def test_utterance_before_seen(self):
        self.assertEqual(ConvState.START, self.g2ky.state.conv_state)

        response = self.g2ky.utterance_detected("Hallo!")
        self.assertEqual("Hi, I can't see you..", response)
        self.assertEqual(ConvState.START, self.g2ky.state.conv_state)

        response = self.g2ky.persons_detected([("id1", Face(EMPTY_ARRAY, Gender.FEMALE, 1))])
        self.assertEqual("Hi Stranger! We haven't met, let me look at your face!", response)
        self.assertEqual(ConvState.GAZE, self.g2ky.state.conv_state)

    def test_name_incorrect(self):
        response = self.g2ky.persons_detected([("id1", Face(EMPTY_ARRAY, Gender.FEMALE, 1))])
        for i in range(10):
            response = self.g2ky.persons_detected([("id1", Face(EMPTY_ARRAY, Gender.FEMALE, 1))])
        response = self.g2ky.persons_detected([("id1", Face(EMPTY_ARRAY, Gender.FEMALE, 1))])
        response = self.g2ky.utterance_detected("Thomass")
        response = self.g2ky.utterance_detected("No, it is Thomas!")

        self.assertEqual("Can you please repeat and only say your name!", response)
        self.assertEqual(ConvState.QUERY, self.g2ky.state.conv_state)

        response = self.g2ky.utterance_detected("Thomas")
        self.assertEqual("So your name is Thomas?", response)
        self.assertEqual(ConvState.CONFIRM, self.g2ky.state.conv_state)

        response = self.g2ky.utterance_detected("Yes")
        self.assertEqual("Nice to meet you, Thomas!", response)
        self.assertEqual(ConvState.KNOWN, self.g2ky.state.conv_state)
