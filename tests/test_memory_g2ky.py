import unittest

import numpy as np
from cltl.backend.api.camera import Bounds
from cltl.face_recognition.api import Face
from emissor.representation.entity import Gender

from cltl.g2ky.memory import MemoryGetToKnowYou

EMPTY_ARRAY = np.empty((0,))

class TestMemoryG2KY(unittest.TestCase):
    def setUp(self) -> None:
        self.g2ky = MemoryGetToKnowYou()

    def test_regular_flow(self):
        self.assertEqual(MemoryGetToKnowYou.State.START, self.g2ky.state)

        response = self.g2ky.persons_detected([("id1", Face(Bounds(0,0,1,1), EMPTY_ARRAY, Gender.FEMALE, 1))])
        self.assertEqual("Hi Stranger! We haven't met, let me look at your face!", response)
        self.assertEqual(MemoryGetToKnowYou.State.GAZE, self.g2ky.state)

        for i in range(9):
            response = self.g2ky.persons_detected([("id1", Face(Bounds(0,0,1,1), EMPTY_ARRAY, Gender.FEMALE, 1))])
            self.assertIsNone(response)
            self.assertEqual(MemoryGetToKnowYou.State.GAZE, self.g2ky.state)

        response = self.g2ky.persons_detected([("id1", Face(Bounds(0,0,1,1), EMPTY_ARRAY, Gender.FEMALE, 1))])
        self.assertEqual("What is your name, stranger?", response)
        self.assertEqual(MemoryGetToKnowYou.State.QUERY, self.g2ky.state)

        response = self.g2ky.persons_detected([("id1", Face(Bounds(0,0,1,1), EMPTY_ARRAY, Gender.FEMALE, 1))])
        self.assertIsNone(response)
        self.assertEqual(MemoryGetToKnowYou.State.QUERY, self.g2ky.state)

        response = self.g2ky.utterance_detected("Thomas")
        self.assertEqual("So your name is Thomas?", response)
        self.assertEqual(MemoryGetToKnowYou.State.CONFIRM, self.g2ky.state)

        response = self.g2ky.utterance_detected("Yes, it is!")
        self.assertEqual("Nice to meet you, Thomas!", response)
        self.assertEqual(MemoryGetToKnowYou.State.KNOWN, self.g2ky.state)

        response = self.g2ky.utterance_detected("Bla")
        self.assertIsNone(response)
        self.assertEqual(MemoryGetToKnowYou.State.KNOWN, self.g2ky.state)

        response = self.g2ky.persons_detected([("id1", Face(Bounds(0,0,1,1), EMPTY_ARRAY, Gender.FEMALE, 1))])
        self.assertIsNone(response)
        self.assertEqual(MemoryGetToKnowYou.State.KNOWN, self.g2ky.state)

        response = self.g2ky.persons_detected([("id2", Face(Bounds(0,0,1,1), EMPTY_ARRAY, Gender.FEMALE, 1))])
        self.assertEqual("Hi Stranger! We haven't met, let me look at your face!", response)
        self.assertEqual(MemoryGetToKnowYou.State.GAZE, self.g2ky.state)

    def test_utterance_while_gaze(self):
        response = self.g2ky.persons_detected([("id1", Face(Bounds(0,0,1,1), EMPTY_ARRAY, Gender.FEMALE, 1))])

        response = self.g2ky.utterance_detected("Hallo!")
        self.assertEqual("One more second, stranger, I'm memorizing your face.", response)
        self.assertEqual(MemoryGetToKnowYou.State.GAZE, self.g2ky.state)

    def test_utterance_before_seen(self):
        self.assertEqual(MemoryGetToKnowYou.State.START, self.g2ky.state)

        response = self.g2ky.utterance_detected("Hallo!")
        self.assertEqual("Hi, I can't see you..", response)
        self.assertEqual(MemoryGetToKnowYou.State.START, self.g2ky.state)

        response = self.g2ky.persons_detected([("id1", Face(Bounds(0,0,1,1), EMPTY_ARRAY, Gender.FEMALE, 1))])
        self.assertEqual("Hi Stranger! We haven't met, let me look at your face!", response)
        self.assertEqual(MemoryGetToKnowYou.State.GAZE, self.g2ky.state)

    def test_name_incorrect(self):
        response = self.g2ky.persons_detected([("id1", Face(Bounds(0,0,1,1), EMPTY_ARRAY, Gender.FEMALE, 1))])
        for i in range(10):
            response = self.g2ky.persons_detected([("id1", Face(Bounds(0,0,1,1), EMPTY_ARRAY, Gender.FEMALE, 1))])
        response = self.g2ky.persons_detected([("id1", Face(Bounds(0,0,1,1), EMPTY_ARRAY, Gender.FEMALE, 1))])
        response = self.g2ky.utterance_detected("Thomass")
        response = self.g2ky.utterance_detected("No, it is Thomas!")

        self.assertEqual("Can you please repeat and only say your name!", response)
        self.assertEqual(MemoryGetToKnowYou.State.QUERY, self.g2ky.state)

        response = self.g2ky.utterance_detected("Thomas")
        self.assertEqual("So your name is Thomas?", response)
        self.assertEqual(MemoryGetToKnowYou.State.CONFIRM, self.g2ky.state)

        response = self.g2ky.utterance_detected("Yes")
        self.assertEqual("Nice to meet you, Thomas!", response)
        self.assertEqual(MemoryGetToKnowYou.State.KNOWN, self.g2ky.state)
