import unittest

from cltl_service.g2ky.groupby import SizeGroupProcessor, GroupByProcessor


class TestGroupBy(unittest.TestCase):
    def test_grouping(self):
        group_events = []
        grouping = SizeGroupProcessor(3, lambda events: group_events.extend(events))
        groupby = GroupByProcessor(grouping, key=lambda x: x[0])

        self.assertEqual("1", groupby.get_key("10"))

        groupby.process("10")
        self.assertEqual([], group_events)
        groupby.process("11")
        self.assertEqual([], group_events)
        groupby.process("12")
        self.assertEqual(["10", "11", "12"], group_events)

        groupby.process("13")
        self.assertEqual(["10", "11", "12"], group_events)

    def test_multiple_groups_keep_first(self):
        group_events = dict()
        grouping = SizeGroupProcessor(2, lambda events: group_events.update({events[0][0]: events}))
        groupby = GroupByProcessor(grouping, key=lambda x: x[0])

        self.assertEqual("1", groupby.get_key("10"))

        groupby.process("10")
        groupby.process("20")
        groupby.process("21")
        groupby.process("22")
        groupby.process("30")
        self.assertEqual(0, len(group_events))
        groupby.process("11")
        groupby.process("12")
        self.assertEqual(["10", "11"], group_events["1"])

        groupby.process("40")
        groupby.process("41")
        self.assertEqual(["40", "41"], group_events["4"])



