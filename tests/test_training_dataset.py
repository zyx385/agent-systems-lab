from __future__ import annotations

import unittest

from training.redisdoctor.build_dataset import TrainingExample, split_examples


def example(index: int, group: str | None = None) -> TrainingExample:
    return TrainingExample(
        id=f"example-{index}",
        source_group=group or f"group-{index}",
        messages=[
            {"role": "user", "content": f"question {index}"},
            {"role": "assistant", "content": "{}"},
        ],
    )


class DatasetSplitTest(unittest.TestCase):
    def test_exact_independent_split(self) -> None:
        splits = split_examples([example(index) for index in range(820)])
        self.assertEqual({name: len(rows) for name, rows in splits.items()}, {"train": 600, "validation": 100, "test": 120})
        groups = [{row.source_group for row in rows} for rows in splits.values()]
        self.assertFalse(groups[0] & groups[1])
        self.assertFalse(groups[0] & groups[2])

    def test_rejects_wrong_total(self) -> None:
        with self.assertRaises(ValueError):
            split_examples([example(1)])


if __name__ == "__main__":
    unittest.main()
