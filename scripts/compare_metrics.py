"""Compare deux artefacts de métriques avec une petite tolérance numérique."""
import json
import math
import pathlib
import sys


def _compare(expected, actual, path: str = "root", tolerance: float = 1e-4) -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        errors = []
        if expected.keys() != actual.keys():
            errors.append(f"{path}: clés différentes")
            return errors
        for key in expected:
            errors.extend(_compare(expected[key], actual[key], f"{path}.{key}", tolerance))
        return errors
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{path}: tailles différentes"]
        errors = []
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual, strict=True)):
            errors.extend(
                _compare(expected_item, actual_item, f"{path}[{index}]", tolerance)
            )
        return errors
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not math.isclose(expected, actual, rel_tol=tolerance, abs_tol=tolerance):
            return [f"{path}: attendu {expected}, obtenu {actual}"]
        return []
    return [] if expected == actual else [f"{path}: attendu {expected!r}, obtenu {actual!r}"]


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: compare_metrics.py EXPECTED.json ACTUAL.json")
    expected = json.loads(pathlib.Path(sys.argv[1]).read_text())
    actual = json.loads(pathlib.Path(sys.argv[2]).read_text())
    errors = _compare(expected, actual)
    if errors:
        raise SystemExit("Métriques non reproductibles:\n" + "\n".join(errors))


if __name__ == "__main__":
    main()
