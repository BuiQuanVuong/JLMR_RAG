r"""Tests for the Scholar-style query parser and whole-paper matching (real database, no GPU, no API).

  .venv\Scripts\python tests\test_boolq.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tests.fixture import use_fixture  # noqa: E402

use_fixture()                          # JMLR_TEST_FIXTURE=1: synthetic index (must precede importing common)

import boolq  # noqa: E402
from common import connect  # noqa: E402


def labels(q):
    return [c.label() for c in boolq.parse(q)]


# detection: plain questions (including hyphenated words) stay natural-language queries
assert not boolq.is_boolean("kernel two-sample test")
assert not boolq.is_boolean("How do GP bandits handle non-stationary rewards?")
for q in ('"exact phrase"', "a OR b", "kernel -bandit", "intitle:causal", "author:Gretton"):
    assert boolq.is_boolean(q), q

# parsing: phrases, hyphenated terms, OR binding, exclusions, prefixes
assert labels('"privacy-preserving" "knowledge sharing" multimodal medical federated prototypes OR rules') == [
    '"privacy preserving"', '"knowledge sharing"', "multimodal", "medical", "federated", "prototypes OR rules"]
assert labels('federated "differential privacy" -bandit') == ["federated", '"differential privacy"', "-bandit"]
assert labels('intitle:"two-sample" kernel author:Gretton') == ['intitle:"two sample"', "kernel", "author:Gretton"]
assert labels("causal OR causality OR causation discovery") == ["causal OR causality OR causation", "discovery"]
assert labels("OR leading") == ["leading"]                       # a dangling OR is ignored
assert boolq.plain_text(boolq.parse('federated "differential privacy" -bandit author:X')) == \
    "federated differential privacy"

# matching on the real index
con = connect()
r = boolq.match(con, 'intitle:"two-sample" kernel author:Gretton')
assert "v13/gretton12a" in r.strict, r.strict                    # A Kernel Two-Sample Test (2012)
excl = boolq.match(con, 'intitle:"two-sample" kernel author:Gretton -aggregated')
assert excl.strict <= r.strict                                   # an exclusion can only remove papers
r = boolq.match(con, '"privacy-preserving" "knowledge sharing" multimodal medical federated prototypes OR rules')
assert not r.strict and r.partial and r.partial[0][1] == 4       # nothing has all 6; best partial has 4
assert all(len(miss) == 6 - n for _, n, miss in r.partial)
print("All boolean-query checks passed.")
