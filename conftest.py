"""pytest setup for the whole repo (gui/*_test.py).

With JMLR_TEST_FIXTURE=1 the tests run against a small synthetic index (tests/fixture.py) built once into a
temp folder, instead of the real one; this must happen before `common` is imported, which is why it lives
here (conftest.py is loaded before any test module). Without the variable nothing changes.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from tests.fixture import use_fixture  # noqa: E402

use_fixture()
