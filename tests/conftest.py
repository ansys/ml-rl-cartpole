import pytest
from testbook import testbook


@pytest.fixture(scope="module")
def tb():
    with testbook("./notebook.ipynb", execute=True, timeout=9999) as tb:
        yield tb
