import pytest
from testbook import testbook


@pytest.fixture(scope="module")
def tb():
    with testbook("./notebook.ipynb", execute=True) as tb:
        yield tb
