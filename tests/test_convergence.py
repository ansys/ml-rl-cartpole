def test_get_details(tb):
    l_ = tb.get("my_test")
    assert l_ == list(range(1, 6))


def test_stdout(tb):
    assert tb.cell_output_text(1) == "helloooo"


def test_reaching_end(tb):
    print("asdf")
    assert tb.cell_output_text() == "Final theta angle:"
