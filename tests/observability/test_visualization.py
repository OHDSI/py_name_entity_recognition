from unittest.mock import MagicMock, patch

from py_name_entity_recognition.observability.visualization import (
    _get_color,
    display_biores,
    render_biores_html,
)


def test_get_color():
    """Tests that _get_color returns a valid and consistent color."""
    color1 = _get_color("PERSON")
    color2 = _get_color("PERSON")
    color3 = _get_color("LOCATION")

    assert color1 == color2
    assert color1 != color3
    assert color1.startswith("#")
    assert len(color1) == 7


def test_render_biores_html_simple():
    """Tests basic HTML rendering for BIOS-tagged tokens."""
    tagged_tokens = [
        ("John", "S-PERSON"),
        ("works", "O"),
        ("at", "O"),
        ("Google", "S-ORG"),
    ]
    html = render_biores_html(tagged_tokens)

    assert "John" in html
    assert "works" in html
    assert "Google" in html
    assert "PERSON" in html
    assert "ORG" in html
    assert 'style="background-color:' in html


def test_render_biores_html_empty():
    """Tests rendering with no tokens."""
    html = render_biores_html([])
    assert "<body></body>" in html.replace(" ", "")


def test_render_biores_html_malformed_tag():
    """Tests that a malformed tag is handled gracefully."""
    tagged_tokens = [("test", "MALFORMED")]
    html = render_biores_html(tagged_tokens)
    assert "MALFORMED" not in html
    assert "test" in html


def test_display_biores_with_ipython():
    """Tests that display_biores calls IPython's display and HTML."""
    mock_display = MagicMock()
    mock_html = MagicMock()
    with patch.dict(
        "sys.modules",
        {
            "IPython": MagicMock(),
            "IPython.display": MagicMock(display=mock_display, HTML=mock_html),
        },
    ):
        tagged_tokens = [("test", "S-ENTITY")]
        display_biores(tagged_tokens)
        mock_html.assert_called_once()
        mock_display.assert_called_once()


def test_display_biores_without_ipython(capsys):
    """Tests the fallback behavior when IPython is not installed."""
    with patch.dict("sys.modules", {"IPython": None}):
        tagged_tokens = [("test", "S-ENTITY")]
        display_biores(tagged_tokens)
        captured = capsys.readouterr()
        assert "IPython is not installed" in captured.out
