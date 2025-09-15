import hashlib

# A simple color palette for fallbacks, can be extended
_COLOR_PALETTE = [
    "#ff6961",
    "#ffb480",
    "#f8f38d",
    "#42d6a4",
    "#08cad1",
    "#59adf6",
    "#9d94ff",
    "#c780e8",
]


def _get_color(text: str) -> str:
    """
    Generates a consistent, pseudo-random color for a given entity type string.

    This ensures that the same entity type always gets the same color within a
    visualization and across different visualizations.

    Args:
        text: The string to generate a color for (e.g., "PERSON").

    Returns:
        A hex color string (e.g., "#ff6961").
    """
    # Use a hash to get a consistent color index from the palette
    hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
    color_index = hash_val % len(_COLOR_PALETTE)
    return _COLOR_PALETTE[color_index]


def render_biores_html(tagged_tokens: list[tuple[str, str]]) -> str:
    """
    Renders BIOSES-tagged tokens as an HTML string for visualization.

    This function generates a self-contained HTML document with styled spans
    representing the extracted entities.

    Args:
        tagged_tokens: A list of (token, BIOSES tag) tuples.

    Returns:
        A string containing a full HTML document for display.
    """
    html_spans = ""
    colors: dict[str, str] = {}

    for token, tag in tagged_tokens:
        if tag == "O":
            html_spans += f"{token} "
            continue

        try:
            _, entity_type = tag.split("-", 1)
        except ValueError:
            html_spans += f"{token} "
            continue

        if entity_type not in colors:
            colors[entity_type] = _get_color(entity_type)

        color = colors[entity_type]

        style = (
            f"background-color: {color}33; "  # Add alpha for background
            f"border: 1px solid {color}; "
            "padding: 0.2em 0.4em; "
            "margin: 0 0.2em; "
            "line-height: 1; "
            "border-radius: 0.35em;"
        )

        label_style = "font-size: 0.8em; " "font-weight: bold; " "margin-left: 0.4em;"

        html_spans += (
            f'<span style="{style}">'
            f"{token}"
            f'<span style="{label_style}">{entity_type}</span>'
            f"</span> "
        )

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>pyNameEntityRecognition Visualization</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; line-height: 1.6; font-size: 16px; padding: 20px; }}
        </style>
    </head>
    <body>{html_spans.strip()}</body>
    </html>
    """


def display_biores(tagged_tokens: list[tuple[str, str]]):
    """
    Displays the rendered BIOSES HTML in environments like Jupyter notebooks.

    This function is a convenience wrapper around `render_biores_html` that
    utilizes IPython's display capabilities.

    Args:
        tagged_tokens: A list of (token, BIOSES tag) tuples.
    """
    try:
        from IPython.display import HTML, display

        # We only need the inner part of the HTML for display, not the full document
        html_content = render_biores_html(tagged_tokens)
        display(HTML(html_content))
    except ImportError:
        print("IPython is not installed. Cannot display visualization.")
        print("Please run `pip install ipython` in your environment.")
