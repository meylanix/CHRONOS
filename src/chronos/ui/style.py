"""Chart palette and helpers (reference data-viz palette; categorical hues in fixed order)."""

from __future__ import annotations

from typing import Any

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQUENTIAL = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281"]
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
FRESHNESS = {  # icon + label always accompany the colour
    "CURRENT": ("✓", STATUS["good"]),
    "LAGGED": ("⚠", STATUS["warning"]),
    "STALE": ("✖", STATUS["critical"]),
    "NO DATA": ("∅", MUTED),
    "DOCUMENTED POINT ESTIMATES": ("§", INK_SECONDARY),
}


def evidence_color(level: int) -> str:
    """Ordinal blue ramp for evidence levels 0..5 (level 0 nearest the surface still visible)."""
    return SEQUENTIAL[min(max(level, 0), 5) + 2]


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def layout(fig: Any, title: str | None = None, height: int = 380, y_title: str | None = None,
           x_title: str | None = None) -> Any:
    fig.update_layout(
        title={"text": title, "x": 0, "xanchor": "left", "y": 0.98, "yanchor": "top"} if title else None,
        height=height, margin={"l": 10, "r": 10, "t": 48 if title else 16, "b": 10},
        font={"family": "system-ui, -apple-system, Segoe UI, sans-serif", "size": 13},
        legend={"orientation": "h", "yanchor": "top", "y": -0.12, "x": 0},
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, title=x_title, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, gridwidth=1, title=y_title, zeroline=False)
    return fig


class ColorKeeper:
    """Colour follows the entity, never its rank: a slot is assigned on first appearance and kept."""

    def __init__(self, store: dict[str, int]) -> None:
        self.store = store

    def __call__(self, key: str) -> str:
        if key not in self.store:
            used = set(self.store.values())
            free = [i for i in range(len(CATEGORICAL)) if i not in used]
            self.store[key] = free[0] if free else len(self.store) % len(CATEGORICAL)
        return CATEGORICAL[self.store[key]]
