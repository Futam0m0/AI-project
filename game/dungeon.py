"""
Dungeon grid for the D&D dungeon expedition simulator.

Beginner-friendly 2D grid.  The dungeon is a list of lists of single chars:

    grid[row][col] in {'#', '.', 'W', 'H', 'S', 'G', 'R', 'B', 'E'}

Legend:
    # = Wall (blocks movement)
    . = Empty floor

Party (autonomous):
    W = Warrior, H = Healer, S = Scout
Enemies:
    G = Goblin,  R = Redbrand Ruffian,  B = Bugbear
    E = Exit (reaching this wins)

Only '.' cells can be walked on.  Walls block everything.  Each party member and
each enemy occupies EXACTLY one cell.  Two entities can never stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

# Cell symbols we use for empty floor.
FLOOR = "."
WALL = "#"
EXIT = "E"

# Player symbols (in one set for quick membership tests.
PARTY_SYMBOLS = frozenset({"W", "H", "S"})
ENEMY_SYMBOLS = frozenset({"G", "R", "B"})
ENTITY_SYMBOLS = PARTY_SYMBOLS | ENEMY_SYMBOLS
# Symbol → readable names (for human-readable error messages).
SYMBOL_NAMES = {
    "#": "Wall",
    ".": "Floor",
    "E": "Exit",
    "W": "Warrior",
    "H": "Healer",
    "S": "Scout",
    "G": "Goblin",
    "R": "Redbrand Ruffian",
    "B": "Bugbear",
}

# Role → symbol map and reverse.
ROLE_TO_SYMBOL = {
    "Warrior": "W",
    "Healer": "H",
    "Scout": "S",
    "Goblin": "G",
    "Redbrand Ruffian": "R",
    "Bugbear": "B",
}
SYMBOL_TO_ROLE = {v: k for k, v in ROLE_TO_SYMBOL.items()}

# 4-neighbour movement (rook-steps, no diagonals).  Simpler for beginners
# and matches most D&D battle-maps (5-foot grid.
MOVEMENT_STEPS: tuple[tuple[int, int], ...] = (
    (-1, 0),  # up
    (1, 0),   # down
    (0, -1),  # left
    (0, 1),   # right
)


@dataclass(frozen=True)
class Position:
    """Row, col pair.  Immutable →, easy equality (row, col)."""
    row: int
    col: int

    def __iter__(self):
        return iter((self.row, self.col))

    def __add__(self, step: tuple[int, int]) -> "Position":
        dr, dc = step
        return Position(self.row + dr, self.col + dc)

    def manhattan(self, other: "Position") -> int:
        return abs(self.row - other.row) + abs(self.col - other.col)

    def chebyshev(self, other: "Position") -> int:
        """Chebyshev (king-move) distance = max(|dr|, |dc|).

        Used for "melee range" checks: a creature one step in any direction
        (including diagonally) has Chebyshev ≤ 1, within-2 zone ≤ 2, etc.
        """
        return max(abs(self.row - other.row), abs(self.col - other.col))


def chebyshev(a: Position, b: Position) -> int:
    """Module-level convenience alias of Position.chebyshev for readable imports
    ``from game.dungeon import chebyshev`` (matches manhattan usage pattern).
    """
    return a.chebyshev(b)


class Dungeon:
    """
    Fixed-size rectangular dungeon grid.

    Parameters
    ----------
    rows, cols:
        Grid dimensions.  Default 10 cols × 8 rows = manageable, 80 cells which
        fits into a terminal box.
    seed_layout:
        Optional list-of-strings layout.  Each string is one row; length
        must equal cols and len(rows).  When provided, the dungeon starts in
        exactly this layout (useful for tests + fixed-seed reproducibility).
    """

    def __init__(
        self,
        rows: int = 8,
        cols: int = 10,
        seed_layout: list[str] | None = None,
    ) -> None:
        if seed_layout is not None:
            # Validate + use as-is.
            assert all(len(row) == len(seed_layout[0]) for row in seed_layout), \
                "All seed_layout rows must be same length."
            self.rows = len(seed_layout)
            self.cols = len(seed_layout[0])
            self.grid: list[list[str]] = [list(row) for row in seed_layout]
        else:
            self.rows = rows
            self.cols = cols
            # Default: fully floored dungeon with walls around edges + 2 simple
            # wall obstacles.  Simple enough to test movement.
            self.grid = [[FLOOR for _ in range(cols)] for _ in range(rows)]
            # Edge walls.
            for r in range(rows):
                self.grid[r][0] = WALL
                self.grid[r][cols - 1] = WALL
            for c in range(cols):
                self.grid[0][c] = WALL
                self.grid[rows - 1][c] = WALL
            # Internal wall pillars.
            if rows >= 5 and cols >= 6:
                self.grid[2][4] = WALL
                self.grid[3][4] = WALL
                self.grid[4][5] = WALL
                self.grid[5][5] = WALL
                self.grid[5][6] = WALL
            # Exit bottom-right.
            self.grid[rows - 2][cols - 2] = EXIT

    # --------- introspection helpers ----------------------------
    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols

    def is_wall(self, row: int, col: int) -> bool:
        if not self.in_bounds(row, col):
            return True
        return self.grid[row][col] == WALL

    def is_exit(self, row: int, col: int) -> bool:
        return self.in_bounds(row, col) and self.grid[row][col] == EXIT

    def is_passable(self, row: int, col: int) -> bool:
        """True iff (row, col) is in bounds, not a wall, and OCCUPIED check is
        handled separately (the caller passes allow-occupied flag)."""
        if not self.in_bounds(row, col):
            return False
        if self.is_wall(row, col):
            return False
        # Exit is passable (stepping onto it wins the game).
        return True

    def get(self, row: int, col: int) -> str:
        return self.grid[row][col]

    # --------- place / move entities --------------------------------
    def place(self, symbol: str, row: int, col: int) -> bool:
        """Place ``symbol`` at (row, col).  Fails if cell already occupied
        by another entity or wall.  Returns True on success."""
        if not self.is_passable(row, col):
            return False
        if self.grid[row][col] != FLOOR and self.grid[row][col] != EXIT:
            return False
        self.grid[row][col] = symbol
        return True

    def move(self, symbol: str, from_pos: Position, step: tuple[int, int]) -> bool:
        """Move ``symbol`` one ``step`` (dr, dc).  Returns True on success."""
        fr, fc = from_pos
        if self.grid[fr][fc] != symbol:
            return False
        to = from_pos + step
        tr, tc = to
        if not self.is_passable(tr, tc):
            return False
        dest_contents = self.grid[tr][tc]
        # Cannot step onto another entity directly.  We allow stepping onto
        # EXIT or FLOOR only.
        if dest_contents not in (FLOOR, EXIT):
            return False
        # Restore previous cell back to what it was before (floor / exit).
        # NOTE: we never had an entity here before placing.
        self.grid[fr][fc] = FLOOR
        # If destination was the exit, we still show the symbol there so
        # "party wins".  Overwrite EXIT marker?  We store last to FLOOR after
        # moving to make cell empty again.  Actually for simplicity we show symbol
        # on EXIT cell → caller will check "is party now on exit".
        self.grid[tr][tc] = symbol
        return True

    def remove_entity(self, row: int, col: int) -> None:
        """Clear an entity cell → back to FLOOR (used when enemies are defeated)."""
        if self.grid[row][col] in ENTITY_SYMBOLS:
            self.grid[row][col] = FLOOR

    # --------- lookup helpers -------------------------------------
    def find_symbol(self, symbol: str) -> Position | None:
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == symbol:
                    return Position(r, c)
        return None

    def find_all_symbols(self, symbols: Iterable[str]) -> list[tuple[str, Position]]:
        result = []
        for r in range(self.rows):
            for c in range(self.cols):
                cell = self.grid[r][c]
                if cell in symbols:
                    result.append((cell, Position(r, c)))
        return result

    def find_exit(self) -> Position | None:
        return self.find_symbol(EXIT)

    def find_nearest(
        self,
        from_pos: Position,
        symbols: Iterable[str],
    ) -> tuple[str, Position] | None:
        """Return (symbol, Position) of the nearest matching symbol by
        Manhattan distance.  Ties broken by (row, col)."""
        best: tuple[int, int, int, str, Position] | None = None
        for sym, pos in self.find_all_symbols(symbols):
            d = from_pos.manhattan(pos)
            key = (d, pos.row, pos.col)
            if best is None or key < (best[0], best[1], best[2]):
                best = (d, pos.row, pos.col, sym, pos)
        return None if best is None else (best[3], best[4])

    def find_nearest_enemy(self, from_pos: Position) -> tuple[str, Position] | None:
        """Shortcut for the most common lookup in companion AI."""
        return self.find_nearest(from_pos, ENEMY_SYMBOLS)

    def find_nearest_party(self, from_pos: Position) -> tuple[str, Position] | None:
        """Shortcut used by enemy AI."""
        return self.find_nearest(from_pos, PARTY_SYMBOLS)

    # --------- rendering helpers ---------------------------------------
    def to_string(self) -> str:
        """Return a printable string with a newline between rows."""
        return "\n".join("".join(row) for row in self.grid)

    def __str__(self) -> str:  # pragma: no cover - pretty print
        return self.to_string()
