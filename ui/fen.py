from enum import Enum


class Side(Enum):
    KINGSIDE = 0b1
    QUEENSIDE = 0b10


class FEN:
    __piece_map = {
        "br": "r",
        "bn": "n",
        "bb": "b",
        "bq": "q",
        "bk": "k",
        "bp": "p",
        "wr": "R",
        "wn": "N",
        "wb": "B",
        "wq": "Q",
        "wk": "K",
        "wp": "P",
        "ee": "1",
    }

    def __init__(self, matrix) -> None:
        self.__matrix = matrix
        self._main_fen = self._fen_from_matrix()
        self._move = "w"
        self._can_w_kingside = True
        self._can_w_queenside = True
        self._can_b_kingside = True
        self._can_b_queenside = True
        self._epilogue = "- 0 1"

    def __eq__(self, fen: str) -> bool:
        return self._main_fen == fen

    def set_white_move(self) -> "FEN":
        self._move = "w"
        return self

    def set_black_move(self) -> "FEN":
        self._move = "b"
        return self

    def set_white_can_castle(self, side_bitmask) -> "FEN":
        self._can_w_kingside = bool(side_bitmask & 1)
        self._can_w_queenside = bool((side_bitmask >> 1) & 1)
        return self

    def set_black_can_castle(self, side_bitmask) -> "FEN":
        self._can_b_kingside = bool(side_bitmask & 1)
        self._can_b_queenside = bool((side_bitmask >> 1) & 1)
        return self

    def get_fen(self) -> str:
        return f"{self._main_fen} {self._move} {self._castles()} {self._epilogue}"

    def _castles(self) -> str:
        res = ""
        if self._can_w_kingside:
            res += "K"
        if self._can_w_queenside:
            res += "Q"
        if self._can_b_kingside:
            res += "k"
        if self._can_b_queenside:
            res += "q"
        return res

    def _fen_from_matrix(self) -> str:

        fen_rows = []
        for row in self.__matrix:
            fen_row = "".join(self.__piece_map[cell] for cell in row)

            compressed_row = ""
            count = 0
            for char in fen_row:
                if char == "1":
                    count += 1
                else:
                    if count:
                        compressed_row += str(count)
                        count = 0
                    compressed_row += char
            if count:
                compressed_row += str(count)

            fen_rows.append(compressed_row)

        return "/".join(fen_rows)
