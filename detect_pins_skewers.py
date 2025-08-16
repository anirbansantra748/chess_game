#!/usr/bin/env python3
"""
Chess Pins & Skewers Detector — Eval‑Gated (Ultra‑Debug)

What’s new in this diagnostic build:
- **Ultra‑verbose debug logging**: toggle with `--debug`. Prints *every step* of
  evaluation/engine calls, PV handling, tactic detection counts, and node/ply.
- **Version‑agnostic engine.analyse handling**: some python‑chess versions return
  a **list** when `multipv` is set (even `multipv=1`). This previously caused
  `TypeError: list indices must be integers or slices, not str` when indexing
  like `info["score"]`. We now normalize infos so both list and dict forms work.
- **Safer best‑move retrieval**: prefer PV[0] from analyze; fallback to
  `engine.play()` if PV is missing.
- **Detailed try/except with context** around *every* engine call and key step.

Run example (Windows path shown):
  python detect_pins_skewers.py \
    --pgn games.pgn \
    --stockfish ./stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe \
    --depth 12 --lookahead 6 --threshold 120 --max_games 5 --out tactics_analysis.json \
    --debug
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import chess
import chess.engine
import chess.pgn


# ------------------------------ Models ------------------------------

@dataclass
class TacticalEvent:
    move_number: int
    ply: int
    side: str  # "white" | "black"
    tactic: str  # "pin" | "skewer"
    piece: str
    target: str
    square_from: str
    square_to: str
    description: str


# --------------------------- Detector Core --------------------------

class ChessTacticsDetector:
    def __init__(
        self,
        stockfish_path: str = "stockfish",
        depth_eval: int = 12,
        lookahead_plies: int = 6,
        threshold_cp: int = 120,
        debug: bool = False,
    ) -> None:
        self.stockfish_path = stockfish_path
        self.depth_eval = depth_eval
        self.lookahead_plies = lookahead_plies
        self.threshold_cp = threshold_cp
        self.debug = debug
        self.piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 100,
        }

    # -------------------- Utility / Helpers --------------------

    def log(self, msg: str) -> None:
        if self.debug:
            print(msg, flush=True)

    @staticmethod
    def get_piece_name(piece: Optional[chess.Piece]) -> str:
        if not piece:
            return "empty"
        names = {
            chess.PAWN: "pawn",
            chess.KNIGHT: "knight",
            chess.BISHOP: "bishop",
            chess.ROOK: "rook",
            chess.QUEEN: "queen",
            chess.KING: "king",
        }
        return names.get(piece.piece_type, "unknown")

    @staticmethod
    def color_str(color: chess.Color) -> str:
        return "white" if color == chess.WHITE else "black"

    # ---- Normalizers for analyse output (handles dict OR list of dicts)

    def _normalize_info(self, info: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Dict[str, Any]:
        if isinstance(info, dict):
            return info
        # List form: return primary (MultiPV #1)
        if isinstance(info, list):
            return info[0] if info else {}
        return {}

    def _analyse(self, engine: chess.engine.SimpleEngine, board: chess.Board, tag: str) -> Dict[str, Any]:
        self.log(f"[_analyse] Enter tag={tag}, depth={self.depth_eval}, FEN={board.fen()}")
        try:
            raw = engine.analyse(board, chess.engine.Limit(depth=self.depth_eval), multipv=1)
            info = self._normalize_info(raw)
            self.log(f"[_analyse] type(raw)={type(raw).__name__}, keys={list(info.keys())}")
            return info
        except Exception as e:
            self.log(f"[_analyse] EXCEPTION at tag={tag}: {e}")
            raise

    def cp(self, engine: chess.engine.SimpleEngine, board: chess.Board, pov: chess.Color, tag: str) -> int:
        self.log(f"[cp] Start tag={tag}, POV={self.color_str(pov)}")
        info = self._analyse(engine, board, tag=f"{tag}:cp")
        try:
            score_obj = info.get("score")
            if score_obj is None:
                self.log(f"[cp] WARNING: info has no 'score'. info keys={list(info.keys())}")
                return 0
            s = score_obj.pov(pov)
            cp_val = int(s.score(mate_score=100000))
            self.log(f"[cp] OK tag={tag}, cp={cp_val}")
            return cp_val
        except Exception as e:
            self.log(f"[cp] EXCEPTION extracting score at tag={tag}: {e}; info={info}")
            raise

    def best_move(self, engine: chess.engine.SimpleEngine, board: chess.Board, tag: str) -> Optional[chess.Move]:
        self.log(f"[best_move] Start tag={tag}")
        try:
            info = self._analyse(engine, board, tag=f"{tag}:bm")
            pv = info.get("pv")
            if pv:
                mv = pv[0]
                self.log(f"[best_move] From PV: {mv}")
                return mv
        except Exception as e:
            self.log(f"[best_move] analyse path failed: {e}")
        # Fallback to engine.play
        try:
            res = engine.play(board, chess.engine.Limit(depth=self.depth_eval))
            self.log(f"[best_move] From play(): {res.move}")
            return res.move
        except Exception as e:
            self.log(f"[best_move] engine.play() failed: {e}")
            return None

    # -------------------- Pin / Skewer Detectors --------------------

    def detect_pins(self, board: chess.Board, color: chess.Color) -> List[Dict[str, str]]:
        pins: List[Dict[str, str]] = []
        enemy = not color
        king_sq = board.king(enemy)
        if king_sq is None:
            return pins
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.color == enemy and board.is_pinned(enemy, sq):
                pin_sq = self.find_pinning_piece(board, sq, color)
                if pin_sq is not None:
                    pins.append(
                        {
                            "attacking_piece": self.get_piece_name(board.piece_at(pin_sq)),
                            "attacking_square": chess.square_name(pin_sq),
                            "pinned_piece": self.get_piece_name(piece),
                            "pinned_square": chess.square_name(sq),
                            "target": "king",
                            "target_square": chess.square_name(king_sq),
                        }
                    )
        return pins

    def detect_skewers(self, board: chess.Board, color: chess.Color) -> List[Dict[str, str]]:
        skewers: List[Dict[str, str]] = []
        for piece_type in (chess.ROOK, chess.BISHOP, chess.QUEEN):
            for piece_square in board.pieces(piece_type, color):
                for direction in self.get_attack_directions(piece_type):
                    info = self.check_skewer_in_direction(board, piece_square, direction, color)
                    if info:
                        skewers.append(info)
        return skewers

    @staticmethod
    def get_attack_directions(piece_type: int) -> List[Tuple[int, int]]:
        if piece_type == chess.ROOK:
            return [(0, 1), (0, -1), (1, 0), (-1, 0)]
        if piece_type == chess.BISHOP:
            return [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        if piece_type == chess.QUEEN:
            return [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        return []

    def check_skewer_in_direction(
        self, board: chess.Board, piece_square: int, direction: Tuple[int, int], color: chess.Color
    ) -> Optional[Dict[str, str]]:
        dx, dy = direction
        current = piece_square
        ray: List[Tuple[int, Optional[chess.Piece]]] = []
        while True:
            f = chess.square_file(current) + dx
            r = chess.square_rank(current) + dy
            if not (0 <= f <= 7 and 0 <= r <= 7):
                break
            current = chess.square(f, r)
            piece = board.piece_at(current)
            if piece and piece.color != color:
                ray.append((current, piece))
                if len(ray) >= 2:
                    break
            elif piece:
                break
        if len(ray) >= 2:
            (sq1, p1), (sq2, p2) = ray[0], ray[1]
            v1 = self.piece_values.get(p1.piece_type, 0)
            v2 = self.piece_values.get(p2.piece_type, 0)
            if v1 > v2 and v1 >= 5:
                between = self.get_squares_between(sq1, sq2)
                if not any(board.piece_at(sq) for sq in between):
                    if sq1 in board.attacks(piece_square):
                        return {
                            "attacking_piece": self.get_piece_name(board.piece_at(piece_square)),
                            "attacking_square": chess.square_name(piece_square),
                            "front_piece": self.get_piece_name(p1),
                            "front_square": chess.square_name(sq1),
                            "target": self.get_piece_name(p2),
                            "target_square": chess.square_name(sq2),
                        }
        return None

    def find_pinning_piece(self, board: chess.Board, pinned_square: int, color: chess.Color) -> Optional[int]:
        enemy = not color
        king_sq = board.king(enemy)
        if king_sq is None:
            return None
        for piece_type in (chess.ROOK, chess.BISHOP, chess.QUEEN):
            for sq in board.pieces(piece_type, color):
                if pinned_square in board.attacks(sq):
                    temp = board.copy()
                    temp.remove_piece_at(pinned_square)
                    if king_sq in temp.attacks(sq):
                        return sq
        return None

    @staticmethod
    def get_squares_between(square1: int, square2: int) -> List[int]:
        squares: List[int] = []
        f1, r1 = chess.square_file(square1), chess.square_rank(square1)
        f2, r2 = chess.square_file(square2), chess.square_rank(square2)
        if f1 == f2:
            for r in range(min(r1, r2) + 1, max(r1, r2)):
                squares.append(chess.square(f1, r))
        elif r1 == r2:
            for f in range(min(f1, f2) + 1, max(f1, f2)):
                squares.append(chess.square(f, r1))
        elif abs(f1 - f2) == abs(r1 - r2):
            df = 1 if f2 > f1 else -1
            dr = 1 if r2 > r1 else -1
            f, r = f1 + df, r1 + dr
            while f != f2 and r != r2:
                squares.append(chess.square(f, r))
                f += df
                r += dr
        return squares

    # -------------------- Eval‑Gated Analysis --------------------

    def analyse_game(self, game: chess.pgn.Game) -> Dict[str, List[TacticalEvent]]:
        board = game.board()
        results: Dict[str, List[TacticalEvent]] = {"executed": [], "missed": [], "allowed": []}
        try:
            self.log(f"[engine] Launching Stockfish at: {self.stockfish_path}")
            with chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as engine:
                move_number = 1
                ply_index = 1
                for node in game.mainline():
                    move = node.move
                    mover = board.turn
                    self.log("" + "-" * 70)
                    self.log(f"[node] ply={ply_index} move_number={move_number} mover={self.color_str(mover)} move={move.uci()}")
                    self.log(f"[node] FEN before move: {board.fen()}")

                    # Eval before
                    eval_before = self.cp(engine, board, pov=mover, tag=f"ply{ply_index}:before")
                    self.log(f"[eval] before={eval_before}cp")

                    # Best move & delta_best
                    best_mv = self.best_move(engine, board, tag=f"ply{ply_index}")
                    delta_best = None
                    if best_mv is not None:
                        btmp = board.copy()
                        if btmp.is_legal(best_mv):
                            btmp.push(best_mv)
                            eval_after_best = self.cp(engine, btmp, pov=mover, tag=f"ply{ply_index}:after_best")
                            delta_best = eval_after_best - eval_before
                            self.log(f"[best] mv={best_mv.uci()} eval_after_best={eval_after_best}cp delta_best={delta_best}cp")
                        else:
                            self.log(f"[best] WARNING: best_mv not legal in this position: {best_mv}")
                    else:
                        self.log("[best] No best move available (both analyse and play failed)")

                    # Played move delta
                    board.push(move)
                    self.log(f"[node] FEN after move: {board.fen()}")
                    eval_after_played = self.cp(engine, board, pov=mover, tag=f"ply{ply_index}:after_played")
                    delta_played = eval_after_played - eval_before
                    self.log(f"[play] eval_after_played={eval_after_played}cp delta_played={delta_played}cp")

                    # Immediate tactics after the move
                    pins_now = self.detect_pins(board, mover)
                    skews_now = self.detect_skewers(board, mover)
                    self.log(f"[tactics-now] pins={len(pins_now)} skewers={len(skews_now)}")

                    # Lookahead along the actual game for a small window
                    pins_future: List[Dict[str, str]] = []
                    skews_future: List[Dict[str, str]] = []
                    look_node = node
                    look_board = board.copy()
                    for i in range(self.lookahead_plies - 1):
                        if look_node.variations:
                            look_node = look_node.variation(0)
                            look_board.push(look_node.move)
                            pf = self.detect_pins(look_board, mover)
                            sf = self.detect_skewers(look_board, mover)
                            pins_future.extend(pf)
                            skews_future.extend(sf)
                            self.log(f"[lookahead {i+1}] pins+={len(pf)} skewers+={len(sf)} (cum pins={len(pins_future)} skewers={len(skews_future)})")
                        else:
                            break

                    # EXECUTED (eval jump for mover + tactic present now/soon)
                    if delta_played is not None and delta_played >= self.threshold_cp and (
                        pins_now or skews_now or pins_future or skews_future
                    ):
                        self.log("[event] EXECUTED triggered")
                        for p in (pins_now + pins_future):
                            results["executed"].append(
                                TacticalEvent(
                                    move_number=move_number,
                                    ply=ply_index,
                                    side=self.color_str(mover),
                                    tactic="pin",
                                    piece=p["attacking_piece"],
                                    target=p["pinned_piece"],
                                    square_from=chess.square_name(move.from_square),
                                    square_to=chess.square_name(move.to_square),
                                    description=f"Eval +{delta_played}cp gated: {p['attacking_piece']} pins {p['pinned_piece']}",
                                )
                            )
                        for s in (skews_now + skews_future):
                            results["executed"].append(
                                TacticalEvent(
                                    move_number=move_number,
                                    ply=ply_index,
                                    side=self.color_str(mover),
                                    tactic="skewer",
                                    piece=s["attacking_piece"],
                                    target=s["target"],
                                    square_from=chess.square_name(move.from_square),
                                    square_to=chess.square_name(move.to_square),
                                    description=f"Eval +{delta_played}cp gated: {s['attacking_piece']} skewers {s['front_piece']} → {s['target']}",
                                )
                            )

                    # MISSED (best move big gain + tactic appears; played didn't clear gate)
                    if (
                        delta_best is not None
                        and delta_best >= self.threshold_cp
                        and (delta_played is None or delta_played < self.threshold_cp)
                        and best_mv is not None
                    ):
                        self.log("[event] MISSED check path")
                        pos_before = node.parent.board() if node.parent else game.board()
                        tmp_board = pos_before.copy()
                        if tmp_board.is_legal(best_mv):
                            tmp_board.push(best_mv)
                            best_pins = self.detect_pins(tmp_board, mover)
                            best_skews = self.detect_skewers(tmp_board, mover)
                            # extend along PV if available
                            try:
                                info = self._analyse(engine, pos_before, tag=f"ply{ply_index}:missed:pv")
                                pv = info.get("pv", [])
                            except Exception:
                                pv = []
                            if pv and len(pv) > 1:
                                b2 = tmp_board.copy()
                                for mv in pv[1 : self.lookahead_plies]:
                                    if not b2.is_legal(mv):
                                        break
                                    b2.push(mv)
                                    best_pins.extend(self.detect_pins(b2, mover))
                                    best_skews.extend(self.detect_skewers(b2, mover))
                            self.log(f"[missed] best_pins={len(best_pins)} best_skewers={len(best_skews)}")
                            if best_pins or best_skews:
                                if best_pins:
                                    results["missed"].append(
                                        TacticalEvent(
                                            move_number=move_number,
                                            ply=ply_index,
                                            side=self.color_str(mover),
                                            tactic="pin",
                                            piece=self.get_piece_name(tmp_board.piece_at(best_mv.from_square)),
                                            target="enemy_piece",
                                            square_from=chess.square_name(best_mv.from_square),
                                            square_to=chess.square_name(best_mv.to_square),
                                            description=f"Missed +{delta_best}cp pin chance with {best_mv.uci()}",
                                        )
                                    )
                                if best_skews:
                                    results["missed"].append(
                                        TacticalEvent(
                                            move_number=move_number,
                                            ply=ply_index,
                                            side=self.color_str(mover),
                                            tactic="skewer",
                                            piece=self.get_piece_name(tmp_board.piece_at(best_mv.from_square)),
                                            target="enemy_piece",
                                            square_from=chess.square_name(best_mv.from_square),
                                            square_to=chess.square_name(best_mv.to_square),
                                            description=f"Missed +{delta_best}cp skewer chance with {best_mv.uci()}",
                                        )
                                    )

                    # ALLOWED (mover's eval tanks → opponent gains tactic now/soon)
                    if delta_played is not None and delta_played <= -self.threshold_cp:
                        opp = not mover
                        opp_pins = self.detect_pins(board, opp)
                        opp_skews = self.detect_skewers(board, opp)
                        look_node2 = node
                        look_board2 = board.copy()
                        for i in range(self.lookahead_plies - 1):
                            if look_node2.variations:
                                look_node2 = look_node2.variation(0)
                                look_board2.push(look_node2.move)
                                opf = self.detect_pins(look_board2, opp)
                                osf = self.detect_skewers(look_board2, opp)
                                opp_pins.extend(opf)
                                opp_skews.extend(osf)
                                self.log(f"[allowed-look {i+1}] opp pins+={len(opf)} skewers+={len(osf)}")
                            else:
                                break
                        if opp_pins or opp_skews:
                            self.log("[event] ALLOWED triggered")
                            for p in opp_pins:
                                results["allowed"].append(
                                    TacticalEvent(
                                        move_number=move_number,
                                        ply=ply_index,
                                        side=self.color_str(opp),
                                        tactic="pin",
                                        piece=p["attacking_piece"],
                                        target=p["pinned_piece"],
                                        square_from=chess.square_name(move.from_square),
                                        square_to=chess.square_name(move.to_square),
                                        description=f"Allowed (−{abs(delta_played)}cp): opponent can pin {p['pinned_piece']} with {p['attacking_piece']}",
                                    )
                                )
                            for s in opp_skews:
                                results["allowed"].append(
                                    TacticalEvent(
                                        move_number=move_number,
                                        ply=ply_index,
                                        side=self.color_str(opp),
                                        tactic="skewer",
                                        piece=s["attacking_piece"],
                                        target=s["target"],
                                        square_from=chess.square_name(move.from_square),
                                        square_to=chess.square_name(move.to_square),
                                        description=f"Allowed (−{abs(delta_played)}cp): opponent can skewer {s['front_piece']} → {s['target']}",
                                    )
                                )

                    # bookkeeping
                    if mover == chess.BLACK:
                        move_number += 1
                    ply_index += 1
        except Exception as e:
            print(f"Engine error during analysis: {e}")
        return results

    # ------------------------- IO Helpers -------------------------

    def process_pgn_file(self, pgn_file_path: str, max_games: int = 5) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        try:
            with open(pgn_file_path, "r", encoding="utf-8") as pgnf:
                game_count = 0
                while game_count < max_games:
                    game = chess.pgn.read_game(pgnf)
                    if game is None:
                        break
                    white = game.headers.get("White", "Unknown")
                    black = game.headers.get("Black", "Unknown")
                    date = game.headers.get("Date", "Unknown")
                    print(f"Analyzing Game {game_count + 1}… {white} vs {black} ({date})")
                    analysis = self.analyse_game(game)
                    ser = {k: [e.__dict__ for e in v] for k, v in analysis.items()}
                    results[f"game_{game_count + 1}"] = {
                        "white": white,
                        "black": black,
                        "date": date,
                        "analysis": ser,
                    }
                    game_count += 1
        except FileNotFoundError:
            print(f"Error: PGN file '{pgn_file_path}' not found.")
        except Exception as e:
            print(f"Error processing PGN file: {e}")
        return results

    @staticmethod
    def print_summary(results: Dict[str, Any]) -> None:
        print("" + "=" * 50)
        print("CHESS TACTICS ANALYSIS SUMMARY (Eval‑Gated)")
        print("=" * 50)
        for gk, gd in results.items():
            if "analysis" not in gd:
                continue
            a = gd["analysis"]
            ep = len([e for e in a["executed"] if e["tactic"] == "pin"])  # type: ignore
            es = len([e for e in a["executed"] if e["tactic"] == "skewer"])  # type: ignore
            mp = len([e for e in a["missed"] if e["tactic"] == "pin"])  # type: ignore
            ms = len([e for e in a["missed"] if e["tactic"] == "skewer"])  # type: ignore
            ap = len([e for e in a["allowed"] if e["tactic"] == "pin"])  # type: ignore
            assk = len([e for e in a["allowed"] if e["tactic"] == "skewer"])  # type: ignore
            print(f"{gk.upper()}:")
            print(f"  White: {gd.get('white', 'Unknown')}")
            print(f"  Black: {gd.get('black', 'Unknown')}")
            print(f"  Executed → {ep} pins, {es} skewers")
            print(f"  Missed   → {mp} pins, {ms} skewers")
            print(f"  Allowed  → {ap} pins, {assk} skewers")


# ------------------------------ Main -------------------------------

def main():
    parser = argparse.ArgumentParser(description="Eval‑gated chess pins/skewers detector (ultra‑debug)")
    parser.add_argument("--pgn", default="games.pgn", help="Path to PGN file")
    parser.add_argument("--stockfish", default="stockfish", help="Path to Stockfish executable")
    parser.add_argument("--max_games", type=int, default=5, help="Max games to analyze")
    parser.add_argument("--depth", type=int, default=12, help="Search depth for evaluation")
    parser.add_argument("--lookahead", type=int, default=6, help="Lookahead plies for tactic confirmation")
    parser.add_argument("--threshold", type=int, default=120, help="Eval swing (cp) to gate events")
    parser.add_argument("--out", default="tactics_analysis.json", help="Output JSON path")
    parser.add_argument("--debug", action="store_true", help="Print ultra‑verbose debug logs")
    args = parser.parse_args()

    detector = ChessTacticsDetector(
        stockfish_path=args.stockfish,
        depth_eval=args.depth,
        lookahead_plies=args.lookahead,
        threshold_cp=args.threshold,
        debug=args.debug,
    )
    results = detector.process_pgn_file(args.pgn, max_games=args.max_games)
    detector.print_summary(results)
    try:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Detailed results saved to '{args.out}'")
    except Exception as e:
        print(f"Error saving results: {e}")


if __name__ == "__main__":
    main()
