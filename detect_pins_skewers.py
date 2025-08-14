#!/usr/bin/env python3
"""
Chess Pins and Skewers Detector
Analyzes chess games from PGN files to detect tactical motifs (pins and skewers)
"""

import chess
import chess.engine
import chess.pgn
import json
import io
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

@dataclass
class TacticalEvent:
    """Represents a tactical event (pin or skewer)"""
    move_number: int
    tactic: str  # "pin" or "skewer"
    piece: str
    target: str
    square_from: str
    square_to: str
    description: str

class ChessTacticsDetector:
    """Main class for detecting pins and skewers in chess games"""

    def __init__(self, stockfish_path: str = "stockfish"):
        """
        Initialize the detector with Stockfish engine

        Args:
            stockfish_path: Path to Stockfish executable
        """
        self.stockfish_path = stockfish_path
        self.piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 100
        }

    def get_piece_name(self, piece: chess.Piece) -> str:
        """Convert chess piece to string name"""
        names = {
            chess.PAWN: "pawn",
            chess.KNIGHT: "knight",
            chess.BISHOP: "bishop",
            chess.ROOK: "rook",
            chess.QUEEN: "queen",
            chess.KING: "king"
        }
        return names.get(piece.piece_type, "unknown")

    def detect_pins(self, board: chess.Board, color: chess.Color) -> List[Dict]:
        """
        Detect all pins on the board for a given color using python-chess built-in methods

        Args:
            board: Current chess board position
            color: Color to check for pins (True for white, False for black)

        Returns:
            List of pin information dictionaries
        """
        pins = []
        enemy_color = not color
        
        # Get all squares where enemy pieces are pinned
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == enemy_color:
                # Check if this piece is pinned
                if board.is_pinned(enemy_color, square):
                    # Find the pinning piece
                    pinning_square = self.find_pinning_piece(board, square, color)
                    if pinning_square is not None:
                        pins.append({
                            'attacking_piece': self.get_piece_name(board.piece_at(pinning_square)),
                            'attacking_square': chess.square_name(pinning_square),
                            'pinned_piece': self.get_piece_name(piece),
                            'pinned_square': chess.square_name(square),
                            'target': 'king',
                            'target_square': chess.square_name(board.king(enemy_color))
                        })
        
        return pins

    def detect_skewers(self, board: chess.Board, color: chess.Color) -> List[Dict]:
        """
        Detect all skewers on the board for a given color

        Args:
            board: Current chess board position
            color: Color to check for skewers

        Returns:
            List of skewer information dictionaries
        """
        skewers = []
        sliding_pieces = [chess.ROOK, chess.BISHOP, chess.QUEEN]

        for piece_type in sliding_pieces:
            pieces = board.pieces(piece_type, color)

            for piece_square in pieces:
                directions = self.get_attack_directions(piece_type)

                for direction in directions:
                    skewer_info = self.check_skewer_in_direction(
                        board, piece_square, direction, color
                    )

                    if skewer_info:
                        skewers.append(skewer_info)

        return skewers

    def get_attack_directions(self, piece_type: int) -> List[Tuple[int, int]]:
        """Get direction vectors for sliding pieces"""
        if piece_type == chess.ROOK:
            return [(0, 1), (0, -1), (1, 0), (-1, 0)]
        elif piece_type == chess.BISHOP:
            return [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        elif piece_type == chess.QUEEN:
            return [(0, 1), (0, -1), (1, 0), (-1, 0),
                   (1, 1), (1, -1), (-1, 1), (-1, -1)]
        return []

    def check_pin_in_direction(self, board: chess.Board, piece_square: int,
                              king_square: int, direction: Tuple[int, int],
                              color: chess.Color) -> Tuple[Optional[int], Optional[Dict]]:
        """Check for a pin in a specific direction"""
        dx, dy = direction
        current_square = piece_square
        pieces_in_line = []

        # Move in direction until edge of board
        while True:
            file = chess.square_file(current_square) + dx
            rank = chess.square_rank(current_square) + dy

            if not (0 <= file <= 7 and 0 <= rank <= 7):
                break

            current_square = chess.square(file, rank)
            piece = board.piece_at(current_square)

            if piece:
                pieces_in_line.append((current_square, piece))

        # Check if we have exactly one enemy piece between our piece and enemy king
        enemy_pieces = [p for p in pieces_in_line if p[1].color != color]

        if len(enemy_pieces) == 1 and enemy_pieces[0][0] != king_square:
            # Check if king is in line after the enemy piece
            enemy_square = enemy_pieces[0][0]
            remaining_squares = [p[0] for p in pieces_in_line if
                               self.is_beyond_square(piece_square, enemy_square, p[0], direction)]

            if king_square in remaining_squares:
                return enemy_square, {
                    'type': 'pin',
                    'attacking_piece': piece_square,
                    'pinned_piece': enemy_square,
                    'target': king_square
                }

        return None, None

    def check_skewer_in_direction(self, board: chess.Board, piece_square: int,
                                 direction: Tuple[int, int], color: chess.Color) -> Optional[Dict]:
        """Check for a skewer in a specific direction"""
        dx, dy = direction
        current_square = piece_square
        pieces_in_line = []

        # Move in direction until edge of board
        while True:
            file = chess.square_file(current_square) + dx
            rank = chess.square_rank(current_square) + dy

            if not (0 <= file <= 7 and 0 <= rank <= 7):
                break

            current_square = chess.square(file, rank)
            piece = board.piece_at(current_square)

            if piece and piece.color != color:
                pieces_in_line.append((current_square, piece))

        # Check for skewer: high-value piece followed by lower-value piece
        if len(pieces_in_line) >= 2:
            first_piece = pieces_in_line[0]
            second_piece = pieces_in_line[1]

            first_value = self.piece_values.get(first_piece[1].piece_type, 0)
            second_value = self.piece_values.get(second_piece[1].piece_type, 0)

            # A skewer is when a high-value piece is in front of a lower-value piece
            # and both are on the same ray from the attacking piece
            if first_value > second_value and first_value >= 5:  # Higher threshold for more realistic skewers
                # Verify there are no pieces between the two pieces
                between_squares = self.get_squares_between(first_piece[0], second_piece[0])
                if not any(board.piece_at(sq) for sq in between_squares):
                    # Additional check: the attacking piece should be able to capture the front piece
                    if board.is_attacked_by(color, first_piece[0]):
                        return {
                            'attacking_piece': self.get_piece_name(board.piece_at(piece_square)),
                            'attacking_square': chess.square_name(piece_square),
                            'front_piece': self.get_piece_name(first_piece[1]),
                            'front_square': chess.square_name(first_piece[0]),
                            'target': self.get_piece_name(second_piece[1]),
                            'target_square': chess.square_name(second_piece[0])
                        }

        return None

    def find_pinning_piece(self, board: chess.Board, pinned_square: int, color: chess.Color) -> Optional[int]:
        """Find the piece that is pinning the given square"""
        enemy_color = not color
        king_square = board.king(enemy_color)
        
        if king_square is None:
            return None
            
        # Check all sliding pieces of the given color
        for piece_type in [chess.ROOK, chess.BISHOP, chess.QUEEN]:
            for square in board.pieces(piece_type, color):
                # Check if this piece can attack the pinned square
                if board.is_attacked_by(color, pinned_square):
                    # Check if removing the pinned piece would allow attack on king
                    temp_board = board.copy()
                    temp_board.remove_piece_at(pinned_square)
                    if temp_board.is_attacked_by(color, king_square):
                        return square
        return None

    def get_squares_between(self, square1: int, square2: int) -> List[int]:
        """Get all squares between two squares (exclusive)"""
        squares = []
        file1, rank1 = chess.square_file(square1), chess.square_rank(square1)
        file2, rank2 = chess.square_file(square2), chess.square_rank(square2)
        
        # Determine direction
        if file1 == file2:  # Same file (vertical)
            start_rank, end_rank = min(rank1, rank2), max(rank1, rank2)
            for rank in range(start_rank + 1, end_rank):
                squares.append(chess.square(file1, rank))
        elif rank1 == rank2:  # Same rank (horizontal)
            start_file, end_file = min(file1, file2), max(file1, file2)
            for file in range(start_file + 1, end_file):
                squares.append(chess.square(file, rank1))
        elif abs(file1 - file2) == abs(rank1 - rank2):  # Diagonal
            file_step = 1 if file2 > file1 else -1
            rank_step = 1 if rank2 > rank1 else -1
            file, rank = file1 + file_step, rank1 + rank_step
            while file != file2 and rank != rank2:
                squares.append(chess.square(file, rank))
                file += file_step
                rank += rank_step
        
        return squares

    def is_beyond_square(self, start: int, middle: int, end: int,
                        direction: Tuple[int, int]) -> bool:
        """Check if 'end' square is beyond 'middle' square from 'start' in given direction"""
        dx, dy = direction

        start_file, start_rank = chess.square_file(start), chess.square_rank(start)
        middle_file, middle_rank = chess.square_file(middle), chess.square_rank(middle)
        end_file, end_rank = chess.square_file(end), chess.square_rank(end)

        # Check if all three squares are collinear in the direction
        if dx == 0:  # Vertical movement
            if start_file != middle_file or middle_file != end_file:
                return False
            if dy > 0:
                return start_rank < middle_rank < end_rank
            else:
                return start_rank > middle_rank > end_rank
        elif dy == 0:  # Horizontal movement
            if start_rank != middle_rank or middle_rank != end_rank:
                return False
            if dx > 0:
                return start_file < middle_file < end_file
            else:
                return start_file > middle_file > end_file
        else:  # Diagonal movement
            file_diff1 = middle_file - start_file
            rank_diff1 = middle_rank - start_rank
            file_diff2 = end_file - middle_file
            rank_diff2 = end_rank - middle_rank

            return (file_diff1 * dx > 0 and rank_diff1 * dy > 0 and
                   file_diff2 * dx > 0 and rank_diff2 * dy > 0)

    def analyze_move(self, board: chess.Board, move: chess.Move,
                    engine: chess.engine.SimpleEngine) -> Dict[str, Any]:
        """
        Analyze a single move for tactical content

        Args:
            board: Board position before the move
            move: The move to analyze
            engine: Stockfish engine instance

        Returns:
            Dictionary containing tactical analysis
        """
        color = board.turn

        # Get position before move
        pins_before = self.detect_pins(board, color)
        skewers_before = self.detect_skewers(board, color)

        # Make the move
        board.push(move)

        # Get position after move
        pins_after = self.detect_pins(board, color)
        skewers_after = self.detect_skewers(board, color)

        # Get engine's best move for comparison
        try:
            result = engine.analyse(board, chess.engine.Limit(depth=10))
            best_move = result['pv'][0] if result.get('pv') else None
        except:
            best_move = None

        # Undo the move
        board.pop()

        # Check if the move created new tactical opportunities
        new_pins = [pin for pin in pins_after if pin not in pins_before]
        new_skewers = [skewer for skewer in skewers_after if skewer not in skewers_before]

        analysis = {
            'executed_pins': len(new_pins),
            'executed_skewers': len(new_skewers),
            'pins_created': new_pins,
            'skewers_created': new_skewers,
            'best_move': str(best_move) if best_move else None,
            'move_played': str(move),
            'pins_before': pins_before,
            'skewers_before': skewers_before,
            'pins_after': pins_after,
            'skewers_after': skewers_after
        }

        return analysis

    def analyze_game(self, game: chess.pgn.Game) -> Dict[str, List[TacticalEvent]]:
        """
        Analyze a complete chess game for pins and skewers

        Args:
            game: PGN game object

        Returns:
            Dictionary with executed, missed, and allowed tactical events
        """
        board = game.board()
        results = {
            'executed': [],
            'missed': [],
            'allowed': []
        }

        try:
            with chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as engine:
                move_number = 1

                for node in game.mainline():
                    move = node.move
                    color = board.turn

                    # Analyze current position
                    analysis = self.analyze_move(board, move, engine)

                    # Check if tactical motifs were executed
                    if analysis['executed_pins'] > 0:
                        for pin in analysis['pins_created']:
                            event = TacticalEvent(
                                move_number=move_number,
                                tactic="pin",
                                piece=pin['attacking_piece'],
                                target=pin['pinned_piece'],
                                square_from=chess.square_name(move.from_square),
                                square_to=chess.square_name(move.to_square),
                                description=f"Pin executed: {pin['attacking_piece']} pins {pin['pinned_piece']} on {pin['pinned_square']}"
                            )
                            results['executed'].append(event)

                    if analysis['executed_skewers'] > 0:
                        for skewer in analysis['skewers_created']:
                            event = TacticalEvent(
                                move_number=move_number,
                                tactic="skewer",
                                piece=skewer['attacking_piece'],
                                target=skewer['target'],
                                square_from=chess.square_name(move.from_square),
                                square_to=chess.square_name(move.to_square),
                                description=f"Skewer executed: {skewer['attacking_piece']} skewers {skewer['front_piece']} and {skewer['target']}"
                            )
                            results['executed'].append(event)

                    # Check for missed opportunities (if engine suggests a better tactical move)
                    if analysis['best_move'] and analysis['best_move'] != str(move):
                        # Check if the best move would create a pin or skewer
                        temp_board = board.copy()
                        best_move = chess.Move.from_uci(analysis['best_move'])
                        temp_board.push(best_move)
                        
                        best_pins = self.detect_pins(temp_board, color)
                        best_skewers = self.detect_skewers(temp_board, color)
                        
                        if len(best_pins) > len(analysis['pins_after']) or len(best_skewers) > len(analysis['skewers_after']):
                            # Player missed a tactical opportunity
                            if len(best_pins) > len(analysis['pins_after']):
                                event = TacticalEvent(
                                    move_number=move_number,
                                    tactic="pin",
                                    piece=self.get_piece_name(board.piece_at(best_move.from_square)),
                                    target="enemy_piece",
                                    square_from=chess.square_name(best_move.from_square),
                                    square_to=chess.square_name(best_move.to_square),
                                    description=f"Missed pin opportunity with {analysis['best_move']}"
                                )
                                results['missed'].append(event)
                            
                            if len(best_skewers) > len(analysis['skewers_after']):
                                event = TacticalEvent(
                                    move_number=move_number,
                                    tactic="skewer",
                                    piece=self.get_piece_name(board.piece_at(best_move.from_square)),
                                    target="enemy_piece",
                                    square_from=chess.square_name(best_move.from_square),
                                    square_to=chess.square_name(best_move.to_square),
                                    description=f"Missed skewer opportunity with {analysis['best_move']}"
                                )
                                results['missed'].append(event)

                    # Check for allowed opportunities (opponent could have executed pin/skewer)
                    # After making the move, check if opponent has tactical opportunities
                    board.push(move)
                    opponent_color = not color
                    
                    opponent_pins = self.detect_pins(board, opponent_color)
                    opponent_skewers = self.detect_skewers(board, opponent_color)
                    
                    if opponent_pins:
                        for pin in opponent_pins:
                            event = TacticalEvent(
                                move_number=move_number,
                                tactic="pin",
                                piece=pin['attacking_piece'],
                                target=pin['pinned_piece'],
                                square_from=chess.square_name(move.from_square),
                                square_to=chess.square_name(move.to_square),
                                description=f"Allowed pin: opponent could pin {pin['pinned_piece']} with {pin['attacking_piece']}"
                            )
                            results['allowed'].append(event)
                    
                    if opponent_skewers:
                        for skewer in opponent_skewers:
                            event = TacticalEvent(
                                move_number=move_number,
                                tactic="skewer",
                                piece=skewer['attacking_piece'],
                                target=skewer['target'],
                                square_from=chess.square_name(move.from_square),
                                square_to=chess.square_name(move.to_square),
                                description=f"Allowed skewer: opponent could skewer {skewer['front_piece']} and {skewer['target']}"
                            )
                            results['allowed'].append(event)

                    move_number += 1

        except Exception as e:
            print(f"Engine error during analysis: {e}")

        return results

    def process_pgn_file(self, pgn_file_path: str, max_games: int = 5) -> Dict[str, Any]:
        """
        Process PGN file and analyze games for tactical content

        Args:
            pgn_file_path: Path to PGN file
            max_games: Maximum number of games to analyze

        Returns:
            Dictionary containing analysis results for all games
        """
        results = {}

        try:
            with open(pgn_file_path, 'r', encoding='utf-8') as pgn_file:
                game_count = 0

                while game_count < max_games:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break

                    print(f"Analyzing Game {game_count + 1}...")

                    # Get game info
                    white = game.headers.get('White', 'Unknown')
                    black = game.headers.get('Black', 'Unknown')
                    date = game.headers.get('Date', 'Unknown')

                    # Analyze the game
                    game_analysis = self.analyze_game(game)

                    # Convert TacticalEvent objects to dictionaries for JSON serialization
                    game_results = {}
                    for category in ['executed', 'missed', 'allowed']:
                        game_results[category] = []
                        for event in game_analysis[category]:
                            game_results[category].append({
                                'move_number': event.move_number,
                                'tactic': event.tactic,
                                'piece': event.piece,
                                'target': event.target,
                                'square_from': event.square_from,
                                'square_to': event.square_to,
                                'description': event.description
                            })

                    results[f"game_{game_count + 1}"] = {
                        'white': white,
                        'black': black,
                        'date': date,
                        'analysis': game_results
                    }

                    game_count += 1

        except FileNotFoundError:
            print(f"Error: PGN file '{pgn_file_path}' not found.")
        except Exception as e:
            print(f"Error processing PGN file: {e}")

        return results

    def print_summary(self, results: Dict[str, Any]) -> None:
        """Print a summary of the analysis results"""
        print("\n" + "="*50)
        print("CHESS TACTICS ANALYSIS SUMMARY")
        print("="*50)

        for game_key, game_data in results.items():
            if 'analysis' in game_data:
                analysis = game_data['analysis']
                print(f"\n{game_key.upper()}:")
                print(f"  White: {game_data.get('white', 'Unknown')}")
                print(f"  Black: {game_data.get('black', 'Unknown')}")

                executed_pins = len([e for e in analysis['executed'] if e['tactic'] == 'pin'])
                executed_skewers = len([e for e in analysis['executed'] if e['tactic'] == 'skewer'])
                missed_pins = len([e for e in analysis['missed'] if e['tactic'] == 'pin'])
                missed_skewers = len([e for e in analysis['missed'] if e['tactic'] == 'skewer'])
                allowed_pins = len([e for e in analysis['allowed'] if e['tactic'] == 'pin'])
                allowed_skewers = len([e for e in analysis['allowed'] if e['tactic'] == 'skewer'])

                print(f"  Executed → {executed_pins} pins, {executed_skewers} skewers")
                print(f"  Missed → {missed_pins} pins, {missed_skewers} skewers")
                print(f"  Allowed → {allowed_pins} pins, {allowed_skewers} skewers")

def main():
    """Main function to run the chess tactics detector"""
    # Initialize detector (adjust Stockfish path as needed)
    detector = ChessTacticsDetector(stockfish_path="./stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe")

    # Process PGN file
    pgn_file = "games.pgn"
    results = detector.process_pgn_file(pgn_file, max_games=5)

    # Print summary
    detector.print_summary(results)

    # Save results to JSON file
    output_file = "tactics_analysis.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to '{output_file}'")
    except Exception as e:
        print(f"Error saving results: {e}")

if __name__ == "__main__":
    main()
