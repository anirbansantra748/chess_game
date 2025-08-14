#!/usr/bin/env python3
"""
Test script to verify Stockfish is working correctly
"""

import chess
import chess.engine

def test_stockfish():
    """Test if Stockfish engine is accessible and working"""

    # Try different possible paths
    possible_paths = [
        "./stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe",  # Correct path
        "./stockfish.exe",  # If you copied it to main folder
        "./stockfish-windows-x86-64-avx2/stockfish.exe",  # In subfolder
    ]

    for path in possible_paths:
        print(f"Testing path: {path}")
        try:
            with chess.engine.SimpleEngine.popen_uci(path) as engine:
                # Create a simple position
                board = chess.Board()

                # Get engine analysis
                result = engine.analyse(board, chess.engine.Limit(time=1.0))
                best_move = result.get('pv', [None])[0]

                print(f"✅ SUCCESS! Stockfish is working with path: {path}")
                print(f"   Best move from starting position: {best_move}")
                print(f"   Engine evaluation: {result.get('score', 'N/A')}")
                return path

        except FileNotFoundError:
            print(f"❌ File not found: {path}")
            continue
        except Exception as e:
            print(f"❌ Error with {path}: {e}")
            continue

    print("❌ Could not find working Stockfish executable!")
    print("Please check:")
    print("1. Stockfish is extracted properly")
    print("2. The executable file exists")
    print("3. The file path is correct")
    return None

if __name__ == "__main__":
    print("Testing Stockfish setup...")
    print("="*40)
    working_path = test_stockfish()

    if working_path:
        print(f"\n🎉 Use this path in your main program: {working_path}")
    else:
        print("\n❌ Please fix Stockfish setup before running the main program.")
