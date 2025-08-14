# Chess Pins and Skewers Detector

A Python program that analyzes chess games from PGN files to detect tactical motifs (pins and skewers) and classify them as executed, missed, or allowed opportunities.

## Features

- **Pin Detection**: Identifies when pieces are pinned to the king or other valuable pieces
- **Skewer Detection**: Finds skewer tactics where high-value pieces are attacked through lower-value pieces
- **Tactical Analysis**: Compares player moves with Stockfish engine suggestions
- **Classification**: Categorizes tactical opportunities into:
  - **Executed**: Player successfully created or exploited a pin/skewer
  - **Missed**: Player had the chance but chose another move
  - **Allowed**: Opponent could execute a pin/skewer but player's move allowed it

## Requirements

- Python 3.7+
- python-chess library
- Stockfish chess engine

## Installation

1. **Install Python dependencies:**

   ```bash
   pip install python-chess
   ```

2. **Download Stockfish:**

   - The program includes a Windows Stockfish executable
   - For other platforms, download from [Stockfish official website](https://stockfishchess.org/download/)

3. **Verify Stockfish setup:**
   ```bash
   python test_stockfish.py
   ```

## Usage

### Basic Usage

Run the main program to analyze games:

```bash
python detect_pins_skewers.py
```

### Program Structure

- **`detect_pins_skewers.py`**: Main program with tactical detection logic
- **`games.pgn`**: Sample PGN file with 5 chess games
- **`test_stockfish.py`**: Test script to verify Stockfish installation
- **`tactics_analysis.json`**: Output file with detailed analysis results

### Output Format

The program generates a JSON file with the following structure:

```json
{
  "game_1": {
    "white": "Player1",
    "black": "Player2",
    "date": "2024.01.15",
    "analysis": {
      "executed": [
        {
          "move_number": 12,
          "tactic": "pin",
          "piece": "bishop",
          "target": "queen",
          "square_from": "c4",
          "square_to": "d5",
          "description": "Pin executed: bishop pins queen on d5"
        }
      ],
      "missed": [...],
      "allowed": [...]
    }
  }
}
```

### Console Output

The program also displays a summary:

```
Game 1:
  Executed → 2 pins, 1 skewer
  Missed → 1 pin
  Allowed → 1 skewer
```

## How It Works

### Pin Detection

- Uses python-chess built-in `board.is_pinned()` method
- Identifies pieces that cannot move without exposing the king to attack
- Finds the attacking piece that creates the pin

### Skewer Detection

- Scans in sliding-piece directions (bishop, rook, queen)
- Looks for high-value pieces followed by lower-value pieces
- Verifies no pieces block the tactical line

### Analysis Process

1. **Position Analysis**: Examines each position for tactical opportunities
2. **Move Comparison**: Compares player moves with Stockfish engine suggestions
3. **Opportunity Classification**: Determines if tactics were executed, missed, or allowed
4. **Result Generation**: Creates detailed JSON output and console summary

## Customization

### Adding Your Own Games

- Replace `games.pgn` with your own PGN file
- Ensure PGN format is valid and moves are legal

### Adjusting Detection Sensitivity

- Modify piece value thresholds in `piece_values` dictionary
- Adjust Stockfish analysis depth in `analyze_move()` method

### Changing Stockfish Path

- Update `stockfish_path` in the `main()` function
- Use the path returned by `test_stockfish.py`

## Troubleshooting

### Common Issues

1. **Stockfish not found:**

   - Run `python test_stockfish.py` to find correct path
   - Ensure Stockfish executable has proper permissions

2. **No tactics detected:**

   - Check if PGN file contains valid games
   - Verify Stockfish is working correctly
   - Games may not contain pin/skewer opportunities

3. **Memory issues with large games:**
   - Reduce Stockfish analysis depth
   - Process games one at a time

### Performance Tips

- Lower Stockfish depth for faster analysis
- Use smaller PGN files for testing
- Close other applications to free memory

## Technical Details

### Key Classes

- **`ChessTacticsDetector`**: Main detector class
- **`TacticalEvent`**: Data structure for tactical events
- **`ChessTacticsDetector.detect_pins()`**: Pin detection logic
- **`ChessTacticsDetector.detect_skewers()`**: Skewer detection logic

### Dependencies

- **python-chess**: Chess board representation and PGN parsing
- **chess.engine**: Stockfish integration
- **json**: Output formatting
- **dataclasses**: Data structure definitions

## License

This project is open source. Feel free to modify and distribute according to your needs.

## Contributing

Improvements and bug reports are welcome! Focus areas:

- Better tactical detection algorithms
- Additional tactical motif types
- Performance optimizations
- User interface enhancements

output -
<img width="1918" height="1026" alt="image" src="https://github.com/user-attachments/assets/6c187702-fa1e-4a59-bafd-24d46357175f" />

