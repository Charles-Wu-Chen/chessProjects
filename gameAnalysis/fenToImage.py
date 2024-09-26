import chess
import chess.svg
import cairosvg
from PIL import Image, ImageDraw, ImageFont
import os, io
import csv
import functions

def generate_chess_image(fen: str, output_file: str, white_player: str = "White", black_player: str = "Black", 
                         previous_move: str = "", id: str = "", sharpness: float = 0.0) -> None:
    """
    Generates a chess board image from a FEN string, adds additional information, and saves it to a file.
    
    :param fen: The FEN string representing the chess position.
    :param output_file: The name of the output image file (e.g., "chess_board.png").
    :param white_player: The name of the player controlling white pieces.
    :param black_player: The name of the player controlling black pieces.
    :param previous_move: The last move made in algebraic notation (e.g., "e2e4").
    :param game_move: The current game move or any notable move to highlight.
    """
    # Create a chess board from the FEN string
    board = chess.Board(fen)

    # Determine if the board should be flipped
    flip = not board.turn  # Flip when it's Black's turn

    # Generate an SVG of the board
    svg_image = chess.svg.board(board, flipped=flip)

    # Convert SVG to PNG using CairoSVG
    png_image = cairosvg.svg2png(bytestring=svg_image)

    # Open the image using PIL
    image = Image.open(io.BytesIO(png_image))

    # Determine the image size and create a larger canvas to add text
    width, height = image.size
    new_height = height + 100  # Additional space for text
    new_image = Image.new("RGB", (width, new_height), "white")
    new_image.paste(image, (0, 0))

    # Resize the image to 50% of its original size
    new_width = int(width * 0.5)
    new_height = int(new_height * 0.5)
    new_image = new_image.resize((new_width, new_height), Image.LANCZOS)

    # Draw the additional text
    draw = ImageDraw.Draw(new_image)

    # Load a font
    try:
        font = ImageFont.truetype("arial.ttf", 8)  # Font size 8
    except IOError:
        font = ImageFont.load_default()

    # Text content, each info on a new line with adjusted height
    side_to_move = "White" if board.turn else "Black"
    line_height = 9  # Slightly reduced line height
    start_y = int(height * 0.5) + 2

    draw.text((5, start_y), f"ID: {id}", fill="black", font=font)
    draw.text((5, start_y + line_height), f"{white_player} (W) vs {black_player} (B)", fill="black", font=font)
    draw.text((5, start_y + 2*line_height), f"Side to Move: {side_to_move}", fill="black", font=font)
    draw.text((5, start_y + 3*line_height), f"Previous Move: {previous_move}", fill="black", font=font)
    draw.text((5, start_y + 4*line_height), f"Sharpness: {sharpness}", fill="black", font=font)

    # Save the final image
    new_image.save(output_file)
    print(f"Chess board image saved as {output_file}")


def generate_images_from_csv(csv_file_path: str, output_image_path: str):
    # Read the CSV file
    with open(csv_file_path, mode='r') as csv_file:
        reader = csv.DictReader(csv_file)

        for i, row in enumerate(reader):
            fen = row["FEN"]
            white_player = row["White Player"]
            black_player = row["Black Player"]
            previous_move = row["Previous Move"]
            id = row ["ID"]
            sharpness = row ["Sharpness"]
            
            # Define output file name (you can customize this)
            output_file = output_image_path + "\\" + f"chess_image_{i+1}.png"
            
            # Generate the image
            generate_chess_image(fen, output_file, white_player, black_player, previous_move, id, sharpness)


# Example usage:
# fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
# output_file = "chess_board_verbose.png"
# generate_chess_image(fen, output_file, white_player="Magnus Carlsen", black_player="Hikaru Nakamura", 
#                      previous_move="e2e4", pgn="filename", id= "1")


# csv_file_path = functions.relativePathToAbsPath(r'\out\pgn\0831Kevin\comment\20240831145327move_details.csv')
# output_image_path = functions.relativePathToAbsPath(r'\out\puzzle\0831Kevin')
# os.makedirs(output_image_path , exist_ok=True)
# generate_images_from_csv(csv_file_path, output_image_path)
