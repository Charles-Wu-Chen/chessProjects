import os
import sys
import re
import csv
import shutil
import traceback
import chess
import logging

from datetime import datetime
from glob import glob


from chess import engine, Board 


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import functions
from functions import configureEngine, analysisCPnWDL, analysisCP, analysisWDL, formatInfo
import fenToImage
import evalDB

# Constants
ENCODING = 'utf-8'

# Add this near the top of the file, after the imports
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def normalize_name(name: str) -> str:
    """Convert to lowercase, remove whitespace and special characters."""
    return re.sub(r'\W+', '', name.lower())

def get_wdl_from_comment(comment: str) -> list:
    """Extract WDL values from a comment string."""
    pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
    match = re.search(pattern, comment)
    return list(map(int, match.groups())) if match else []

def make_comments(games_file: str, outfile: str, analysis, limit: int, engine: engine, sf: engine, cache: bool = False) -> list:
    """
    This function plays thorugh the games in a file and makes comments to them.
    The specific comments depend on the analysis method chosen
    gamesFile: str
        The name of a PGN file containing the games to analyse
    outfile: str
        The path of the output PGN file with the WDL comments
    analysis
        This is a function which analyses to positions. I kept it separate sicne 
        it's easier to change the type of analysis (WDL, centipawns, ...)
    engine: engine
        The engine to analyse. Note that it must fit with the analysis function (a bit inelegant)
    cache: bool
        If this is set to true, caching will be enabled, using evalDB
    return -> list
        A list of lists for each game, containing the WDL and score after every move
    """
    logger.info(f"Starting to make comments for file: {games_file}")
    game_nr = 1
    with open(games_file, 'r', encoding=ENCODING) as pgn:
        while (game := chess.pgn.read_game(pgn)):
            print(f'Starting with game {game_nr}...')
            game_nr += 1

            board = game.board()
            new_game = chess.pgn.Game()
            new_game.headers = game.headers

            for move in game.mainline_moves():
                print(move)
                node = new_game.add_variation(move) if board == game.board() else node.add_variation(move)
                board.push(move)

                if cache:
                    pos = board.fen()
                    posDB = functions.modifyFEN(pos)
                    if evalDB.contains(posDB):
                        # TODO: not general enough
                        evalDict = evalDB.getEval(posDB)
                        wdl = evalDict['wdl']
                        cp = evalDict['cp']
                        if evalDict['depth'] <= 0:
                            info = analysisCP(board, None, 4)
                            cp = info['score']
                        if evalDict['nodes'] <= 0:
                            wdl = analysisWDL(board, engine, limit, sf)
                            wdlList = [int(x) for x in wdl[1:-1].split(',')]
                            print(wdlList)
                            evalDB.update(position=posDB, nodes=limit, w=wdlList[0], d=wdlList[1], l=wdlList[2])
                            print(f'WDL calculated: {wdl}')
                        print('Cache hit!')
                        print(wdl, cp)
                        node.comment = f'{str(wdl)};{cp}'
                    else:
                        infos = analysis(board, engine, limit, sf)
                        if infos:
                            iLC0, iSF = infos
                            ana = formatInfo(iLC0, iSF)
                            print(ana)
                            node.comment = ana
                            cp = int(ana.split(';')[1])
                            wdl = [ int(w) for w in ana.split(';')[0].replace('[', '').replace(']', '').strip().split(',') ]
                            evalDB.insert(posDB, nodes=limit, cp=cp, w=wdl[0], d=wdl[1], l=wdl[2], depth=iSF['depth'])
                else:
                    node.comment = analysis(board, engine, limit)

            print(new_game, file=open(outfile, 'a+'), end='\n\n')
    logger.info(f"Finished making comments for file: {games_file}")
    return []

def find_mistakes(pgn_path: str, sf: engine, player_name: str = None, mistake_value: int = 200) -> list:
    """Find mistakes in a chess game based on WDL evaluations."""
    logger.info(f"Starting to find mistakes in file: {pgn_path}")
    positions = []
    pgn_file_name = os.path.basename(pgn_path)

    with open(pgn_path, 'r', encoding=ENCODING) as pgn:
        while (game := chess.pgn.read_game(pgn)):
            node = game
            white_player = game.headers.get("White", "Unknown")
            black_player = game.headers.get("Black", "Unknown")
            previous_move = None

            while not node.is_end():
                if node.comment:
                    last_wdl = get_wdl_from_comment(node.comment)
                else:
                    node = node.variations[0]
                    continue

                board = node.board()
                pos = board.fen()
                sharpness = functions.sharpnessLC0(last_wdl)
                node = node.variations[0]
                turn = "White" if board.turn else "Black"
                current_player = white_player if board.turn else black_player

                if node.comment:
                    curr_wdl = get_wdl_from_comment(node.comment)
                    diff = calculate_wdl_diff(curr_wdl, last_wdl, node.turn())

                    if diff > mistake_value:
                        if should_generate_puzzle(player_name, current_player):
                            best_move = sf.analyse(board, engine.Limit(depth=20))['pv'][0]
                            positions.append(create_position_dict(white_player, black_player, turn, previous_move, board, node.move, best_move, pgn_file_name, pos, sharpness, last_wdl))

                previous_move = board.san(node.move)

    logger.info(f"Finished finding mistakes in file: {pgn_path}. Found {len(positions)} positions.")
    return positions

def calculate_wdl_diff(curr_wdl, last_wdl, turn):
    if turn == chess.WHITE: 
        return curr_wdl[0] + curr_wdl[1] * 0.5 - (last_wdl[0] + last_wdl[1] * 0.5)
    else:
        return curr_wdl[2] + curr_wdl[1] * 0.5 - (last_wdl[2] + last_wdl[1] * 0.5)

def should_generate_puzzle(player_name, current_player):
    if player_name:
        return normalize_name(player_name) in normalize_name(current_player)
    return True

def create_position_dict(white_player, black_player, turn, previous_move, board, game_move, best_move, pgn_file_name, pos, sharpness, last_wdl):
    return {
        "id": None,
        "White Player": white_player,
        "Black Player": black_player,
        "Current Turn": turn,
        "Previous Move": previous_move,
        "Game Move": board.san(game_move),
        "Best Move": board.san(best_move),
        "PGN File": pgn_file_name,
        "FEN": pos,
        "Sharpness": f"{sharpness:.3f}",  # Format sharpness to 3 decimal places
        "WDL": str(last_wdl)  # Add the WDL value
    }

def process_pgn_folder(input_folder: str, sf: engine, player_name: str = None, mistake_value: int = 200):
    """Process all PGN files in a folder to find mistakes."""
    logger.info(f"Starting to process PGN folder: {input_folder}")
    if not os.path.isdir(input_folder):
        raise ValueError(f"The input folder '{input_folder}' does not exist.")
    
    os.makedirs(os.path.join(input_folder, "done"), exist_ok=True)
    pgn_files = glob(os.path.join(input_folder, "*.pgn"))

    timestamp_str = current_time_str("%Y%m%d%H%M%S")
    filename = f"{timestamp_str}move_details.csv"
    output_csv_path = os.path.join(input_folder, filename)

    with open(output_csv_path, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=[
            "ID", "White Player", "Black Player", "Current Turn",
            "Previous Move", "Game Move", "Best Move", "PGN File", "FEN", "Sharpness", "WDL"
        ])
        writer.writeheader()

    running_number = 1
    for pgn_file in pgn_files:
        try:
            results = find_mistakes(pgn_file, sf, player_name, mistake_value)
            
            with open(output_csv_path, mode='a', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=[
                    "id", "White Player", "Black Player", "Current Turn",
                    "Previous Move", "Game Move", "Best Move", "PGN File", "FEN", "Sharpness", "WDL"
                ])
                for result in results:
                    result["id"] = running_number
                    running_number += 1
                    writer.writerow(result)
                    print(f"Result line {result}")
                
            shutil.move(pgn_file, os.path.join(input_folder, "done"))

        except Exception as e: 
            print(f"An error occurred: {e}")
            traceback.print_exc()
            continue

    logger.info(f"Finished processing PGN folder. Output CSV: {output_csv_path}")
    return filename

def current_time_str(format: str):
    """Get current time as a formatted string."""
    return datetime.now().strftime(format)

def split_pgn_file(pgn_file_path, output_directory):
    """Split a PGN file into individual game files."""
    logger.info(f"Starting to split PGN file: {pgn_file_path}")
    os.makedirs(output_directory, exist_ok=True)
    
    with open(pgn_file_path, encoding=ENCODING) as pgn_file:
        game_count = 1
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            
            white_player = game.headers.get("White", "Unknown").replace(" ", "_")
            black_player = game.headers.get("Black", "Unknown").replace(" ", "_")
            
            timestamp_str = current_time_str("%d%H%M%S%f")
            filename = f"{game_count}_{white_player}_vs_{black_player}_{timestamp_str}.pgn"
            filename = re.sub(r'[,<>:"/\\|?*]', '_', filename)
            output_path = os.path.join(output_directory, filename)
            
            with open(output_path, 'w') as output_file:
                exporter = chess.pgn.FileExporter(output_file)
                game.accept(exporter)
            
            game_count += 1

    logger.info(f"Finished splitting PGN file into {game_count - 1} individual games.")

if __name__ == '__main__':
    logger.info("Starting main execution of findMistake.py")

    input_folder = r"\out\pgn\0929OlympiadAusWomen"
    player_name = None
    mistakeValue = 200

    # input_file = r"KevinZhangXY_vs_citso_2024.08.30.pgn"
    input_directory = functions.relativePathToAbsPath(input_folder)

    output_directory = functions.relativePathToAbsPath(input_folder + "\\out" )
    split_directory = functions.relativePathToAbsPath(input_folder + "\\split" )
    comment_directory = functions.relativePathToAbsPath(input_folder + "\\comment" )
    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(split_directory, exist_ok=True)
    os.makedirs(comment_directory, exist_ok=True)

    #  step 1 split into individual file per game
    raw_pgn_files = glob(os.path.join(input_directory, "*.pgn"))
    for pgn_file in raw_pgn_files:
        split_pgn_file(pgn_file, output_directory)
        shutil.move(pgn_file, split_directory)

    # step 2 make comment with WDL and eva
    op = {'WeightsFile': r'K:\leela\lc0-v0.30.0-windows-gpu-nvidia-cudnn\791556.pb.gz', 'UCI_ShowWDL': 'true'}

    leela = configureEngine(r'K:\leela\lc0-v0.30.0-windows-gpu-nvidia-cudnn\lc0.exe', op)
    sf = configureEngine(r'K:\github\stockfish-windows-x86-64\stockfish\stockfish-windows-x86-64.exe', {'Threads': '10', 'Hash': '4096'})

    #this is test one single file 
    # makeComments(functions.relativePathToAbsPath(r'\resources\pgn\lichess_study_2024-2nd-half_chapter-12-johansen-darryl-vsshen-zhiyuan_by_wuchen1_2024.07.15.pgn'), functions.relativePathToAbsPath(r'\out\pgn\johansen-darryl-vsshen-zhiyuan.pgn'), analysisCPnWDL, 5000, leela, True)
    # print(findMistakes(functions.relativePathToAbsPath(r'\out\pgn\johansen-darryl-vsshen-zhiyuan.pgn'), sf))

    pgn_files = glob(os.path.join(output_directory, "*.pgn"))
    print(f"after split pgn loaded {pgn_files}")


    # Process each PGN file
    try:
        for pgn_file in pgn_files:
            file_name = os.path.basename(pgn_file)
            try: 
                results = make_comments(pgn_file, comment_directory+"\\"+file_name, analysisCPnWDL, 5000, leela, sf, True)
                os.makedirs(split_directory+"\\done\\", exist_ok=True)
                shutil.move(pgn_file, split_directory+"\\done\\"+file_name)
            except Exception as exc:
                print (traceback.format_exc())
                print (exc)
        
        detail_file_name = process_pgn_folder(comment_directory, sf, player_name, mistakeValue)
    # print(findMistakes(functions.relativePathToAbsPath(r'\out\pgn\2024arvostudy\Kevin _vs_Ariyathilaka,Saheli _2024.08.09.pgn'), sf))

    finally:
        sf.quit()
        leela.quit()

    csv_file_path = comment_directory+"\\"+detail_file_name
    output_image_path = comment_directory+"\\img\\"
    os.makedirs(output_image_path , exist_ok=True)
    fenToImage.generate_images_from_csv(csv_file_path, output_image_path)
    logger.info("Finished main execution of findMistake.py")
