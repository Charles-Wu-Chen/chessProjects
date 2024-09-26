# This file should contain different functions to analyse a chess game
# TODO: analysis with WDL+CP is very janky
import os, sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chess import engine, pgn, Board
import chess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import functions
from functions import *
import logging
import evalDB
from glob import glob
import csv
import shutil
import traceback
import fenToImage



# encoding = "GB2312"
encoding = 'utf-8'


def makeComments(gamesFile: str, outfile: str, analysis, limit: int, engine: engine, sf: engine, cache: bool = False) -> list:
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
    
    gameNR = 1
    with open(gamesFile, 'r',encoding=encoding) as pgn:
        while (game := chess.pgn.read_game(pgn)):
            print(f'Starting with game {gameNR}...')
            gameNR += 1

            board = game.board()
            
            # I found no way to add comments to an existing PGN, so I create a new PGN with the comments
            newGame = chess.pgn.Game()
            newGame.headers = game.headers

            for move in game.mainline_moves():
                print(move)
                # Checks if we have the starting position and need to make a new node
                if board == game.board():
                    node = newGame.add_variation(move)
                else:
                    node = node.add_variation(move)

                board.push(move)
                # Adds a comment after every move with the wdl
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
            print(newGame, file=open(outfile, 'a+'), end='\n\n')
    # engine.quit()
    return []

def normalize_name(name: str) -> str:
    """Convert to lowercase, remove whitespace and special characters."""
    return re.sub(r'\W+', '', name.lower())


def findMistakes(pgnPath: str, sf: engine, playerName: str = None, mistakeValue: int = 200) -> list:
    """
    This function takes a PGN with WDL evaluations and finds the mistakes in the game
    pgnPath: str
        The path to the PGN file
    return: list
        A list with the positions where mistakes occured
    """


    lastWDL = None
    positions = list()
    
    pgn_file_name = os.path.basename(pgnPath)

    with open(pgnPath, 'r', encoding=encoding) as pgn:
        while (game := chess.pgn.read_game(pgn)):
            node = game
            white_player = game.headers.get("White", "Unknown")
            black_player = game.headers.get("Black", "Unknown")
            previous_move = None
            while not node.is_end():
                
                
                if node.comment:
                    lastWDL = getWDLfromComment(node.comment)
                else:
                    node = node.variations[0]
                    continue

                board = node.board()
                pos = board.fen()
                sharpness = functions.sharpnessLC0(lastWDL)
                node = node.variations[0]
                turn = "White" if board.turn else "Black"
                if board.turn == chess.WHITE:
                    current_player = white_player
                else:
                    current_player = black_player
                if node.comment:
                    currWDL = getWDLfromComment(node.comment)
                    if node.turn() == chess.WHITE:
                        diff = currWDL[0]+currWDL[1]*0.5-(lastWDL[0]+lastWDL[1]*0.5)
                    else:
                        diff = currWDL[2]+currWDL[1]*0.5-(lastWDL[2]+lastWDL[1]*0.5)

                    # print(f"diff  {diff} for move {node.move} with current wdl {currWDL}")
                    if diff > mistakeValue:
                        if playerName:
                            normalized_player_name = normalize_name(playerName)
                            normalized_current_player = normalize_name(current_player)
                            if normalized_player_name not in normalized_current_player:
                                previous_move = board.san(node.move)
                                # print (f"normalized_player_name  {normalized_player_name} normalized_current_player {normalized_current_player} ")
                                continue # only generate puzzle for the player if name provided.
                        bestMove = sf.analyse(board, chess.engine.Limit(depth=20))['pv'][0]
                        positions.append(
                            {
                                "id": None,
                                "White Player": white_player,
                                "Black Player": black_player,
                                "Current Turn": turn,
                                "Previous Move": previous_move,
                                "Game Move": board.san(node.move),
                                "Best Move": board.san(bestMove),
                                "PGN File": pgn_file_name,
                                "FEN": pos,
                                "Sharpness": sharpness
                            }
                        )
                previous_move = board.san(node.move)
    return positions


def getWDLfromComment(comment: str) -> []:
        # Regular expression to capture the three numbers inside square brackets
    pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
    
    # Search for the pattern in the string
    match = re.search(pattern, comment)
    
    if match:
        # Extract the numbers and convert them to integers
        num1, num2, num3 = map(int, match.groups())
        return [num1, num2, num3]
    else:
        # Return an empty list if the pattern is not found
        return []
    

def process_pgn_folder(input_folder: str, sf : engine, player_name: str = None, mistakeValue: int = 200):
    # Ensure the input folder exists
    if not os.path.isdir(input_folder):
        raise ValueError(f"The input folder '{input_folder}' does not exist.")
    
    os.makedirs(input_folder+"\\"+"done" , exist_ok=True)
    # Get all PGN files in the input folder
    pgn_files = glob(os.path.join(input_folder, "*.pgn"))

    # Get the current date and time
    timestamp_str = current_time_str("%Y%m%d%H%M%S")
    running_number = 1  # Initialize running number
    # Define the output CSV file path within the input folder
    filename = timestamp_str+"move_details.csv"
    output_csv_path = os.path.join(input_folder, filename)
    with open(output_csv_path, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=[
            "ID",  "White Player", "Black Player", "Current Turn",
            "Previous Move", "Game Move", "Best Move","PGN File", "FEN", "Sharpness"
        ])
        writer.writeheader()
    
    # Process each PGN file
    for pgn_file in pgn_files:
        try:
            results = findMistakes(pgn_file, sf, player_name, mistakeValue)
            
            with open(output_csv_path, mode='a', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=[
                "id", "White Player", "Black Player", "Current Turn",
                "Previous Move", "Game Move", "Best Move", "PGN File", "FEN", "Sharpness"
            ])
                for result in results:
                    id = running_number
                    running_number += 1
                    result["id"] = id
                    writer.writerow(result)
                    print(f"Result line {result}")
                
            shutil.move(pgn_file, input_folder+"\\"+"done")
            

        except Exception as e: 
            # Print the exception type and message
            print(f"An error occurred: {e}")
            
            # Print the detailed traceback information
            traceback.print_exc()
            continue
    return filename

def current_time_str(format : str):
    current_timestamp = datetime.now()

    # Format it as a string
    timestamp_str = current_timestamp.strftime(format)
    return timestamp_str
    
    
    
def split_pgn_file(pgn_file_path, output_directory):
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)
    
    # Open the PGN file
    with open(pgn_file_path, encoding=encoding) as pgn_file:
        game_count = 1
        while True:
            # Parse each game
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            
            # Extract player names
            white_player = game.headers.get("White", "Unknown").replace(" ", "_")
            black_player = game.headers.get("Black", "Unknown").replace(" ", "_")
            
            timestamp_str = current_time_str("%d%H%M%S%f")

            # Create a unique filename
            filename = f"{game_count}_{white_player}_vs_{black_player}_{timestamp_str}.pgn"
            filename = re.sub(r'[,<>:"/\\|?*]', '_', filename)
            output_path = os.path.join(output_directory, filename)
            
            # Save the game to a new PGN file
            with open(output_path, 'w') as output_file:
                exporter = chess.pgn.FileExporter(output_file)
                game.accept(exporter)
            
            game_count += 1

    print(f"Split PGN file into {game_count - 1} individual games.")


if __name__ == '__main__':

    input_folder = r"\out\pgn\0924TueNight"
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
                results = makeComments(pgn_file, comment_directory+"\\"+file_name, analysisCPnWDL, 5000, leela, sf, True)
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

    




    
