import fenToImage
import findMistake

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



input_folder = r"\out\pgn\test"
player_name = "alice"
# encoding = "GB2312"
encoding = 'utf-8'
mis = 200




if __name__ == '__main__':


    # input_file = r"KevinZhangXY_vs_citso_2024.08.30.pgn"
    input_directory = functions.relativePathToAbsPath(input_folder)

    output_directory = functions.relativePathToAbsPath(input_folder + "\\out" )
    split_directory = functions.relativePathToAbsPath(input_folder + "\\split" )
    comment_directory = functions.relativePathToAbsPath(input_folder + "\\comment" )
    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(split_directory, exist_ok=True)
    os.makedirs(comment_directory, exist_ok=True)


    # Ensure the directory exists before attempting to delete
    if os.path.exists(comment_directory):
        # Loop through all the files and subdirectories
        for filename in os.listdir(comment_directory):
            file_path = os.path.join(comment_directory, filename)
            # If it's a file, remove it
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            # If it's a directory, remove it along with its contents
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

    #  step 1 split into individual file per game
    raw_pgn_files = glob(os.path.join(input_directory, "*.pgn"))
    for pgn_file in raw_pgn_files:
        findMistake.split_pgn_file(pgn_file, output_directory)
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
                results = findMistake.makeComments(pgn_file, comment_directory+"\\"+file_name, analysisCPnWDL, 5000, leela, sf, True)
                os.makedirs(split_directory+"\\done\\", exist_ok=True)
                shutil.move(pgn_file, split_directory+"\\done\\"+file_name)
            except Exception as exc:
                print (traceback.format_exc())
                print (exc)
        
        detail_file_name = findMistake.process_pgn_folder(comment_directory, sf, player_name, 100)
    # print(findMistakes(functions.relativePathToAbsPath(r'\out\pgn\2024arvostudy\Kevin _vs_Ariyathilaka,Saheli _2024.08.09.pgn'), sf))

    finally:
        sf.quit()
        leela.quit()

    csv_file_path = comment_directory+"\\"+detail_file_name
    output_image_path = comment_directory+"\\img\\"
    os.makedirs(output_image_path , exist_ok=True)
    fenToImage.generate_images_from_csv(csv_file_path, output_image_path)