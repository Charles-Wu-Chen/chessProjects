# This file produces a report of a tournament given as PGN file
# It uses methods written in analysis.py

import analysis
import chess
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import glob
import os
import sys

import statistics
from scipy import stats

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import functions
from findMistake import split_pgn_file  # Import the split_pgn_file function

# Define the list of players at the top level
FOCUS_PLAYERS = ['Shen, Zhiyuan', 'Zhang, Jilin', 'Vincent, Alaina', 'Ryjanova, Julia', 'Nguyen, Thu Giang']


def getPlayers(pgnPath: str, whiteList: list = None) -> list:
    """
    This function gets the names of the players in a tournament.
    pgnPath: str
        The path to the PGN file of the tournament
    whiteList: list
        A list of player names that should be included. The name has to be the same as in the PGN
        If no whiteList is specified, all players will be included
    return -> list
        A list of the players' names
    """
    players = set()
    with open(pgnPath, 'r') as pgn:
        while (game := chess.pgn.read_game(pgn)):
            for color in ["White", "Black"]:
                player = game.headers[color]
                if not whiteList or player in whiteList:
                    players.add(player)
    return list(players)


def getMoveData(pgnPaths: list) -> pd.DataFrame:
    """
    This function reads PGN files and stores the data from each move in a dataframe
    return -> pandas.DataFrame
        A dataframe containing various fields where each contains a list with the data after each move
    """
    data = {'player': [], 'rating': [], 'acc': [], 'sharp': []}
    for pgnPath in pgnPaths:
        with open(pgnPath, 'r') as pgn:
            while game := chess.pgn.read_game(pgn):
                if 'WhiteElo' in game.headers and 'BlackElo' in game.headers:
                    wRating = int(game.headers['WhiteElo'])
                    bRating = int(game.headers['BlackElo'])
                else:
                    continue
                white = game.headers['White']
                black = game.headers['Black']
                
                node = game
                cpB = None
                
                while not node.is_end():
                    if functions.readComment(node, True, True):
                        sharp = functions.sharpnessLC0(functions.readComment(node, True, True)[0])
                    node = node.variations[0]
                    if not functions.readComment(node, True, True):
                        continue
                    cpA = functions.readComment(node, True, True)[1]
                    if not cpB:
                        cpB = cpA
                        continue

                    if node.turn():
                        wpB = functions.winP(cpB * -1)
                        wpA = functions.winP(cpA * -1)
                        data['player'].append(black)
                        data['rating'].append(bRating)
                    else:
                        wpB = functions.winP(cpB)
                        wpA = functions.winP(cpA)
                        data['player'].append(white)
                        data['rating'].append(wRating)

                    acc = min(100, functions.accuracy(wpB, wpA))
                    data['acc'].append(round(acc))
                    data['sharp'].append(sharp)

                    cpB = cpA
    return pd.DataFrame(data)


def plotAccuracyDistribution(df: pd.DataFrame) -> None:
    """
    This plots the accuracy distribution for all moves in the given PGNs
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ratingBounds = (2600, 2700, 2850)
    colors = ['#689bf2', '#f8a978', '#ff87ca', '#beadfa', '#A1EEBD']
    for i in range(len(ratingBounds) - 1):
        x1 = list(df[df['rating'].isin(range(ratingBounds[i], ratingBounds[i + 1]))]['acc'])
        nMoves = len(x1)
        acc = [0] * 101
        for x in x1:
            acc[x] += 1
        acc = [x / nMoves for x in acc]
        ax.bar([x - 0.5 for x in range(101)], acc, width=1, color=colors[i % len(ratingBounds)], edgecolor='black', linewidth=0.5, alpha=0.5, label=f'{ratingBounds[i]}-{ratingBounds[i + 1]}')

    ax.set_facecolor('#e6f7f2')
    ax.set_yscale('log')
    plt.xlim(0, 100)
    ax.invert_xaxis()
    ax.set_xlabel('Move Accuracy')
    ax.set_ylabel('Relative number of moves')
    ax.legend(loc='best')
    plt.subplots_adjust(bottom=0.1, top=0.95, left=0.1, right=0.95)
    plt.title('Accuracy Distribution')
    ax.yaxis.set_major_formatter(mtick.ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    
    plt.savefig(f'../out/accDist2600-2800.png', dpi=500)
    plt.show()


def generateAccuracyDistributionGraphs(pgnPath: str, players: list):
    """
    This function generates the accuracy distribution graphs for the players in the tournament
    pgnPath: str
        The path to the PGN file of the tournament
    players: list
        A list of the names of the players in the tournament
    """
    for player in players:
        analysis.plotAccuracyDistributionPlayer(pgnPath, player, f'../out/{player}-{pgnPath.split("/")[-1][:-4]}')


def getPlayerScores(pgnPath: str) -> dict:
    """
    This function gets the scores for all players in a tournament
    pgnPath: str
        The path to the PGN file of the tournament
    return -> dict
        A dictionary indexed by the player containing the number of games, points, games with white, points with white, games with black, points with black
    """
    scores = {}
    with open(pgnPath, 'r') as pgn:
        while (game := chess.pgn.read_game(pgn)):
            w = game.headers["White"]
            b = game.headers["Black"]
            if w not in scores:
                scores[w] = [0, 0, 0, 0, 0, 0]
            if b not in scores:
                scores[b] = [0, 0, 0, 0, 0, 0]
            scores[w][0] += 1
            scores[w][2] += 1
            scores[b][0] += 1
            scores[b][4] += 1
            if "1/2" in (r := game.headers["Result"]):
                scores[w][1] += 0.5
                scores[w][3] += 0.5
                scores[b][1] += 0.5
                scores[b][5] += 0.5
            elif r == "1-0":
                scores[w][1] += 1
                scores[w][3] += 1
            elif r == "0-1":
                scores[b][1] += 1
                scores[b][5] += 1
            else:
                print(f"Can't identify result: {r}")
    return scores


def getMoveSituation(pgnPath: str) -> dict:
    """
    This function returns a dictionary index by the players and containing the number of moves where they were:
        much better (1+), slightly better (1-0.5), equal, slightly worse and much worse
    """
    moves = {}
    with open(pgnPath, 'r') as pgn:
        while (game := chess.pgn.read_game(pgn)):
            w = game.headers["White"]
            b = game.headers["Black"]
            if w not in moves:
                moves[w] = [0, 0, 0, 0, 0, 0]
            if b not in moves:
                moves[b] = [0, 0, 0, 0, 0, 0]
            node = game

            while not node.is_end():
                node = node.variations[0]
                if node.comment:
                    cp = int(float(node.comment.split(';')[-1]))
                    if not node.turn():
                        moves[w][0] += 1
                        if cp > 100:
                            moves[w][1] += 1
                        elif cp >= 50:
                            moves[w][2] += 1
                        elif cp >= -50:
                            moves[w][3] += 1
                        elif cp > -100:
                            moves[w][4] += 1
                        else:
                            moves[w][5] += 1
                    else:
                        moves[b][0] += 1
                        if cp > 100:
                            moves[b][5] += 1
                        elif cp >= 50:
                            moves[b][4] += 1
                        elif cp >= -50:
                            moves[b][3] += 1
                        elif cp > -100:
                            moves[b][2] += 1
                        else:
                            moves[b][1] += 1
    return moves


def worseGames(pgnPath: str) -> dict:
    """
    This function counts the number of games where a player was worse and the number of lost games.
    """
    games = {}
    with open(pgnPath, 'r') as pgn:
        while (game := chess.pgn.read_game(pgn)):
            w = game.headers["White"]
            b = game.headers["Black"]
            r = game.headers["Result"]
            if w not in games:
                games[w] = [0, 0]
            if b not in games:
                games[b] = [0, 0]
            if r == '1-0':
                games[b][1] += 1
            elif r == '0-1':
                games[w][1] += 1

            node = game
            rec = [False, False]

            while not node.is_end():
                node = node.variations[0]
                if node.comment:
                    cp = int(float(node.comment.split(';')[-1]))
                    if cp < -100 and not rec[0]:
                        games[w][0] += 1
                        rec[0] = True
                    elif cp > 100 and not rec[1]:
                        games[b][0] += 1
                        rec[1] = True
    return games


def betterGames(pgnPath: str) -> dict:
    games = {}
    with open(pgnPath, 'r') as pgn:
        while (game := chess.pgn.read_game(pgn)):
            w = game.headers["White"]
            b = game.headers["Black"]
            r = game.headers["Result"]
            if w not in games:
                games[w] = [0, 0]
            if b not in games:
                games[b] = [0, 0]
            if r == '1-0':
                games[w][1] += 1
            elif r == '0-1':
                games[b][1] += 1

            node = game
            rec = [False, False]

            while not node.is_end():
                node = node.variations[0]
                if node.comment:
                    cp = int(float(node.comment.split(';')[-1]))
                    if cp < -100 and not rec[0]:
                        games[b][0] += 1
                        rec[0] = True
                    elif cp > 100 and not rec[1]:
                        games[w][0] += 1
                        rec[1] = True
    return games


def sortPlayers(d: dict, index: int) -> list:
    """
    This function takes a dictionary with a list as values and sorts the keys by the value at the index of the list
    """

    players = []
    for i in range(len(d.keys())):
        maximum = -1
        for k, v in d.items():
            if k in players:
                continue
            if v[index] > maximum:
                p = k
                maximum = v[index]
        players.append(p)
    return players


def getInaccMistakesBlunders(pgnPath: str) -> dict:
    games = {}
    # win percentage drop for inaccuracy, mistake and blunder
    bounds = (10, 15, 20)
    with open(pgnPath, 'r') as pgn:
        while (game := chess.pgn.read_game(pgn)):
            w = game.headers["White"]
            b = game.headers["Black"]
            if w not in games:
                games[w] = [0] * 3
            if b not in games:
                games[b] = [0] * 3

            node = game
            cpB = None

            while not node.is_end():
                node = node.variations[0]
                if node.comment:
                    cpA = functions.readComment(node, True, True)[1]
                    if cpB is None:
                        cpB = cpA
                        continue
                    if node.turn():
                        wpB = functions.winP(cpB * -1)
                        wpA = functions.winP(cpA * -1)
                        p = b
                    else:
                        wpB = functions.winP(cpB)
                        wpA = functions.winP(cpA)
                        p = w
                    diff = -wpA + wpB
                    if diff > bounds[2]:
                        print(p, w, b)
                        games[p][2] += 1
                    elif diff > bounds[1]:
                        games[p][1] += 1
                    elif diff > bounds[0]:
                        games[p][0] += 1
                    cpB = cpA
    return games


def createMovePlot(moves: dict, short: dict = None, filename: str = None):
    """
    This creates a plot with the number of moves a player spent being better or worse
    short: dict
        This is a dict that replaces names that are too long with shorter alternatives
    filename: str
        The name of the file to which the graph should be saved. 
        If no name is specified, the graph will be shown instead of saved
    """
    colors = ['#4ba35a', '#9CF196', '#F0EBE3', '#F69E7B', '#EF4B4B']

    fig, ax = plt.subplots(figsize=(10,6))
    ax.set_facecolor('#CDFCF6')
    plt.xticks(rotation=90)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    for player, m in moves.items():
        p = player.split(',')[0]
        if short:
            if p in short.keys():
                p = short[p]
        bottom = 0
        totalMoves = sum(m)
        factor = 200/totalMoves
        for i in range(len(m)-1, 0, -1):
            ax.bar(p, m[i]*factor, bottom=bottom, color=colors[i-1], edgecolor='black', linewidth=0.2)
            bottom += m[i]*factor

    fig.subplots_adjust(bottom=0.2, top=0.95, left=0.15, right=0.95)
    plt.title('Percentage of moves where players were better, equal and worse')
    ax.set_ylabel('Percentage of total moves')
    # TODO legend
    # ax.legend()

    if filename:
        plt.savefig(filename, dpi=500)
    else:
        plt.show()


def plotScoresArmageddon(scores: dict, filename: str = None) -> None:
    """
    Plotting scores with armageddon games (like in Norway chess)
    scores: dict
        The scores indexed by players and the values is a list containing classical score with white and black
        and the armageddon score with white and black
    filename: str
        The name of the file to which the graph should be saved. 
        If no name is specified, the graph will be shown instead of saved
    """
    colors = ['#ffffff', '#111111']
    patterns = ['', '', 'x', 'x']

    fig, ax = plt.subplots(figsize=(10,6))
    plt.xticks(rotation=90)
    
    for player in scores.keys():
        bottom = 0
        for i, s in enumerate(scores[player]):
            ax.bar(player, s, bottom=bottom, color=colors[i%2], edgecolor='grey', linewidth=0.7, hatch=patterns[i])
            bottom += s

    ax.set_facecolor('#e6f7f2')
    # ax.legend()
    fig.subplots_adjust(bottom=0.2, top=0.95, left=0.1, right=0.95)
    plt.title('Scores with White and Black')
    ax.set_ylabel('Tournament Score')

    if filename:
        plt.savefig(filename, dpi=500)
    else:
        plt.show()


def plotScores(scores: dict, short: dict = None, filename: str = None):
    """
    This function plots the scores of the tournament
    filename: str
        The name of the file to which the graph should be saved. 
        If no name is specified, the graph will be shown instead of saved
    """
    sortedPlayers = sortPlayers(scores, 1)
    colors = {3: '#FFFFFF', 5: '#111111'}

    fig, ax = plt.subplots(figsize=(10,6))
    plt.xticks(rotation=90)
    plt.yticks(range(0,10))

    ax.set_facecolor('#e6f7f2')
    for player in sortedPlayers:
        p = player.split(',')[0]
        if short:
            if p in short.keys():
                p = short[p]
        bottom = 0
        for i in [3, 5]:
            ax.bar(p, scores[player][i], bottom=bottom, color=colors[i], edgecolor='black', linewidth=0.7)
            bottom += scores[player][i]

    fig.subplots_adjust(bottom=0.2, top=0.95, left=0.1, right=0.95)
    plt.title('Scores with White and Black')
    ax.set_ylabel('Tournament Score')

    if filename:
        plt.savefig(filename, dpi=500)
    else:
        plt.show()


def plotWorseGames(worse: dict, short: dict = None, filename: str = None):
    """
    This function plots the number of games in which the players were worse and the number of games they lost
    filename: str
        The name of the file to which the graph should be saved. 
        If no name is specified, the graph will be shown instead of saved
    """
    sort = list(reversed(sortPlayers(worse, 0)))
    labels = list()
    for i, player in enumerate(sort):
        p = player.split(',')[0]
        if short:
            if p in short.keys():
                p = short[p]
        labels.append(p)

    fig, ax = plt.subplots(figsize=(10,6))

    ax.set_facecolor('#e6f7f2')
    plt.xticks(rotation=90)
    plt.yticks(range(0,10))
    plt.xticks(ticks=range(1, len(sort)+1), labels=labels)

    ax.bar([ i+1-0.2 for i in range(len(sort)) ], [ worse[p][0] for p in sort ], color='#689bf2', edgecolor='black', linewidth=0.5, width=0.4, label='# of worse games')
    ax.bar([ i+1+0.2 for i in range(len(sort)) ], [ worse[p][1] for p in sort ], color='#5afa8d', edgecolor='black', linewidth=0.5, width=0.4, label='# of lost games')
    ax.legend()
    fig.subplots_adjust(bottom=0.2, top=0.95, left=0.1, right=0.95)

    if filename:
        plt.savefig(filename, dpi=500)
    else:
        plt.show()


def plotBarChart(data: dict, labels: list, title: str, yLabel: str, short: dict = None, filename: str = None, sortIndex: int = 0) -> None:
    """
    This is a general function to create a bar chart with the players on the x-axis
    """
    sort = list(reversed(sortPlayers(data, sortIndex)))
    xLabels = list()
    for i, player in enumerate(sort):
        p = player.split(',')[0]
        if short:
            if p in short.keys():
                p = short[p]
        xLabels.append(p)

    colors = ['#689bf2', '#5afa8d', '#f8a978', '#fa5a5a']

    fig, ax = plt.subplots(figsize=(10,6))
    ax.set_facecolor('#e6f7f2')
    plt.xticks(rotation=90)
    # plt.yticks(range(0,10))
    plt.xticks(ticks=range(1, len(sort)+1), labels=xLabels)
    
    # Number of bars to plot
    nBars = len(data[sort[0]])
    width = 0.8/nBars
    offset = width * (1 / 2 - nBars/2)

    for j in range(nBars):
        ax.bar([i+1+offset+(width*j) for i in range(len(sort))], [data[p][j] for p in sort], color=colors[j], edgecolor='black', linewidth=0.5, width=width, label=labels[j])

    ax.legend()
    plt.title(title)
    ax.set_ylabel(yLabel)
    fig.subplots_adjust(bottom=0.2, top=0.95, left=0.1, right=0.95)

    if filename:
        plt.savefig(filename, dpi=500)
    else:
        plt.show()


def normaliseAccDistribution(accDis: dict) -> dict:
    """
    This takes an accuracy distribution and normalises the values.
    """
    norm = dict()
    total = sum(accDis.values())
    for k,v in accDis.items():
        norm[k] = v/total
    return norm


def plotMultAccDistributions(pgnPaths: list, filename: str = None):
    """
    This function plots multiple accuracy distribution graphs in one graph
    pgnPaths: list
        The path to the PGN files
    filename: str
        The name of the file to which the graph should be saved.
        If no name is specified, the graph will be shown instead of saved.
    """
    colors = ['#689bf2', '#f8a978', '#ff87ca', '#beadfa']
    labels = ['Arjun\nClosed', 'Arjun\nOpen']

    fig, ax = plt.subplots()
    ax.set_facecolor('#e6f7f2')
    ax.set_yscale('log')
    plt.xlim(0, 100)
    ax.invert_xaxis()
    for i, pgn in enumerate(pgnPaths):
        # TODO: handle the players
        accDis = analysis.getAccuracyDistributionPlayer(pgn, 'Erigaisi, Arjun')
        accDis = normaliseAccDistribution(accDis)
        ax.bar(accDis.keys(), accDis.values(), width=1, color=colors[i], edgecolor='black', linewidth='0.5', label=labels[i], alpha=0.5)
    plt.subplots_adjust(bottom=0.1, top=0.95, left=0.1, right=0.95)
    plt.title('Accuracy Distribution')
    ax.legend()

    if filename:
        plt.savefig(filename, dpi=500)
    else:
        plt.show()


def generateTournamentPlots(pgnPath: str, nicknames: dict = None, filename: str = None) -> None:
    players = getPlayers(pgnPath)
    generateAccuracyDistributionGraphs(pgnPath, players)
    scores = getPlayerScores(pgnPath)
    moveSit = getMoveSituation(pgnPath)
    worse = worseGames(pgnPath)
    better = betterGames(pgnPath)
    sharpChange = analysis.sharpnessChangePerPlayer(pgnPath)
    IMB = getInaccMistakesBlunders(pgnPath)
    print(IMB)

    if filename:
        createMovePlot(moveSit, nicknames, f'{filename}-movePlot.png')
        analysis.plotSharpChange(sharpChange, short=nicknames, filename=f'{filename}-sharpChange.png')
        plotScores(scores, nicknames, f'{filename}-scores.png')
        plotBarChart(worse, ['# of worse games', '# of lost games'], 'Number of worse and lost games', 'Number of games', nicknames, f'{filename}-worse.png', sortIndex=1)
        plotBarChart(better, ['# of better games', '# of won games'], 'Number of better and won games', 'Number of games', nicknames, f'{filename}-better.png', sortIndex=1)
        plotBarChart(IMB, ['Inaccuracies', 'Mistakes', 'Blunders'], 'Number of inaccuracies, mistakes and blunders', 'Number of moves', nicknames, f'{filename}-IMB.png', sortIndex=0)
    else:
        createMovePlot(moveSit, nicknames)
        analysis.plotSharpChange(sharpChange, short=nicknames)
        plotScores(scores, nicknames)
        plotBarChart(worse, ['# of worse games', '# of lost games'], 'Number of worse and lost games', 'Number of games', nicknames, sortIndex=1)
        plotBarChart(better, ['# of better games', '# of won games'], 'Number of better and won games', 'Number of games', nicknames, sortIndex=1)
        plotBarChart(IMB, ['Inaccuracies', 'Mistakes', 'Blunders'], 'Number of inaccuracies, mistakes and blunders', 'Number of moves', nicknames, sortIndex=0)


def get_players_from_file(pgn_path):
    """
    Extract player names from a single PGN file.
    """
    players = set()
    with open(pgn_path, 'r') as pgn:
        game = chess.pgn.read_game(pgn)
        if game:
            players.add(game.headers["White"])
            players.add(game.headers["Black"])
    return players


def get_all_players(folder_path, focus_players=None):
    """
    Get all unique player names from PGN files in the specified folder.
    Filter based on focus_players if provided.
    """
    all_players = set()
    pgn_files = glob.glob(os.path.join(folder_path, "*.pgn"))
    
    for pgn_file in pgn_files:
        players = get_players_from_file(pgn_file)
        all_players.update(players)
    
    return sorted(list(all_players))


def get_player_scores(folder_path, focus_players=None):
    """
    Calculate scores for all players across all games in the folder.
    Filter based on focus_players if provided.
    """
    scores = {}
    pgn_files = glob.glob(os.path.join(folder_path, "*.pgn"))
    
    for pgn_file in pgn_files:
        with open(pgn_file, 'r') as pgn:
            game = chess.pgn.read_game(pgn)
            if game:
                white = game.headers["White"]
                black = game.headers["Black"]
                if focus_players and white not in focus_players and black not in focus_players:
                    continue
                result = game.headers["Result"]
                
                for player in [white, black]:
                    if focus_players and player not in focus_players:
                        continue
                    if player not in scores:
                        scores[player] = [0, 0]  # [games played, points]
                    scores[player][0] += 1
                    if result == "1-0" and player == white:
                        scores[player][1] += 1
                    elif result == "0-1" and player == black:
                        scores[player][1] += 1
                    elif result == "1/2-1/2":
                        scores[player][1] += 0.5
    
    return scores


def get_move_situation(folder_path, focus_players=None):
    """
    Analyze the positional advantage of players throughout their games.
    Filter based on focus_players if provided.
    """
    moves = {}
    pgn_files = glob.glob(os.path.join(folder_path, "*.pgn"))
    
    for pgn_file in pgn_files:
        with open(pgn_file, 'r') as pgn:
            game = chess.pgn.read_game(pgn)
            if game:
                white = game.headers["White"]
                black = game.headers["Black"]
                
                # Only process the game if at least one player is in focus_players
                if focus_players and white not in focus_players and black not in focus_players:
                    continue
                
                for player in [white, black]:
                    if focus_players and player not in focus_players:
                        continue
                    if player not in moves:
                        moves[player] = [0, 0, 0, 0, 0, 0]  # [total, much better, slightly better, equal, slightly worse, much worse]
                
                node = game
                while not node.is_end():
                    node = node.variations[0]
                    if node.comment:
                        cp = int(float(node.comment.split(';')[-1]))
                        current_player = white if not node.turn() else black
                        if focus_players and current_player not in focus_players:
                            continue
                        moves[current_player][0] += 1
                        if cp > 100:
                            moves[current_player][1 if current_player == white else 5] += 1
                        elif cp >= 50:
                            moves[current_player][2 if current_player == white else 4] += 1
                        elif cp >= -50:
                            moves[current_player][3] += 1
                        elif cp > -100:
                            moves[current_player][4 if current_player == white else 2] += 1
                        else:
                            moves[current_player][5 if current_player == white else 1] += 1
    return moves


def analyze_player_performance(folder_path, focus_players=None):
    """
    Analyze player performance using accuracy and sharpness metrics.
    Filter based on focus_players if provided.
    """
    player_metrics = {}
    pgn_files = glob.glob(os.path.join(folder_path, "*.pgn"))
    
    for pgn_file in pgn_files:
        with open(pgn_file, 'r') as pgn:
            game = chess.pgn.read_game(pgn)
            if game:
                white = game.headers["White"]
                black = game.headers["Black"]
                if focus_players:
                    players_to_analyze = [p for p in [white, black] if p in focus_players]
                else:
                    players_to_analyze = [white, black]

                for player in players_to_analyze:
                    if player not in player_metrics:
                        player_metrics[player] = {"accuracies": [], "sharpness": []}
                
                node = game
                prev_eval = None
                
                while not node.is_end():
                    node = node.variations[0]
                    current_player = white if node.turn() else black
                    if focus_players and current_player not in focus_players:
                        continue
                    if node.comment:
                        current_eval = functions.readComment(node, True, True)
                        if current_eval and prev_eval:
                            wdl, cp = current_eval
                            prev_wdl, prev_cp = prev_eval
                            
                            # Calculate accuracy
                            wp_before = functions.winP(prev_cp)
                            wp_after = functions.winP(cp)
                            accuracy = functions.accuracy(wp_before, wp_after)
                            
                            # Calculate sharpness
                            sharpness = functions.sharpnessLC0(wdl)
                            
                            # Add to player metrics
                            player_metrics[current_player]["accuracies"].append(accuracy)
                            player_metrics[current_player]["sharpness"].append(sharpness)
                        
                        prev_eval = current_eval
    
    # Calculate average metrics for each player
    for player, metrics in player_metrics.items():
        metrics["avg_accuracy"] = statistics.mean(metrics["accuracies"]) if metrics["accuracies"] else 0
        metrics["avg_sharpness"] = statistics.mean(metrics["sharpness"]) if metrics["sharpness"] else 0
        metrics["total_moves"] = len(metrics["accuracies"])
    
    return player_metrics


def analyze_accuracy_sharpness_correlation(player_performance):
    """
    Analyze the correlation between accuracy and sharpness for each player.
    """
    correlations = {}
    for player, player_metrics in player_performance.items():
        accuracies = player_metrics["accuracies"]
        sharpness = player_metrics["sharpness"]
        
        # Calculate Pearson correlation coefficient
        correlation, _ = stats.pearsonr(accuracies, sharpness)
        correlations[player] = correlation
        
        # Create scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(sharpness, accuracies, alpha=0.5)
        plt.title(f"{player}: Accuracy vs Sharpness")
        plt.xlabel("Sharpness")
        plt.ylabel("Accuracy")
        
        # Add trend line
        z = np.polyfit(sharpness, accuracies, 1)
        p = np.poly1d(z)
        plt.plot(sharpness, p(sharpness), "r--", alpha=0.8)
        
        # Add correlation coefficient to plot
        plt.text(0.05, 0.95, f"Correlation: {correlation:.2f}", transform=plt.gca().transAxes)
        
        plt.savefig(f"accuracy_sharpness_{player.replace(' ', '_')}.png")
        plt.close()
        
        # Calculate moving averages
        window = min(50, len(accuracies) // 10)  # Use 10% of data points or 50, whichever is smaller
        acc_ma = np.convolve(accuracies, np.ones(window), 'valid') / window
        sharp_ma = np.convolve(sharpness, np.ones(window), 'valid') / window
        
        # Plot moving averages
        plt.figure(figsize=(10, 6))
        plt.plot(acc_ma, label="Accuracy MA")
        plt.plot(sharp_ma, label="Sharpness MA")
        plt.title(f"{player}: Moving Averages of Accuracy and Sharpness")
        plt.xlabel("Move Number")
        plt.ylabel("Value")
        plt.legend()
        plt.savefig(f"moving_averages_{player.replace(' ', '_')}.png")
        plt.close()
    
    return correlations


def plot_accuracy_over_moves(player_performance):
    """
    Plot accuracy over move numbers for each player.
    """
    for player, player_metrics in player_performance.items():
        accuracies = player_metrics["accuracies"]
        move_numbers = range(1, len(accuracies) + 1)
        
        plt.figure(figsize=(12, 6))
        plt.plot(move_numbers, accuracies, marker='o', markersize=3, linestyle='-', linewidth=1)
        plt.title(f"{player}: Accuracy over Moves")
        plt.xlabel("Move Number")
        plt.ylabel("Accuracy")
        plt.ylim(0, 100)  # Assuming accuracy is between 0 and 100
        
        # Add a horizontal line for the average accuracy
        avg_accuracy = player_metrics["avg_accuracy"]
        plt.axhline(y=avg_accuracy, color='r', linestyle='--', label=f'Avg Accuracy: {avg_accuracy:.2f}')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"accuracy_over_moves_{player.replace(' ', '_')}.png", dpi=300)
        plt.close()


def plot_accuracy_and_sharpness_over_moves(player_performance):
    """
    Plot accuracy and sharpness over move numbers for each player.
    """
    for player, player_metrics in player_performance.items():
        accuracies = player_metrics["accuracies"]
        sharpness = player_metrics["sharpness"]
        move_numbers = range(1, len(accuracies) + 1)
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot accuracy
        color = 'tab:blue'
        ax1.set_xlabel('Move Number')
        ax1.set_ylabel('Accuracy', color=color)
        ax1.plot(move_numbers, accuracies, color=color, marker='o', markersize=3, linestyle='-', linewidth=1, label='Accuracy')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(0, 100)  # Assuming accuracy is between 0 and 100
        
        # Add a horizontal line for the average accuracy
        avg_accuracy = player_metrics["avg_accuracy"]
        ax1.axhline(y=avg_accuracy, color=color, linestyle='--', label=f'Avg Accuracy: {avg_accuracy:.2f}')
        
        # Create a second y-axis for sharpness
        ax2 = ax1.twinx()
        color = 'tab:orange'
        ax2.set_ylabel('Sharpness', color=color)
        ax2.plot(move_numbers, sharpness, color=color, marker='s', markersize=3, linestyle='-', linewidth=1, label='Sharpness')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Add a horizontal line for the average sharpness
        avg_sharpness = player_metrics["avg_sharpness"]
        ax2.axhline(y=avg_sharpness, color=color, linestyle='--', label=f'Avg Sharpness: {avg_sharpness:.2f}')
        
        plt.title(f"{player}: Accuracy and Sharpness over Moves")
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"accuracy_sharpness_over_moves_{player.replace(' ', '_')}.png", dpi=300)
        plt.close()


if __name__ == '__main__':
    input_folder = 'out/pgn/0929OlympiadAusWomen/comment/done'
    
    # Step 1: Read and print all player names
    players = get_all_players(input_folder, FOCUS_PLAYERS)
    print("Players analyzed in the tournament:")
    for player in players:
        print(f"- {player}")
    print(f"Total number of players analyzed: {len(players)}")
    
    # Step 2: Calculate and display player scores
    scores = get_player_scores(input_folder, FOCUS_PLAYERS)
    print("\nPlayer Scores:")
    for player, (games, points) in sorted(scores.items(), key=lambda x: x[1][1], reverse=True):
        print(f"{player}: {points}/{games} points")
    
    # Step 3: Analyze move situations
    move_situations = get_move_situation(input_folder, FOCUS_PLAYERS)
    print("\nMove Situations:")
    for player, situations in sorted(move_situations.items(), key=lambda x: x[1][0], reverse=True):
        total, much_better, slightly_better, equal, slightly_worse, much_worse = situations
        print(f"{player}:")
        print(f"  Total moves: {total}")
        print(f"  Much better: {much_better} ({much_better/total*100:.2f}%)")
        print(f"  Slightly better: {slightly_better} ({slightly_better/total*100:.2f}%)")
        print(f"  Equal: {equal} ({equal/total*100:.2f}%)")
        print(f"  Slightly worse: {slightly_worse} ({slightly_worse/total*100:.2f}%)")
        print(f"  Much worse: {much_worse} ({much_worse/total*100:.2f}%)")
        print()
    
    # Step 4: Analyze player performance
    player_performance = analyze_player_performance(input_folder, FOCUS_PLAYERS)
    print("\nPlayer Performance Analysis:")
    for player, player_metrics in sorted(player_performance.items(), key=lambda x: x[1]["avg_accuracy"], reverse=True):
        print(f"{player}:")
        print(f"  Total moves: {player_metrics['total_moves']}")
        print(f"  Average accuracy: {player_metrics['avg_accuracy']:.2f}")
        print(f"  Average sharpness: {player_metrics['avg_sharpness']:.2f}")
        print()
    
    # Step 5: Analyze correlation between accuracy and sharpness
    correlations = analyze_accuracy_sharpness_correlation(player_performance)
    print("\nCorrelations between Accuracy and Sharpness:")
    for player, correlation in sorted(correlations.items(), key=lambda x: x[1], reverse=True):
        print(f"{player}: {correlation:.2f}")
        print(f"  Total moves: {player_metrics['total_moves']}")
        print(f"  Average accuracy: {player_metrics['avg_accuracy']:.2f}")
        print(f"  Average sharpness: {player_metrics['avg_sharpness']:.2f}")
        print()
    
    # New Step: Plot accuracy and sharpness over moves for each player
    plot_accuracy_and_sharpness_over_moves(player_performance)