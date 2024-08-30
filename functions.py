from chess import engine, pgn, Board
import chess
import numpy as np
import subprocess
import re
import os

def configureEngine(engineName: str, uci_options: dict) -> engine:
    """
    This method configures a chess engine with the given UCI options and returns the 
    engine.
    engineName: str
        The name of the engine (or the command to start the engine)
    uci_optins: dict
        A dictionary containing the UCI options used for the engine
    return -> engine
        A configuered chess.engine
    """
    eng = engine.SimpleEngine.popen_uci(engineName)
    for k, v in uci_options.items():
        eng.configure({k:v})

    return eng

def formatWDL(wdl: engine.Wdl) -> list:
    """
    This function takes an engine.wdl and turns it into a list of the WDL from
    white's perspective (0-1000 range)
    wdl: wdl
        The engine.Wdl
    return -> list
        A list containing the W,D,L as integers ranging from 0 to 1000
    """
    wl = []
    wdl_w = engine.PovWdl.white(wdl)
    for w in wdl_w:
        wl.append(w)
    return wl


def sharpnessOG(wdl: list) -> float:
    """
    This function calculates the sharpness based on my own formula
    wdl: lsit
        The WDL
    return -> float
        The sharpness
    """

    w, d, l = wdl

    if min(w, l) < 100:
        return 0

    wd = w - d
    ld = l - d

    return min(w, l)/50 * (1 / (1+np.exp(- (w+l)/1000))) * (333/(d+1))

def sharpnessLC0(wdl: list) -> float:
    """
    This function calculates the sharpness score based on a formula posted by the
    LC0 team on Twitter.
    wdl: list
        The WDL as a list of integers ranging from 0 to 1000
    return -> float
        The shaprness score based on the WDL
    """
    # max() to avoid a division by 0, min() to avoid log(0)
    W = min(max(wdl[0]/1000, 0.0001), 0.9999)
    L = min(max(wdl[2]/1000, 0.0001), 0.9999)

    # max() in order to prevent negative values
    # I added the *min(W, L) to reduce the sharpness of completely winning positions
    # The *4 is just a scaling factor
    return (max(2/(np.log((1/W)-1) + np.log((1/L)-1)), 0))**2


def accuracy(winPercentBefore: float, winPercentAfter: float) -> float:
    """
    This function returns the accuracy score for a given move. The formula for the
    calculation is taken from Lichess
    winPercentBefore: float
        The win percentage before the move was played (0-100)
    winPercentAfter: float
        The win percentage after the move was payed (0-100)
    return -> float:
        The accuracy of the move (0-100)
    """
    return min(103.1668 * np.exp(-0.04354 * (winPercentBefore - winPercentAfter)) - 3.1669, 150)


def winP(centipawns: int) -> float:
    """
    This function returns the win percentage for a given centipawn evaluation of a position.
    The formula is taken from Lichess
    centipawns: int
        The evaluation of the position in centipawns
    return -> float:
        The probability of winning given the centipawn evaluation
    """
    return 50 + 50*(2/(1+np.exp(-0.00368208 * centipawns)) - 1)


def readComment(node, wdl: bool, cp: bool) -> tuple:
    """
    This function takes a game node from a PGN with evaluation comments and returns the evaluation.
    Comment structure: [w, d, l]; cp
    node:
        The game node
    wdl: bool
        If the comment contains a WDL evaluation
    cp: bool
        If the comment contains a centipawn evaluation
    return -> tuple
        A tuple containing the WDL and/or CP evaluations
    """
    if not (wdl or cp):
        return None

    if not node.comment:
        return None

    if wdl and not cp:
        wdlList = [ int(w) for w in node.comment.replace('[', '').replace(']', '').strip().split(',') ]
        return (wdlList)
    if cp and not wdl:
        return (int(float(node.comment)))
    sp = node.comment.split(';')
    wdlList = [ int(w) for w in sp[0].replace('[', '').replace(']', '').strip().split(',') ]
    return (wdlList, int(float(sp[1])))

def modifyFEN(fen: str) -> str:
    """
    This function takes a standard FEN string and removes the halfmove clock and the fullmove number
    """
    fenS = fen.split(' ')
    if not fenS[-2].isnumeric():
        return fen
    modFen = fenS[0]
    for s in fenS[1:-2]:
        modFen = f'{modFen} {s}'
    return modFen


def getNumberOfGames(fen: str) -> int:
    """
    This function returns the number of games in the database with the given position.
    """
    script = '~/coding/chessProjects/novelties/searchPosition.tcl'
    db = '/home/julian/chess/database/gameDB/novelties'
    output = str(subprocess.run(['tkscid', script, db, fen], stdout=subprocess.PIPE).stdout)
    print(output)
    games = re.findall(r'\d+', output)[0]
    return int(games)

def relativePathToAbsPath(relative_path : str) -> str:
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    return absolute_path + relative_path


def analysisCP(position: Board, sf: engine, timeLimit: int):
    """
    This function analyses a given position with Stockfish and returns the centipawn score.
    position: Board:
        The position to analyse
    sf:engine
        Stockfish as a configured chess engine
    timeLimit:int
        The time in seconds spent on the position
    return -> str
        The centipawn score
    """
    # TODO: fix this somehow
    # sf = configureEngine('stockfish', {'Threads': '10', 'Hash': '8192'})
    if position.is_game_over():
        return None

    info = sf.analyse(position, chess.engine.Limit(time=timeLimit))
    # sf.quit()
    return info


def analysisWDL(position: Board, lc0: engine, limit: int, time: bool = False):
    """
    This function analyses a given chess position with LC0 to get the WDL from whtie's perspective.
    position:Board
        The position to analyse
    lc0:engine
        LC0 already as a chess engine
    limit:int
        The limit for the analysis, default is nodes, but time can also be selected
    time:bool = False
        If this is true, the limit will be for the time in seconds
    return -> str
        The formated WDL
    """
    # The analyse() method gives an error if the game is over (i.e. checkmate, stalemate or insuffucient material)
    if position.is_game_over():
        return None
    
    if time:
        info = lc0.analyse(position, chess.engine.Limit(time=limit))
    else:
        info = lc0.analyse(position, chess.engine.Limit(nodes=limit))
    return info




def analysisCPnWDL(position: Board, lc0: engine, nodes: int, sf: engine) -> tuple:
    """
    This function analyses a position both with LC0 and Stockfish. It returns the WDL and CP infos as tuple.
    """
    if position.is_game_over():
        return None

    # Defining Stockfish here is not ideal, but it's the easiest way right now
    # sf = configureEngine(r'K:\github\stockfish-windows-x86-64\stockfish\stockfish-windows-x86-64.exe', {'Threads': '10', 'Hash': '8192'})
    iLC0 = lc0.analyse(position, chess.engine.Limit(nodes=nodes))
    iSF = sf.analyse(position, chess.engine.Limit(time=4))

    return (iLC0, iSF)


def formatInfo(infoLC0 = None, infoSF = None) -> str:
    """
    This function takes the info from an engine analysis by LC0 or stockfish and returns the WDL/CP as string
    """
    evaluation = ""
    if infoLC0:
        wdl = []
        wdl_w = engine.PovWdl.white(infoLC0['wdl'])
        for w in wdl_w:
            wdl.append(w)
        evaluation = str(wdl)
    if infoSF:
        if infoLC0:
            evaluation += ';'
        cp = str(infoSF['score'].white())
        if '#' in cp:
            if '+' in cp:
                cp = 10000
            else:
                cp = -10000
        evaluation += str(cp)
    return evaluation