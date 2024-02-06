import chess
from chess import engine, pgn
import numpy as np
import matplotlib.pyplot as plt


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
    return 103.1668 * np.exp(-0.04354 * (winPercentBefore - winPercentAfter)) - 3.1669


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


def gameAccuracy(gamesFile: str, engine: engine, depth: int, outfile: str) -> list:
    """
    This function goes through the games in a PGN file and returns the accuracies of the games.
    gamesFile: str
        The path to the PGN file
    engine: engine
        A configured chess engine
    depth: int
        The depth used on every move
    outfile: str
        The path to the output file
    return -> list
        A list of all the accuracies of the games
    """
    gameAccuracies = list()
    gameNR = 1
    with open(gamesFile, 'r') as pgn:
        while (game := chess.pgn.read_game(pgn)):
            print(f'Starting with game {gameNR}...')
            print(f'{game.headers["White"]} - {game.headers["Black"]}')
            gameNR += 1

            acc = (list(), list())

            board = game.board()
            for move in game.mainline_moves():
                c = board.turn
                cp1 = engine.analyse(board, chess.engine.Limit(depth=depth))["score"].pov(c).score()
                board.push(move)
                if not board.is_game_over():
                    cp2 = engine.analyse(board, chess.engine.Limit(depth=depth))["score"].pov(c).score()
                    if cp1 != None and cp2 != None:
                        if c:
                            acc[0].append(accuracy(winP(cp1), winP(cp2)))
                        else:
                            acc[1].append(accuracy(winP(cp1), winP(cp2)))
                    else:
                        print(cp1, cp2)
            print(sum(acc[0])/len(acc[0]))
            gameAccuracies.append((sum(acc[0])/len(acc[0]), sum(acc[1])/len(acc[1])))
    print(gameAccuracies)
    for acc in gameAccuracies:
        print(acc, file=open(outfile, 'a+'))
    engine.quit()
    return gameAccuracies


def showAccuracy(depths: list):
    """
    This function creates a scatter plot for the accuracies at different engine depths
    depths: list
        A list where each entry is a list of accuracies at a given depth
    """
    for depth in depths:
        x = [d[0] for d in depth]
        y = [d[1] for d in depth]
        plt.scatter(x,y)
        print(sum(x)/len(x), sum(y)/len(y))
    plt.show()


if __name__ == '__main__':
    d10 = [(95.95515426755497, 101.5784212189411), (96.37865789744741, 97.22954281464033), (98.38756087451175, 99.04630383773384), (96.85577627848929, 98.52338796790833), (98.20249027180895, 98.01227304107384), 
            (94.96700546292065, 99.74245049718397), (96.66831626221234, 95.33889855782951), (98.83246367379712, 98.66801675351567), (98.65557708422953, 98.82122933491036), (98.60912143296306, 99.29033587741333), 
            (93.13321699788986, 95.62305506263918), (102.75872474926997, 98.8980196427576), (98.52590139217293, 99.3649125033579), (93.99261696742889, 96.12761446820087), (94.40610636207789, 94.78085186972073), 
            (94.81012951000919, 97.79981197547525), (97.71725215043458, 97.78217500924664), (97.35023333852963, 97.51137028074426), (94.76532282598778, 99.02731772869451), (97.24788484775928, 97.27045697462057), 
            (88.57447340860871, 92.89482114028432), (97.28300992981958, 95.01987190079637), (95.54834303609597, 98.12936317845708), (96.44585626283504, 96.54503368789044), (97.7058780494648, 96.63498767147534), 
            (96.1814303819038, 95.51261011280539), (98.45563234654742, 98.61329048326988), (98.46489939015703, 98.9910057526693), (98.11863334435354, 97.48065490008389), (97.29110276278745, 97.65362876400502), 
            (96.70562111789252, 94.74716050426986), (97.23221545932537, 97.35294477840226), (98.64592755861214, 98.16098454618356), (97.51207760958775, 95.8028916555137), (97.74730703987576, 97.14546718275479), 
            (97.77865759284047, 98.37326091816668), (94.94893433654336, 96.56342325003563), (97.33577079434454, 97.60972330512614), (97.8497236101358, 97.11015855460532), (98.77516557977428, 96.9341834462247), 
            (98.07660155154488, 98.92020784505065), (96.00512735223076, 94.44979986217835), (97.01056858127477, 96.89252984093179), (98.5280348565524, 98.98471880960255), (94.93341934244442, 94.44976879257372), 
            (98.81101639695291, 97.28716902531697), (99.34590669550639, 97.4141073699636), (96.93851587036406, 96.75266711893761), (97.61740737362193, 98.38197315490095), (98.77411501104248, 98.7968097542721), 
            (96.12456468003325, 96.63265264413374), (96.94873865636535, 97.52159317404988), (98.36498530302202, 97.05065385964522), (97.24668132128554, 97.50450723260167), (97.0567155251777, 96.64730834896005), 
            (95.78687651296651, 96.18067471670027), (95.3041671871881, 95.30733765473865), (94.4932319568596, 95.99519525736133), (98.64663044080146, 99.0835069876172), (95.52478993174253, 94.286637669488), 
            (96.13124829454672, 92.96762982410276), (97.98433193215517, 96.89531021367648), (93.46598716489811, 92.88163660394122), (97.7421707593727, 97.67436717195952), (94.06530064153631, 94.95011609780121), 
            (99.09565929313366, 98.91493640021413), (98.38061876700031, 98.32392204392087), (95.14510726327033, 97.80338019639633), (98.16092696208875, 98.7880255088856), (97.87855853453352, 94.69732388016564), 
            (95.81735150065202, 95.51556822324102), (103.1648192471869, 96.49702030400589), (98.2169669045017, 93.52850346090588), (95.33949055369249, 95.66433831719593), (97.77684985148282, 98.95462666141383), 
            (98.56043716767832, 98.24782596819405), (96.60120329006773, 99.04924300648504), (98.12448202728703, 97.83277044654898), (96.49790725120063, 90.44294788198192), (93.14834055472097, 98.2530283727619), 
            (96.5477136458391, 96.54772008796988), (96.38106421455078, 93.72925700493472), (93.30479163722713, 97.90663811618046), (94.66122966294901, 96.37396938299442), (98.5282335972071, 98.66182287833855), 
            (94.36006643619473, 98.64483788294343), (98.36002644280781, 93.53519912909582), (96.77634879036088, 96.46719866860778), (98.19702749819655, 94.26747143362911), (98.72367913956289, 99.13882097464159), 
            (97.88118263667903, 98.04745405623257), (93.07706810046021, 93.5784416285917), (95.13128892208216, 97.0453157404971), (93.90772032821823, 96.97186532931933), (95.5767834867203, 98.35699095253712), 
            (96.95245406634865, 94.35097707917544), (96.67828001540612, 97.57680855068834), (94.88423448875939, 92.74188348983348)]
    d12 = [(96.36669934608379, 99.18899433414045), (95.85937834258415, 98.04859537494907), (98.33878913715378, 98.67387476779236), (97.22986568591067, 99.73727170770586), (97.66823146492266, 98.82207119073138), 
            (94.1278514817373, 97.90521786287155), (97.0799172760237, 96.21466474672917), (98.97093992890795, 99.79530392154356), (99.22593146525995, 99.09653884318465), (98.76427400707884, 99.33536280641896), 
            (93.27517897652874, 94.38986147804489), (98.67471453060288, 96.46958599776009), (98.77795571112976, 98.36233111984416), (93.91999253197193, 97.79980177112832), (95.29245370256153, 96.84071928952149), 
            (93.85936301501735, 97.06078478331175), (97.87650361969774, 98.33414547934744), (97.09287287222588, 97.1496238724623), (93.66404110067678, 98.96065888226565), (98.18274734552024, 98.12004862597936), 
            (90.07139212799677, 95.52728989488729), (96.67765485179689, 94.46740530741783), (96.18397257675461, 98.74077128108102), (95.46792253567753, 96.91681480232238), (97.56840722534113, 97.26725248564142), 
            (97.62025414603768, 97.27709424607701), (98.47004221644086, 99.78994096496331), (98.49387455799751, 99.49481175440104), (98.06978838356231, 98.64557243793791), (98.02403088347242, 97.85365458981939), 
            (97.67388887995146, 95.93300526074151), (99.00507264782247, 97.85054364672074), (98.62015795260044, 98.08489909778696), (98.51255987689309, 95.04621896818476), (97.96758994206623, 97.66613514703681), 
            (98.0469603534462, 97.33326366246128), (95.3554402314761, 97.21809015375952), (97.02376310596843, 96.85320556075399), (98.05170349713158, 98.28985990943862), (98.69171556210594, 96.474796272633), 
            (98.62605787679449, 99.22670438494171), (97.25874684961954, 94.3163666171625), (97.56038341530015, 98.16147916043067), (99.19950373014458, 99.78737603688842), (99.86506950796232, 97.31409462315203), 
            (98.33141380627544, 94.74940304189164), (99.99339324071902, 98.32336210562313), (97.26961492391916, 97.24611937812412), (98.1457024836636, 98.72541725904144), (98.49121591246863, 98.1760992368661), 
            (96.729632661012, 97.2532019732562), (97.39498584388502, 97.84229486375942), (98.8272495723625, 95.70724376897815), (98.5578659586115, 98.84871296830165), (97.41793804645168, 96.17186700238571), 
            (97.08108075191939, 96.30283092296779), (97.31682016849, 95.37376076129335), (93.72260629394344, 95.33277061690168), (99.14438613621768, 98.82229955810855), (94.61613471519337, 93.84493772082537), 
            (97.9137709164844, 92.68877844169312), (96.29450421302012, 94.16637059610187), (95.46235413129398, 91.61794572293894), (97.2927728123865, 97.40058646139653), (93.04704078129971, 95.82948332416986), 
            (98.97319877623966, 99.04745500470212), (98.13916717096197, 97.73477228318305), (95.61211236351592, 95.24359780803513), (98.0953959707336, 98.88808071755065), (96.88986591611165, 95.55980885197283), 
            (98.16491943933357, 97.85602236267849), (97.64167640241538, 94.64455315585506), (96.63080674792238, 93.79759975788897), (94.05158677114964, 94.95895203864166), (97.42367965676569, 98.38587933180791), 
            (98.09851224489678, 98.18889714163173), (95.72611493609315, 98.93319679564247), (97.92633531826736, 98.44888233024304), (96.67058805836442, 91.2573249774008), (94.36262644185426, 98.50350073019672), 
            (97.9910720579146, 96.56452189432817), (95.03179048166488, 93.45777233768004), (93.46584502306571, 97.32417861599079), (96.09861917785169, 96.5307933106885), (98.31541745041825, 98.2424651414119), 
            (92.89251656847989, 96.85663367167164), (99.50802732686527, 94.86075432269077), (97.61700539303597, 95.6937859050279), (99.63934144800601, 95.02105459106315), (99.05767278451513, 98.82439878939891), 
            (97.9081712588277, 98.60840938094387), (94.33029428538067, 94.10784786269369), (96.54159416662328, 97.56596208282933), (93.56334253119611, 95.16333328151929), (93.62228317434, 96.57667208874186), 
            (96.14265992153523, 92.48971799549525), (96.8906761497594, 96.9396927518038), (95.23697003050333, 93.00836541135463)]
    d15 = [(94.66637668811369, 98.32039418608035), (95.97679266082245, 99.32298363882883), (98.11370679381004, 98.00675068189923), (96.79000366892241, 98.65922950086016), (99.04446869903465, 99.42871750002233), (95.31273482370672, 99.57380674024984), (97.7442809500429, 97.01719699589258), (98.8571306685276, 99.62808608434426), (99.09442383929458, 99.07587194397605), (99.72806358810014, 100.2546980855651), (93.88984110933447, 96.39267145534929), (98.79983812561724, 97.47310291585757), (99.4151309194444, 99.9590195946206), (94.42692773841645, 97.04732554746502), (94.0718754819205, 96.79705727615224), (94.28342859465694, 98.35797533176675), (98.27461823019522, 98.50014974062942), (97.85159923893667, 97.47806907068008), (93.69067036494309, 98.30898331131512), (98.49765580620031, 98.54340625967414), (91.721186668405, 94.5260102208558), (97.23734952096531, 96.20886584751089), (96.0153791248742, 98.48931655367991), (97.03158980289288, 96.22197784262656), (97.55854204485524, 96.59094744524097), (98.28718772565517, 98.55369197381106), (97.98570879504821, 99.38271924219791), (98.77592761142573, 99.07548639294967), (98.6615799346813, 98.86826630358367), (97.94687834303795, 99.08240994849156), (97.15995276765729, 95.89998312634243), (98.57745147314864, 97.6382409156967), (99.08276225876688, 99.05470361721251), (96.59413382486123, 94.65130942064155), (97.82527953354273, 98.17048519429585), (98.42682537226055, 97.69197959764503), (95.07690844718721, 98.55138256017692), (97.8036180036916, 98.39768252351479), (97.21160545951815, 96.87103937154009), (98.95085876455086, 95.80963715585136), (99.34036019485389, 99.60807152144312), (96.63005603826218, 94.38413146699848), (97.95114600254912, 98.46874881176095), (98.71730031680539, 99.26588629014866), (95.39750752529724, 94.54985164382353), (96.8969531445407, 95.80194850905092), (99.30443986240675, 97.25252650011558), (97.58557065173761, 98.03647025406633), (98.02574729203283, 98.64915794378062), (98.51779413832394, 99.03214895697847), (97.15734516608508, 97.04418056769218), (98.61847574095856, 98.3754955659312), (98.64355943725317, 96.24530062395435), (98.22778291658749, 98.03573681219488), (97.25569273793138, 96.67787101151778), (96.96733906554395, 96.9089773498128), (98.18556604002227, 92.66245800721111), (93.28827837493711, 95.02030144539411), (98.61061605695194, 99.31406768749731), (96.52412677225124, 93.87401428502226), (98.82300328579588, 95.96767643896146), (98.08153608524786, 93.90178472674225), (95.65797913859787, 91.41406374011672), (97.63937071794315, 97.55327941208131), (92.99354250162327, 95.81282426005816), (99.32195419233989, 99.21630621475722), (98.19913685984046, 98.43415581445474), (94.19717665975234, 96.44164467690736), (99.10522679404565, 99.36570144527747), (97.14508566165576, 93.76855163981601), (97.58115088740286, 98.16928618822398), (96.78241949663874, 92.66505846679658), (97.08769383164716, 92.80378577264389), (94.56567286194638, 94.96352307435689), (98.2197944306208, 98.10390141564086), (97.85581451604293, 98.12340074809534), (94.72550247718404, 98.84744227957637), (98.13228659767843, 98.30247941764638), (96.3157280894906, 91.60964133021216), (94.62620400763063, 99.54674554596134), (97.8929699964203, 96.94589630277805), (95.25401171125529, 92.98630522164737), (93.9551589604412, 98.25528657986267), (93.93341470067018, 95.47561006548072), (98.32365762939318, 99.13382558834222), (93.01865157459488, 95.7759381801198), (99.3384522754625, 94.22355177909222), (96.80372587037398, 94.63612702738713), (96.0991677163052, 92.82402699019812), (98.9011745052051, 98.87014825848968), (97.867907863445, 98.42450071964983), (93.57354525004727, 93.62306366883224), (95.51835713588068, 97.06737836566103), (92.37263609621931, 94.2991376301113), (92.12722561819815, 96.15825497393243), (96.50156601755141, 93.99160551844292), (96.32341193886793, 96.40003774447466), (96.09344880048496, 93.12113895542268)]
    d20 = [(94.36414979350279, 98.83360826795757), (95.8330639729624, 98.52157590532047), (99.17805020711272, 99.1702905073724), (96.35430796574316, 97.48053557447754), (99.19710615481618, 99.68103340296095), (94.20905245293535, 98.90052170762954), (98.0800202504352, 98.18936700343126), (99.11491865229411, 99.29354371477726), (99.25539605926625, 99.16121489265541), (100.00089582038194, 99.00996942326469), (92.00756229273662, 96.47315686150758), (99.6602426175978, 96.59789409720122), (99.26706827607507, 99.04463125146381), (94.38444268773848, 97.0641189066836), (93.61744349104642, 96.17170773914862), (94.13411696598627, 97.1084819688329), (97.78595656004804, 98.14995441941451), (97.1146615321475, 97.5197385814806), (93.26450192967297, 99.29417872770873), (98.1390680354119, 98.50525285821207), (90.49007979580365, 92.56011179876015), (97.71957129800208, 96.63502583773145), (96.46738523115464, 98.45133315164588), (96.5261941284255, 96.93697620050084), (97.18562729451817, 95.3658938641245), (99.14721815094467, 98.65456603124521), (99.60231221804715, 99.35429429074577), (98.87364527419871, 98.92493218211386), (98.44053530363873, 98.90128194946266), (98.78864757456707, 99.26986078066143), (97.05688399352672, 95.56345900263953), (98.02589679610661, 98.24657387780636), (98.98157490933055, 98.49598581380457), (97.23839120933881, 94.38913451641275), (98.11793202889537, 98.5571881255367), (97.58352905978863, 97.90386873724584), (95.39144511732096, 96.79488094821994), (98.05898391942105, 98.22538200211244), (96.98856409992253, 96.61594326130887), (99.20196537051834, 95.54452367633769), (99.74902767287976, 99.39931495963016), (97.03863117647208, 93.08477007034953), (98.1863597792099, 98.86663081996645), (99.28915349457495, 99.74057565238071), (96.35363361590932, 94.93479123024582), (98.09189406641005, 97.43224697854296), (99.08924460167407, 97.75306347321845), (98.40499909523467, 97.80445092191641), (98.47516478571148, 98.66621334391073), (98.75882411344891, 98.6972212546305), (96.66755036362605, 96.47213635693356), (98.47827732047426, 98.26251229382748), (98.87042759106052, 96.60555293110984), (98.55019959270192, 98.39226664507612), (97.30838533136193, 95.78885056107433), (96.73625069675793, 97.25833638713709), (96.6409324268691, 93.45791394948829), (94.70961023529239, 96.11810765296347), (99.00471609469825, 99.76209821671242), (96.26852694882027, 93.17695310981908), (96.6363695443722, 92.93684420472891), (98.27424226671106, 95.52631618831626), (96.6551876026573, 92.62608082765244), (97.33932606630559, 97.37166432514604), (93.79695903378358, 95.61846653096681), (99.29489146414213, 99.31148166783764), (98.92916743366973, 99.16562489194006), (94.55768233776097, 96.90644092909913), (99.44260432154486, 99.17337184509162), (95.5796450551894, 93.70227070129167), (97.05308615262692, 97.73852158235331), (99.34496650065545, 94.08674879022351), (97.31441174171357, 92.02844447439305), (93.92958253367543, 94.84460520336958), (97.48508586324078, 97.4853352582176), (98.37339273765507, 97.99636972933124), (96.40552267663392, 99.26685885426896), (98.43825825504074, 98.93359044986106), (96.24377027338899, 92.59257997145863), (94.26907055771035, 98.35153022999147), (97.91820108786449, 97.00149070662438), (94.73444918309201, 92.60411734643759), (94.10683039492383, 98.52012594739517), (94.86200755129892, 96.34458949216022), (98.89822844914859, 99.70496242826427), (92.01872305993453, 95.2545191745191), (99.76251961001931, 95.32476605013805), (96.74452020579956, 93.99940339768239), (95.87319933903414, 92.07726642108392), (99.32154691858071, 98.18923885720174), (97.79053433483604, 98.60200997875656), (93.77436602896465, 93.50799973363536), (95.87677481305619, 97.16938983867577), (91.49703078376017, 92.6438212265549), (93.1081801121899, 97.66607844610513), (96.05567351498817, 92.9871861895565), (96.84514667252803, 97.06152029024125), (94.81799389850738, 92.78271510423585)]
    carlsend10 = [(89.70687408818246, 93.24081206548804), (90.38021211968287, 84.23217575110918), (90.19414882450239, 93.54061783131742), (96.95838651651427, 94.12841134501355), (87.23182337449384, 90.65645020289966), (96.30927822905569, 95.42496230923246), (93.10196036305574, 96.1100049726857), (98.8485817059284, 93.56328822078453), (93.98229580310039, 93.31285747768953), (95.84847789569615, 91.7589600028934), (89.80176080943176, 92.56096858331472), (90.4119148725837, 90.16670722309112), (94.86774994818805, 92.48419135868146), (96.67868498310051, 96.70161878637698), (94.3976724271077, 92.37149076307429), (95.80893616387431, 92.26537199904074), (96.0145085756888, 92.94681153556103), (94.38391054506035, 96.48969713191661), (92.29540189693981, 96.37105843080812), (92.48053127697982, 89.0183396085892), (95.29260809905955, 94.84738079742893), (90.1970399468063, 91.37913247408633), (92.6345913569109, 91.0945966632684), (87.90998721346226, 94.58604929627147), (89.752674975098, 90.7455541672734), (92.20296574874273, 89.778343121942), (89.89474609980087, 95.46339584944415), (90.29865918349566, 91.47061024957388), (93.84876555265987, 90.2486832631118), (88.3855065209348, 86.55845364476238), (94.78518568273051, 91.21294001136826), (98.97282150309503, 95.41190911172497), (91.4217066954583, 91.7301326824071), (90.85275398144876, 94.31629887150581), (98.42465899346328, 93.73252085940285), (92.51422215913723, 94.80747568191234), (96.11421413569259, 91.77918016869064), (96.59340661079789, 93.78149681731534), (94.72952028071876, 92.2427406026994), (96.63074251308336, 89.59478880632582), (90.88158425731048, 92.63167163558803), (88.7109901304608, 87.06108822864604), (96.01669531013913, 97.02901767712598), (90.34764426088446, 96.42373180575184), (89.62243238911992, 94.06772024082647), (92.24461065333709, 95.3436950931275), (88.81781389468833, 88.53866832590947), (95.40230592046673, 93.9027047640755), (94.56956791017325, 90.38594032219457), (84.46149200936216, 90.21102031532688)]
    carlsend20 = [(88.04783490067047, 91.40839331640134), (89.80102409768338, 82.36219367467092), (89.50804180125596, 92.2177103196664), (97.2250633575125, 93.73608830814617), (88.23554968011874, 91.38868732190109), (96.21563818822875, 96.6651279771059), (92.6373497685296, 96.4627607465542), (98.77836742060298, 95.10046850712378), (93.17279877955956, 91.57637536246213), (95.48919317875774, 91.99666831888861), (90.30237667260326, 93.52828912085585), (95.21904483068859, 95.14936237693217), (94.584245151849, 91.3010386186352), (97.04991565884514, 97.13103505831927), (94.3947965423571, 93.19949970601922), (96.7200768973612, 93.64713757336818), (95.28641469841311, 89.94111622567016), (94.59137342611795, 96.00057363434688), (92.47575973933439, 96.98698203154012), (95.05282940852061, 91.09618069196739), (94.73025080845603, 95.38964372730914), (88.04067318024357, 89.67290248718304), (94.32681960651927, 90.8683504950249), (86.80733767933401, 96.24249900704929), (87.63675894125495, 90.74588026574622), (94.33707160868605, 91.10243894724955), (88.39491996323201, 94.9022080944446), (87.86554337871972, 89.66756087292346), (89.21978047343252, 86.99085464096123), (85.05422122117493, 83.96116336952005), (94.78271009933908, 90.90183352568783), (98.98621354124118, 94.5261992409396), (91.02724836502077, 91.80134378866616), (92.41957065865856, 95.1989412491127), (98.4823579467382, 92.87934596788654), (94.38673898062261, 96.0133257200616), (96.55628285546236, 93.5365534644467), (91.94070833831775, 90.3421976500906), (95.05215980688061, 92.06807223114946), (97.5629691697466, 91.37910096342948), (90.88882292201977, 92.66465611559234), (87.64770357263262, 84.66576782145692), (96.74078377762245, 96.78465541222394), (90.39487311747443, 94.30906845342989), (91.53141882139454, 92.78942632007224), (93.29816932456737, 95.64931178476049), (86.46124964103501, 85.50697271907268), (92.0773356145725, 91.44505441382573), (93.89911650062486, 88.36673917368996), (84.53360614900542, 86.96756933132463)]
    sf = configureEngine('stockfish', {'Threads': '10', 'Hash': '8192'})
    wijk = '../resources/wijkMasters2024.pgn'
    carlsen = '../resources/carlsenBlitz.pgn'
    outfile = '../out/accuracy'
    # gameAccuracy(carlsen, sf, 20, outfile)
    showAccuracy([carlsend10, carlsend20])
    # showAccuracy([d10, d12, d15, d20])
