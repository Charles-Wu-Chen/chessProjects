import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import functions
from flask import Flask, render_template_string
import plotly.graph_objects as go
import plotly.express as px
import chess
import pandas as pd

app = Flask(__name__)


def getAccuracyDistribution(pgnPath: str) -> dict:
    accuracies = {'acc': list(), 'rating': list()}
    with open(pgnPath, 'r') as pgn:
        while game := chess.pgn.read_game(pgn):
            if 'WhiteElo' in game.headers.keys() and 'BlackElo' in game.headers.keys():
                wRating = int(game.headers['WhiteElo'])
                bRating = int(game.headers['BlackElo'])
            else:
                continue
            ratingRange = (2700, 2900)
            node = game
            cpBefore = None
            while not node.is_end():
                node = node.variations[0]
                acc = None
                if not functions.readComment(node, True, True):
                    continue
                cpAfter = functions.readComment(node, True, True)[1]
                if not cpBefore:
                    cpBefore = cpAfter
                    continue
                if node.turn():
                    wpB = functions.winP(cpBefore * -1)
                    wpA = functions.winP(cpAfter * -1)
                    acc = min(100, functions.accuracy(wpB, wpA))
                    accuracies['acc'].append(int(acc))
                    accuracies['rating'].append(bRating)
                else:
                    wpB = functions.winP(cpBefore)
                    wpA = functions.winP(cpAfter)
                    acc = min(100, functions.accuracy(wpB, wpA))
                    accuracies['acc'].append(int(acc))
                    accuracies['rating'].append(wRating)
                cpBefore = cpAfter
    return accuracies


def plotAccuracies(accuracies: dict):
    colors = ['#689bf2', '#f8a978', '#ff87ca', '#beadfa']
    df = pd.DataFrame(accuracies)
    # fig = px.histogram(df[df['rating'].isin(range(2750, 2900))], x='acc', color='rating', log_y=True, histnorm='probability density')
    ratingBoundaries = (2400, 2750, 2900)
    x1 = list(df[df['rating'].isin(range(ratingBoundaries[1], ratingBoundaries[2]))]['acc'])
    x2 = list(df[df['rating'].isin(range(ratingBoundaries[0], ratingBoundaries[1]))]['acc'])

    candidates = getAccuracyDistribution('../out/candidates2024-WDL+CP.pgn')

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=x1, histnorm='probability density', name=f'{ratingBoundaries[1]}-{ratingBoundaries[2]}', marker_color=colors[0]))
    fig.add_trace(go.Histogram(x=x2, histnorm='probability density', name=f'{ratingBoundaries[0]}-{ratingBoundaries[1]}', marker_color=colors[1]))
    fig.add_trace(go.Histogram(x=list(candidates['acc']), histnorm='probability density', marker_color=colors[2]))
    fig.update_yaxes(type="log")
    fig.update_layout(
        height=700,
        xaxis = dict(autorange="reversed"),
        barmode='overlay',
        plot_bgcolor='#e6f7f2'
    )
    fig.update_traces(opacity=0.70)
    return fig.to_html(full_html=False)


def create_plot():
    config = {'scrollZoom': True}
    data = {
            'accuracy': [94, 99, 97, 67, 90],
            'sharpness': [0.5, 0.1, 0.6, 1.2, 0.8],
            'rating': [2700, 2666, 2803, 2770, 2750]
            }
    fig = px.scatter(data, x='accuracy', y='sharpness', size='rating', title="sample figure")
    fig.update_layout(plot_bgcolor='#e6f7f2')
    return fig.to_html(full_html=False, config=config)


@app.route('/')
def home():
    # plot_html = create_plot()
    accuracies = getAccuracyDistribution('../out/2700games2023-out.pgn')
    plot_html = plotAccuracies(accuracies)
    return render_template_string('''
        <!doctype html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
            <title>Interactive Plot</title>
        </head>
        <body>
            <div class="container">
                <h1 class="mt-5">Interactive Plotly Graph</h1>
                <div>{{ plot_html|safe }}</div>
            </div>
        </body>
        </html>
    ''', plot_html=plot_html)


if __name__ == '__main__':
    app.run(debug=True)
