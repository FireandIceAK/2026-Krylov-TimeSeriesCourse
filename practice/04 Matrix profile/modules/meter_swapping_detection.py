import numpy as np

from plotly.subplots import make_subplots
import plotly.graph_objs as go

from modules.mp import compute_mp


def heads_tails(consumptions: dict, cutoff, house_idx: list):
    """
    Split time series into two parts: Head and Tail
    """

    heads, tails = {}, {}
    for i in house_idx:
        series = consumptions[f'House{i}']
        heads[f'H_{i}'] = series[series.index < cutoff]
        tails[f'T_{i}'] = series[series.index >= cutoff]

    return heads, tails


def _series_values(frame):
    values = frame.to_numpy(dtype=float)
    if values.ndim > 1:
        values = values[:, 0]
    return values.reshape(-1)


def meter_swapping_detection(heads: dict, tails: dict, house_idx: list, m: int) -> dict:
    """
    Find the swapped time series pair
    """

    eps = 0.001
    min_score = {
        'i': None,
        'j': None,
        'score': np.inf,
        'mp_j': None,
    }

    for i in house_idx:
        head_i = _series_values(heads[f'H_{i}'])
        tail_i = _series_values(tails[f'T_{i}'])
        self_mp = compute_mp(head_i, m, exclusion_zone=None, ts2=tail_i)
        min_self = float(np.min(self_mp['mp']))

        for j in house_idx:
            if i == j:
                continue
            tail_j = _series_values(tails[f'T_{j}'])
            cross_mp = compute_mp(head_i, m, exclusion_zone=None, ts2=tail_j)
            score = float(np.min(cross_mp['mp']) / (min_self + eps))
            if score < min_score['score']:
                min_score = {
                    'i': i,
                    'j': j,
                    'score': score,
                    'mp_j': cross_mp,
                }

    return min_score


def plot_consumptions_ts(consumptions: dict, cutoff, house_idx: list):
    """
    Plot a set of input time series and cutoff vertical line
    """

    num_ts = len(consumptions)

    fig = make_subplots(rows=num_ts, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.02)

    for i in range(num_ts):
        fig.add_trace(
            go.Scatter(
                x=list(consumptions.values())[i].index,
                y=list(consumptions.values())[i].iloc[:, 0],
                name=f"House {house_idx[i]}",
            ),
            row=i + 1,
            col=1,
        )
        fig.add_vline(x=cutoff, line_width=3, line_dash="dash", line_color="red", row=i + 1, col=1)

    fig.update_annotations(font=dict(size=22, color='black'))
    fig.update_xaxes(showgrid=False,
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=18, color='black'),
                     linewidth=2,
                     tickwidth=2)
    fig.update_yaxes(showgrid=False,
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=18), color='black',
                     zeroline=False,
                     linewidth=2,
                     tickwidth=2)

    fig.update_layout(title='Houses Consumptions',
                      title_x=0.5,
                      title_font=dict(size=26, color='black'),
                      plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor='rgba(0,0,0,0)',
                      height=800,
                      legend=dict(font=dict(size=20, color='black')))

    fig.show()
