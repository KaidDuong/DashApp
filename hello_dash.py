import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import datetime
from flask_caching import Cache
import os
import pandas as pd
import time
import uuid
import numpy as np
import plotly.graph_objects as go
def test():
    external_stylesheets = [
        # Dash CSS
        'https://codepen.io/chriddyp/pen/bWLwgP.css']

    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    df = pd.read_csv("/home/quangdq/Kaid/NSFW/nsfw-api/app/report.csv")
    columns = ['drawings', 'hentai', 'neutral', 'porn', 'sexy', 'model_id']
    dff = df[columns]
    df2 = pd.DataFrame(np.sqrt(np.sum(
        np.square(dff[dff.model_id == 1].iloc[:, :-1].values[:-1] - dff[dff.model_id == 2].iloc[:, :-1].values[:-1]),
        axis=1)), columns=["l2_distance"])
    df3 = pd.DataFrame(np.sqrt(np.sum(
        np.square(dff[dff.model_id == 1].iloc[:, :-1].values[:-1] - dff[dff.model_id == 3].iloc[:, :-1].values[:-1]),
        axis=1)), columns=["l2_distance"])
    df4 = pd.DataFrame(np.sqrt(np.sum(
        np.square(dff[dff.model_id == 2].iloc[:, :-1].values[:-1] - dff[dff.model_id == 3].iloc[:, :-1].values[:-1]),
        axis=1)), columns=["l2_distance"])

    # fig2 = px.scatter(x=df2.index.values, y=df2.l2_distance.values)

    fig3 = go.Figure()
    # Add traces
    fig3.add_trace(go.Scatter(x=df2.index.values, y=df2.l2_distance.values,
                              name='Tflite vs H5', mode="markers"))
    fig3.add_trace(go.Scatter(x=df3.index.values, y=df3.l2_distance.values,
                              name='Onnx vs H5', mode="markers"))
    fig3.add_trace(go.Scatter(x=df4.index.values, y=df3.l2_distance.values,
                              name='Onnx vs Tflite', mode="markers"))
    fig3.update_layout(
        title={
            'text': "The difference between the other Models ",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title="Frame Idxes",
        yaxis_title="Euclidean distance",
        legend_title="Models",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"
        )
    )
    fig3.update_yaxes(type="log", range=[-10, 0])

    app.layout = html.Div(children=[
        dcc.Graph(
            id='graph_distance',
            figure=fig3
        ),
        dcc.Graph(id='graph-with-slider'),
        dcc.Slider(
            id='input-slider',
            min=df['frame_idxs'].min(),
            max=df['frame_idxs'].max(),
            # value=df['frame_idxs'].min(),
            # marks={str(idx): str(idx) for idx in df['frame_idxs'].unique()[:100]},
            step=1
        ),
        html.Div(id='slider-drag-output', style={'margin-top': 20, 'margin-left': 100})
    ])

    @app.callback(Output('slider-drag-output', 'children'),
                  Input('input-slider', 'value'))
    def display_value(value):
        return 'Frame: {}'.format(value)

    @app.callback(
        Output('graph-with-slider', 'figure'),
        Input('input-slider', 'value'))
    def update_figure(selected_frame):
        filtered_df = df[df.frame_idxs == selected_frame]
        columns = ['drawings', 'hentai', 'neutral', 'porn', 'sexy', 'model_id']
        labels = {k: v for k, v in zip(columns, ['Drawings', 'Hentai', 'Neutral', 'Porn', 'Sexy', 'Model IDs'])}
        fig = px.parallel_coordinates(filtered_df, color='model_id',
                                      dimensions=columns,
                                      labels=labels,
                                      color_continuous_scale=px.colors.diverging.Armyrose,
                                      color_continuous_midpoint=2)
        fig.update_layout(
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="RebeccaPurple"
            )
        )
        fig.update_layout(transition_duration=500)

        return fig

    @app.callback(
        Output('input-slider', 'value'),
        Input('graph_distance', 'hoverData'))
    def update_slider(hoverData):
        if hoverData is not None:
            return hoverData["points"][0]["x"]
        else:
            return df['frame_idxs'].min()

    return app
if __name__ == '__main__':
    app = test()
    app.run_server(debug=True)