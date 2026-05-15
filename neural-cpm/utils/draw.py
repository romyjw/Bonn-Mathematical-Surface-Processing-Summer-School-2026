# draw.py
#
# In plain words: render a 3-D point cloud coloured by a scalar field with
# Plotly. If `values` is 1-D it is a static plot; if it is (T, N) the function
# builds an animation with Play/Pause buttons and a time slider.
#
#   * If `output_file` is given the figure is also written as a self-contained
#     HTML file (handy for sharing, and for the web demo).
#   * The function ALWAYS returns the Plotly Figure so the notebook can
#     display it inline regardless of which kernel / front-end is running.

import plotly.graph_objects as go
import numpy as np

def plot_3d_point_cloud(points, values, title="3D Point Cloud", output_file=None,
                        frame_duration=200, colorscale="Magma", marker_size=2):
    """
        points: (N, 3)
        values: (N,) or (T, N) for time sequence
        output_file: optional HTML path; None means do not write to disk
        frame_duration: int, duration of each frame in milliseconds

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    values = np.array(values)
    
    # Handle single frame (N,) -> (1, N)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    
    T, N = values.shape
    
    # Create the initial trace
    trace = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=marker_size,
            color=values[0],
            colorscale=colorscale,
            opacity=0.85,
            colorbar=dict(title='u'),
            cmin=float(np.min(values)),
            cmax=float(np.max(values)),
        )
    )

    # If we only have one frame, just plot it statically
    if T == 1:
        markup_layout = dict(
            title=title,
            scene=dict(
                xaxis_title='X', 
                yaxis_title='Y', 
                zaxis_title='Z', 
                aspectmode='data'
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )
        fig = go.Figure(data=[trace], layout=markup_layout)
        if output_file:
            fig.write_html(output_file)
            print(f"Plot saved to {output_file}")
        return fig

    # If multiple frames, create animation
    frames = []
    for t in range(T):
        frames.append(go.Frame(
            data=[go.Scatter3d(
                marker=dict(color=values[t])
            )],
            name=str(t)
        ))

    # Layout with sliders and play button
    fig = go.Figure(data=[trace], frames=frames)

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X', 
            yaxis_title='Y', 
            zaxis_title='Z', 
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        updatemenus=[{
            'type': 'buttons',
            'buttons': [{
                'label': 'Play',
                'method': 'animate',
                'args': [None, {
                    'frame': {'duration': frame_duration, 'redraw': True},
                    'fromcurrent': True,
                    'transition': {'duration': 0}
                }]
            }, {
                'label': 'Pause',
                'method': 'animate',
                'args': [[None], {
                    'frame': {'duration': 0, 'redraw': False},
                    'mode': 'immediate',
                    'transition': {'duration': 0}
                }]
            }]
        }],
        sliders=[{
            'steps': [{
                'method': 'animate',
                'args': [[str(t)], {
                    'mode': 'immediate',
                    'frame': {'duration': frame_duration, 'redraw': True},
                    'transition': {'duration': 0}
                }],
                'label': str(t)
            } for t in range(T)],
            'currentvalue': {'prefix': 'Time: '}
        }]
    )

    if output_file:
        fig.write_html(output_file)
        print(f"Animation saved to {output_file}")
    return fig

