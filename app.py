import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import base64
import numpy as np
import tensorflow as tf
from flask import Flask

# Load the MNIST model
model = tf.keras.models.load_model('./mnist_model.h5')

# Create the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Define the layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("MNIST Digit Recognition"), className="text-center mb-4")
    ]),
    dbc.Row([
        dbc.Col(
            dcc.Upload(
                id='upload-image',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Image', className="font-weight-bold text-primary")
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                multiple=False
            ),
            width={'size': 6, 'offset': 3}
        )
    ]),
    dbc.Row([
        dbc.Col(
            html.Button("Predict", id='predict-button', className="btn-primary", style={'width': '100%'}),
            width={'size': 6, 'offset': 3}
        )
    ]),
    dbc.Row([
        dbc.Col(
            html.Div(id='output-container', style={'textAlign': 'center', 'marginTop': '20px'}),
            width={'size': 6, 'offset': 3}
        )
    ]),
    dbc.Row([
        dbc.Col(
            html.Div(id='output-image-container', style={'textAlign': 'center', 'marginTop': '20px'}),
            width={'size': 6, 'offset': 3}
        )
    ])
], className="mt-5")

# Define the callback function for image upload and prediction
@app.callback(Output('output-container', 'children'),
              Output('output-image-container', 'children'),
              [Input('predict-button', 'n_clicks')],
              [State('upload-image', 'contents')])
def predict_image(n_clicks, contents):
    if n_clicks and contents:
        # Convert the uploaded image to base64 string
        _, content_string = contents.split(',')
        encoded_string = content_string.encode("utf-8")
        decoded_image = base64.b64decode(encoded_string)

        # Resize and normalize the image
        image = tf.image.decode_image(decoded_image, channels=1)
        image = tf.image.resize(image, [28, 28])
        image = np.array(image) / 255.0
        image = image.reshape(1, 28, 28, 1)

        # Make predictions
        prediction = np.argmax(model.predict(image))

        # Display the image
        img_src = f'data:image/png;base64,{encoded_string.decode()}'
        image_element = html.Img(src=img_src, style={'width': '200px', 'height': '200px'})

        return dbc.Row([
            dbc.Col(html.H3(f"Predicted digit: {prediction}", className="text-center"))
        ]), dbc.Row([
            dbc.Col(image_element)
        ])

if __name__ == '__main__':
    server.run(host='0.0.0.0')
