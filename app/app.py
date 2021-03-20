import base64
import io

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

import numpy as np
import tensorflow as tf
from PIL import Image

from constants import CLASSES


IMAGE_WIDTH = 30
IMAGE_HEIGHT = IMAGE_WIDTH

# Load DNN model
classifier = tf.keras.models.load_model('../models/traffic_signs_2021-03-19_13-51-00.h5')

def classify_image(image, model, image_box=None):
  """Classify image by model

  Parameters
  ----------
  content: image content
  model: tf/keras classifier

  Returns
  -------
  class id returned by model classifier
  """
  images_list = []
  image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), box=image_box)
                                        # box argument clips image to (x1, y1, x2, y2)
  image = np.array(image)
  images_list.append(image)
  
  return model.predict_classes(np.array(images_list))


app = dash.Dash('Traffic Signs Recognition') #, external_stylesheets=dbc.themes.BOOTSTRAP)


pre_style = {
    'whiteSpace': 'pre-wrap',
    'wordBreak': 'break-all',
    'whiteSpace': 'normal'
}


# Define application layout
app.layout = html.Div([
    dcc.Upload(
        id='bouton-chargement',
        children=html.Div([
            'Cliquer-déposer ou ',
                    html.A('sélectionner une image')
        ]),
        style={
            'width': '50%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        }
    ),
    html.Div(id='mon-image'),
    dcc.Input(id='mon-champ-texte', value='valeur initiale', type='text'),
    html.Div(id='ma-zone-resultat')
])

@app.callback(Output('mon-image', 'children'),
              [Input('bouton-chargement', 'contents')])
def update_output(contents):
    if contents is not None:
        content_type, content_string = contents.split(',')
        if 'image' in content_type:
            image = Image.open(io.BytesIO(base64.b64decode(content_string)))
            predicted_class = classify_image(image, classifier)[0]
            return html.Div([
                html.Hr(),
                html.Img(src=contents),
                html.H3('Classe prédite : {}'.format(CLASSES[predicted_class])),
                html.Hr(),
                #html.Div('Raw Content'),
                #html.Pre(contents, style=pre_style)
            ])
        else:
            try:
                # Décodage de l'image transmise en base 64 (cas des fichiers ppm)
                # fichier base 64 --> image PIL
                image = Image.open(io.BytesIO(base64.b64decode(content_string)))
                # image PIL --> conversion PNG --> buffer mémoire 
                buffer = io.BytesIO()
                image.save(buffer, format='PNG')
                # buffer mémoire --> image base 64
                buffer.seek(0)
                img_bytes = buffer.read()
                content_string = base64.b64encode(img_bytes).decode('ascii')
                # Appel du modèle de classification
                predicted_class = classify_image(image, classifier)[0]
                # Affichage de l'image
                return html.Div([
                    html.Hr(),
                    html.Img(src='data:image/png;base64,' + content_string),
                    html.H3('Classe prédite : {}'.format(CLASSES[predicted_class])),
                    html.Hr(),
                ])
            except:
                return html.Div([
                    html.Hr(),
                    html.Div('Uniquement des images svp : {}'.format(content_type)),
                    html.Hr(),                
                    html.Div('Raw Content'),
                    html.Pre(contents, style=pre_style)
                ])
            

# Manage interactions with callbacks
@app.callback(
    Output(component_id='ma-zone-resultat', component_property='children'),
    [Input(component_id='mon-champ-texte', component_property='value')]
)
def update_output_div(input_value):
    return html.H3('Valeur saisie ici "{}"'.format(input_value))


# Start the application
if __name__ == '__main__':
    app.run_server(debug=True)
