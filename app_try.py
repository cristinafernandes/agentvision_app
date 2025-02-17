import base64
import io
import requests
import zipfile
from PIL import Image, ImageDraw, ImageFont

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State

# Initialize Dash with a Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout
app.layout = dbc.Container([
    # Hidden store for API key
    dcc.Store(id='api_key', storage_type='session'),

    # Hidden store for detection results
    dcc.Store(id='detection_store', storage_type='memory'),

    # A hidden element to trigger loading the key on startup
    html.Div("", id="load-api-key-div", style={"display": "none"}),

    #Title
    dbc.Row(
        dbc.Col(
            html.H2("Object Detection with VisionAgent", className="text-center my-4"),
            width=12
        )
    ),

    # Upload and Options Section
    dbc.Row([
        # Column: Image Upload
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Upload an Image"),
                dbc.CardBody(
                    [
                        dcc.Upload(
                            id='upload-image',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select an Image', style={'color': 'blue', 'textDecoration': 'underline'})
                            ]),
                            style={
                                'width': '100%',
                                'height': '125px',
                                'lineHeight': '100px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                            },
                            multiple=False
                        ),
                    ]
                )
            ], className="mb-4 shadow-sm"),
            md=6 # takes up half the width (6/12)
        ),

        # Column: Detection Options
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Detection Options"),
                dbc.CardBody([
                    dbc.Label("Prompts (comma-separated)"),
                    dcc.Input(
                        id='input-prompts',
                        type='text',
                        placeholder='E.g. "green pepper, round table"',
                        style={'width': '100%'},
                    ),
                    html.Br(),
                    html.Br(),
                    dbc.Button(
                        "Detect Objects",
                        id='detect-button',
                        color='primary',
                        n_clicks=0,
                        style={'width': '100%'}
                    )
                ])
            ], className="mb-4 shadow-sm"),
            md=6 # takes up half the width (6/12)
        )
    ]),

    # Display Section: Uploaded Image & Detection Results
    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Uploaded Image"),
                dbc.CardBody(
                    html.Div(id='uploaded-image-container', style={'textAlign': 'center'})
                )
            ],
            className="mb-4 shadow-sm"),
            md=6 # takes up half the width (6/12)
        ),
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Detection Results"),
                dbc.CardBody([
                    dcc.Loading(
                        id="loading-results",
                        type="default",
                        children=[
                            html.Div(id='annotated-image-container', style={'textAlign': 'center'}),
                            html.Div(id='detection-results', style={'whiteSpace': 'pre-wrap'}),
                        ],
                    ),
                    # Download button for exporting results
                    dbc.Button(
                        "Download Results",
                        id="download-button",
                        color="primary",
                        n_clicks=0,
                        style={'display':'none'}, #{'width': '100%', 'marginTop': '1rem'}
                    ),
                    dcc.Download(id="download-results") # For downloading the results
                ])
            ],
            className="mb-4 shadow-sm"),
            md=6 # takes up half the width (6/12)
        )
    ])
], fluid=True)

def call_agentic_object_detection_api(image_bytes, prompts, api_key):
    """
    Calls the VisionAgent / Agentic Object Detection API.
    """
    url = "https://api.landing.ai/v1/tools/agentic-object-detection"
    prompts_list = [p.strip() for p in prompts.split(",") if p.strip()]
    files = {
        'image': ('uploaded_image.jpg', image_bytes, 'image/jpeg')
    }
    data = {
        'prompts': prompts_list,
        'model': 'agentic'
    }
    headers = {
        'Authorization': f'Basic {api_key}'
    }
    response = requests.post(url, files=files, data=data, headers=headers)
    response.raise_for_status()
    return response.json()


def draw_bounding_boxes(decoded_image, detections):
    image = Image.open(io.BytesIO(decoded_image))
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    scores = [det.get("score", 0) for det in detections]
    min_score, max_score = min(scores), max(scores)
    
    for det in sorted(detections, key=lambda x: x.get("score", 0), reverse=True):
        label, score = det.get("label", "N/A"), det.get("score", 0)
        box = det.get("bounding_box", [])
        xmin, ymin, xmax, ymax = box
        
        norm_score = (score - min_score) / (max_score - min_score) if max_score > min_score else 1
        r, g, b = int(255 * (1 - norm_score)), int(255 * norm_score), 0  # Green to Red gradient
        color = (r, g, b)
        
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
        draw.text((xmin + 2, ymin - 15), f"{label} ({score:.2f})", fill=color)
    
    output_buffer = io.BytesIO()
    image.save(output_buffer, format='JPEG')
    return output_buffer.getvalue()


@app.callback(
    Output('api_key', 'data'),
    Input('load-api-key-div', 'children'),
)
def store_api_key(_):
    """
    Reads the API key from file "visionagent_api_key.txt" and
    stores it in the dcc.Store component when the app starts.
    """
    with open("visionagent_api_key.txt", 'r') as f:
        api_key = f.read().strip()
    return api_key

@app.callback(
    Output('uploaded-image-container', 'children'),
    Output('detection-results', 'children'),
    Output('annotated-image-container', 'children'),
    Output('detection_store', 'data'),
    Input('upload-image', 'contents')
)
def display_uploaded_image(contents):
    """
    Show the uploaded image and clear the previous detection results.
    """
    if contents is not None:
        # Display the newly uploaded image
        return (
            html.Img(src=contents, style={'maxWidth': '100%', 'height': 'auto'}),
            "",  # clear detection results
            None, # clear annotated image
            None, # clear detection store
        )
    # If no image uploaded yet
    return (html.Div("No image uploaded yet."), "", html.Div("No image uploaded yet."), None)

@app.callback(
    Output('detection-results', 'children', allow_duplicate=True),
    Output('annotated-image-container', 'children', allow_duplicate=True),
    Output('detection_store', 'data', allow_duplicate=True),
    Input('detect-button', 'n_clicks'),
    State('upload-image', 'contents'),
    State('input-prompts', 'value'),
    State('api_key', 'data'),
    prevent_initial_call=True
)
def detect_objects(n_clicks, uploaded_image_contents, prompt_text, api_key):
    """
    When 'Detect Objects' is clicked, call the VisionAgent detection API,
    process results, and store detection outputs for later download.
    """
    if not uploaded_image_contents:
        return ("No image uploaded yet.", None, None)

    if not prompt_text:
        return ("Please enter at least one prompt.", None, None)

    if not api_key:
        return ("API key missing. Please ensure you have saved your API key.", None, None)

    # Decode the uploaded image
    content_type, content_string = uploaded_image_contents.split(',')
    decoded_image = base64.b64decode(content_string)

    # Call the VisionAgent detection API
    try:
        result = call_agentic_object_detection_api(decoded_image, prompt_text, api_key)
    except Exception as e:
        return (f"API call failed: {str(e)}", None, None)

    if "data" not in result or len(result["data"]) == 0:
        return (f"Unexpected response: {result}", None, None)

    detections = result["data"][0]
    if not detections:
        return ("No objects detected for the given prompts.", None, None)

    # Sort detections by score (descending) and build the detection text
    detections = sorted(detections, key=lambda x: x.get("score", 0), reverse=True)
    output_str = "Objects Detected:\n"
    for det in detections:
        label = det.get("label", "N/A")
        score = det.get("score", 0)
        box = det.get("bounding_box", {})
        output_str += f"- Label: {label}, Score: {score:.2f}, Box: {box}\n"

    # Draw bounding boxes on the image
    annotated_bytes = draw_bounding_boxes(decoded_image, detections)
    annotated_base64 = base64.b64encode(annotated_bytes).decode('utf-8')
    annotated_image_data = f"data:image/jpeg;base64,{annotated_base64}"

    annotated_image_element = html.Img(
        src=annotated_image_data,
        style={'maxWidth': '100%', 'height': 'auto', 'border': '2px solid #ccc'}
    )

    # Store the detection results for exporting
    store_data = {
        "annotated_image": annotated_image_data,
        "detection_text": output_str
    }

    return (output_str, annotated_image_element, store_data)

# Callback to toggle the download button's visibility.
# It listens to both the detection_store and the detect-button clicks.
@app.callback(
    Output("download-button", "style"),
    [Input("detection_store", "data"),
     Input("detect-button", "n_clicks")]
)
def toggle_download_button(detection_data, detect_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return {'display': 'none'}
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    # If the detect-button was just clicked, hide the download button immediately.
    if triggered_id == 'detect-button':
        return {'display': 'none'}
    # Otherwise, if detection data exists, show the button.
    if detection_data:
        return {'width': '100%', 'marginTop': '1rem'}
    return {'display': 'none'}

@app.callback(
    Output("download-results", "data"),
    Input("download-button", "n_clicks"),
    State("detection_store", "data"),
    prevent_initial_call=True
)
def download_results(n_clicks, detection_data):
    if not detection_data:
        return dash.no_update

    # Extract annotated image data and detection text
    annotated_image = detection_data.get("annotated_image", None)
    detection_text = detection_data.get("detection_text", "")

    # Decode the annotated image from base64
    if annotated_image and annotated_image.startswith("data:image"):
        annotated_bytes = base64.b64decode(annotated_image.split(",")[1])
    else:
        annotated_bytes = b""

    # Create a ZIP file in memory containing both the annotated image and text summary
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("annotated_image.jpg", annotated_bytes)
        zf.writestr("detection_results.txt", detection_text)
    buffer.seek(0)
    return dcc.send_bytes(buffer.getvalue(), "results.zip")


if __name__ == '__main__':
    app.run_server(debug=True)