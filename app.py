import base64
import io
import requests
from PIL import Image, ImageDraw, ImageFont

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the improved layout
app.layout = dbc.Container([
    # Store component to hold the API key (loaded from a file)
    dcc.Store(id='api_key', storage_type='session'),
    # Hidden div to trigger saving the API key
    html.Div("", id="save-api-key-div", style={"display": "none"}),

    # App Header
    dbc.Row(
        dbc.Col(
            html.H2("Object Detection with Vision-Agent", className="text-center my-4"),
            width=12
        )
    ),

    # Upload and Options Section
    dbc.Row([
        # Left Column: Image Upload
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Upload Image"),
                dbc.CardBody(
                    dcc.Upload(
                        id='upload-image',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select an Image', style={'color': 'blue', 'textDecoration': 'underline'})
                        ]),
                        style={
                            'width': '100%',
                            'height': '100px',
                            'lineHeight': '100px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px 0'
                        },
                        multiple=False
                    )
                )
            ]),
            md=6
        ),
        # Right Column: Detection Options
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Detection Options"),
                dbc.CardBody([

                    html.Div([
                        dbc.Label("Prompts (comma-separated)"),
                        dcc.Input(
                            id='input-prompts',
                            type='text',
                            placeholder='E.g. "green shoe, gaming chair"',
                            style={'width': '100%'},
                        )
                    ], className="mb-3"),

                    dbc.Button(
                        "Detect Objects",
                        id='detect-button',
                        color='primary',
                        n_clicks=0,
                        style={'width': '100%'}
                    )
                ])
            ]),
            md=6
        )
    ], className="mb-4"),

    # Display Section: Uploaded Image and Detection Results
    dbc.Row([
        # Uploaded Image Card
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Uploaded Image"),
                dbc.CardBody(
                    html.Div(id='uploaded-image-container', style={'textAlign': 'center'})
                )
            ]),
            md=6
        ),
        # Detection Results Card (wrapped in dcc.Loading)
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Detection Results"),
                dbc.CardBody(
                        dcc.Loading(
                            id="loading-results",
                            type="default",
                            children=[
                                html.Div(id='annotated-image-container', style={'textAlign': 'center', 'marginBottom': '20px'}),
                                html.Div(id='detection-results', style={'whiteSpace': 'pre-wrap'}),
                            ]
                        )
                    )
            ]),
            md=6
        )
    ])
], fluid=True)


def call_agentic_object_detection_api(image_bytes, prompts, api_key):
    """
    Calls the VisionAgent API with the given image bytes and prompts.
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
    return response.json()


# def draw_bounding_boxes(decoded_image, detections):
#     """
#     Draws bounding boxes and labels on the image using PIL.
#     """
#     image = Image.open(io.BytesIO(decoded_image))
#     draw = ImageDraw.Draw(image)
#     width, height = image.size
#     font = None  # You can load a custom font if available

#     for det in detections:
#         label = det.get("label", "N/A")
#         box = det.get("bounding_box", {})  # Expecting [xmin, ymin, xmax, ymax] in pixel values

#         # Unpack box coordinates (assumed to be already in pixel values)
#         xmin, ymin, xmax, ymax = box

#         # Draw bounding box
#         draw.rectangle([xmin, ymin, xmax, ymax], outline='red', width=3)
        
#         # Draw label background rectangle for readability
#         text = label
#         text_bbox = draw.textbbox((0, 0), text, font=font)
#         text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
#         draw.rectangle([xmin, ymin - text_height, xmin + text_width + 4, ymin], fill='red')
#         draw.text((xmin + 2, ymin - text_height), text, fill='white', font=font)

#     # Convert annotated image back to bytes
#     output_buffer = io.BytesIO()
#     image.save(output_buffer, format='JPEG')
#     return output_buffer.getvalue()

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
    Input('save-api-key-div', 'children'),
)
def store_api_key(_):
    """
    Reads the API key from file "visionagent_api_key.txt" and stores it in the dcc.Store component.
    """
    with open("visionagent_api_key.txt", 'r') as f:
        api_key = f.read().strip()
    return api_key


@app.callback(
    Output('detection-results', 'children'),
    Output('annotated-image-container', 'children'),
    Input('detect-button', 'n_clicks'),
    State('upload-image', 'contents'),
    State('input-prompts', 'value'),
    State('api_key', 'data'),
    prevent_initial_call=True
)
def detect_objects(n_clicks, uploaded_image_contents, prompt_text, api_key):
    # Clear previous results: (they’ll be overwritten when the callback returns)
    if not uploaded_image_contents:
        return "No image uploaded yet.", None
    if not prompt_text:
        return "Please enter at least one prompt.", None
    if not api_key:
        return "API key missing. Please ensure you have saved your API key.", None

    # Decode the uploaded image
    content_type, content_string = uploaded_image_contents.split(',')
    decoded_image = base64.b64decode(content_string)

    # Call the API
    try:
        result = call_agentic_object_detection_api(decoded_image, prompt_text, api_key)
    except Exception as e:
        return f"API call failed: {str(e)}", None

    if "data" not in result:
        return f"Unexpected response: {result}", None

    detections = result["data"][0]
    if not detections:
        return "No objects detected for the given prompts.", None

   # Order output by score
    detections = sorted(detections, key=lambda x: x.get("score", 0), reverse=True)

    # Build detection result text
    output_str = "Objects Detected:\n"
    for det in detections:
        label = det.get("label", "N/A")
        score = det.get("score", "N/A")
        box = det.get("bounding_box", {})
        output_str += f"Label: {label}, Score: {score}, Box: {box}\n"

 
    # Draw bounding boxes on the image
    annotated_bytes = draw_bounding_boxes(decoded_image, detections)
    annotated_base64 = base64.b64encode(annotated_bytes).decode('utf-8')
    annotated_image_data = f"data:image/jpeg;base64,{annotated_base64}"
    annotated_image_element = html.Img(
        src=annotated_image_data,
        style={'maxWidth': '100%', 'height': 'auto', 'border': '2px solid #ccc'}
    )

    return output_str, annotated_image_element


@app.callback(
    Output('uploaded-image-container', 'children'),
    Input('upload-image', 'contents')
)
def display_uploaded_image(contents):
    if contents is not None:
        return html.Img(src=contents, style={'maxWidth': '100%', 'height': 'auto'})
    return html.Div("No image uploaded yet.")


if __name__ == '__main__':
    app.run_server(debug=True)