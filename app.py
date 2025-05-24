import base64
import io
import requests
import zipfile
import csv
import os
from dotenv import load_dotenv

from PIL import Image, ImageDraw

import dash
import dash_bootstrap_components as dbc
from dash import dash_table
from dash import dcc, html, Input, Output, State


# Initialize Dash with a Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout
app.layout = dbc.Container(
    [
        # Hidden store for API key
        dcc.Store(id="api_key", storage_type="session"),
        # Hidden store for detection results
        dcc.Store(id="detection_store", storage_type="memory"),
        # A hidden element to trigger loading the key on startup
        html.Div("", id="load-api-key-div", style={"display": "none"}),
        # Title
        dbc.Row(
            dbc.Col(
                html.H2(
                    "Object Detection with VisionAgent", className="text-center my-4"
                ),
                width=12,
            )
        ),
        # Upload and Options Section
        dbc.Row(
            [
                # Column: Image Upload
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Upload an Image"),
                            dbc.CardBody(
                                [
                                    dcc.Upload(
                                        id="upload-image",
                                        children=html.Div(
                                            [
                                                "Drag and Drop or ",
                                                html.A(
                                                    "Select an Image",
                                                    style={
                                                        "color": "blue",
                                                        "textDecoration": "underline",
                                                    },
                                                ),
                                            ]
                                        ),
                                        style={
                                            "width": "100%",
                                            "height": "125px",
                                            "lineHeight": "100px",
                                            "borderWidth": "1px",
                                            "borderStyle": "dashed",
                                            "borderRadius": "5px",
                                            "textAlign": "center",
                                        },
                                        multiple=False,
                                    ),
                                ]
                            ),
                        ],
                        className="mb-4 shadow-sm",
                    ),
                    md=6,
                ),
                # Column: Detection Options
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Detection Options"),
                            dbc.CardBody(
                                [
                                    dbc.Label("Prompts (comma-separated, up to 3)"),
                                    dcc.Input(
                                        id="input-prompts",
                                        type="text",
                                        placeholder='E.g. "car, bike, person"',
                                        style={"width": "100%"},
                                    ),
                                    html.Br(),
                                    html.Br(),
                                    dbc.Button(
                                        "Detect Objects",
                                        id="detect-button",
                                        color="primary",
                                        n_clicks=0,
                                        style={"width": "100%"},
                                    ),
                                ]
                            ),
                        ],
                        className="mb-4 shadow-sm",
                    ),
                    md=6,
                ),
            ]
        ),
        # Display Section: Uploaded Image & Detection Results
        dbc.Row(
            [
                # Left Card: Uploaded Image
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Uploaded Image"),
                            dbc.CardBody(
                                html.Div(
                                    id="uploaded-image-container",
                                    style={"textAlign": "center"},
                                )
                            ),
                        ],
                        # Make card stretch to match height of the Decetion Results card
                        className="flex-fill shadow-sm", 
                        style={"display": "flex", "flexDirection": "column"},
                    ),
                    md=6,
                    className="d-flex mb-4",  # Ensures column acts as flex container
                ),

                # Right Card: Detection Results
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Detection Results"),
                            dbc.CardBody(
                                [
                                    dcc.Loading(
                                        id="loading-results",
                                        type="default",
                                        children=[
                                            html.Div(
                                                id="annotated-image-container",
                                                style={"textAlign": "center"},
                                            ),
                                            html.Div(
                                                id="detection-results",
                                                style={
                                                    "whiteSpace": "pre-wrap",
                                                    "marginTop": "20px",
                                                },
                                            ),
                                        ],
                                    ),
                                    dbc.Button(
                                        "Download Results",
                                        id="download-button",
                                        color="primary",
                                        n_clicks=0,
                                        style={"display": "none"},
                                    ),
                                    dcc.Download(id="download-results"),
                                ],
                                # Make card body fill remaining vertical space
                                style={"flex": "1", "display": "flex", "flexDirection": "column"}
                            ),
                        ],
                        # Match height with Uploaded Image card
                        className="flex-fill shadow-sm",
                        style={"display": "flex", "flexDirection": "column"},
                    ),
                    md=6,
                    className="d-flex mb-4", # Ensures column acts as flex container
                ),
            ]
        ),
    ],
    fluid=True,
)


def call_agentic_object_detection_api(image_bytes, prompt, api_key):
    """
    Calls the VisionAgent / Agentic Object Detection API for a single object (prompt).
    VisionAgent documentation say: "Only one object type can be detected at a time."
    """
    url = "https://api.va.landing.ai/v1/tools/agentic-object-detection"
    files = {"image": ("uploaded_image.jpg", image_bytes, "image/jpeg")}
    data = {"prompts": prompt, "model": "agentic"}
    headers = {"Authorization": f"Basic {api_key}"}
    response = requests.post(url, files=files, data=data, headers=headers)
    response.raise_for_status()
    return response.json()


def draw_bounding_boxes(decoded_image, detections, prompt_colors):
    """
    Draws bounding boxes on the image with colors based on the prompt and score.
    Score=1 → base prompt color; score=0 → black; gradient in between.
    """
    # Convert to RGB to avoid RGBA→JPEG errors
    image = Image.open(io.BytesIO(decoded_image)).convert("RGB")
    draw = ImageDraw.Draw(image)

    for det in detections:
        label = det.get("label", "N/A")
        score = det.get("score", 0.0)
        box = det.get("bounding_box", [])
        if len(box) != 4:
            continue

        xmin, ymin, xmax, ymax = box
        base_color = prompt_colors.get(label, (0, 0, 0))
        # interpolate toward black
        r = int(base_color[0] * score)
        g = int(base_color[1] * score)
        b = int(base_color[2] * score)
        color = (r, g, b)

        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
        draw.text((xmin + 2, ymin - 15), f"{label} ({score:.2f})", fill=color)

    output_buffer = io.BytesIO()
    image.save(output_buffer, format="JPEG")
    return output_buffer.getvalue()


@app.callback(
    Output("api_key", "data"),
    Input("load-api-key-div", "children"),
)
def store_api_key(_):
    load_dotenv()
    api_key = os.getenv("VISIONAGENT_API_KEY")
    return api_key


@app.callback(
    Output("uploaded-image-container", "children"),
    Output("detection-results", "children"),
    Output("annotated-image-container", "children"),
    Output("detection_store", "data"),
    Input("upload-image", "contents"),
)
def display_uploaded_image(contents):
    if contents is not None:
        return (
            html.Img(src=contents, style={"maxWidth": "100%", "height": "auto"}),
            "",  # clear detection results
            None,  # clear annotated image
            None,  # clear detection store
        )
    return (
        html.Div("No image uploaded yet."),
        "",
        html.Div("No image uploaded yet."),
        None,
    )


@app.callback(
    Output("detection-results", "children", allow_duplicate=True),
    Output("annotated-image-container", "children", allow_duplicate=True),
    Output("detection_store", "data", allow_duplicate=True),
    Input("detect-button", "n_clicks"),
    State("upload-image", "contents"),
    State("input-prompts", "value"),
    State("api_key", "data"),
    prevent_initial_call=True,
)
def detect_objects(n_clicks, uploaded_image_contents, prompt_text, api_key):
    if not uploaded_image_contents:
        return ("No image uploaded yet.", None, None)
    if not prompt_text:
        return ("Please enter at least one prompt.", None, None)
    if not api_key:
        return (
            "API key missing. Please ensure you have saved your API key.",
            None,
            None,
        )

    raw_prompts = [p.strip() for p in prompt_text.split(",") if p.strip()]
    prompts_list = raw_prompts[:3]
    if not prompts_list:
        return ("No valid prompts found.", None, None)

    # assign base colors: green, blue, red
    base_palette = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]
    prompt_colors = {prompts_list[i]: base_palette[i] for i in range(len(prompts_list))}

    _, content_string = uploaded_image_contents.split(",")
    decoded_image = base64.b64decode(content_string)

    combined_detections = []
    status = [
        f"{len(prompts_list)} object type(s) prompted: {', '.join(prompts_list)}.\n"
    ]

    for prompt in prompts_list:
        try:
            res = call_agentic_object_detection_api(decoded_image, prompt, api_key)
        except Exception as e:
            status.append(f"Failed detecting '{prompt}': {e}\n")
            continue
        if "data" in res and res["data"]:
            for grp in res["data"]:
                detections = grp if isinstance(grp, list) else [grp]
                status.append(f"Found {len(detections)} detection(s) for '{prompt}'\n")
                combined_detections.extend(detections)
        else:
            status.append(f"No detections returned for '{prompt}'\n")

    if not combined_detections:
        return ("".join(status) + "\nNo objects detected.", None, None)

    combined_detections.sort(key=lambda x: x.get("score", 0), reverse=True)

    table_data = []
    style_cond = []
    for idx, det in enumerate(combined_detections):
        label = det.get("label", "N/A")
        score = det.get("score", 0.0)
        bbox = det.get("bounding_box", {})
        table_data.append(
            {
                "Label": label,
                "Score": f"{score:.2f}",
                "Bounding Box": str(bbox),
            }
        )
        base = prompt_colors.get(label, (0, 0, 0))
        r, g, b = int(base[0] * score), int(base[1] * score), int(base[2] * score)
        style_cond.append(
            {
                "if": {"row_index": idx, "column_id": "Label"},
                "color": f"rgb({r},{g},{b})",
            }
        )

    detection_table = dash_table.DataTable(
        columns=[
            {"name": "Label", "id": "Label"},
            {"name": "Score", "id": "Score"},
            {"name": "Bounding Box", "id": "Bounding Box"},
        ],
        data=table_data,
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "5px"},
        style_data_conditional=style_cond,
    )

    annotated_bytes = draw_bounding_boxes(
        decoded_image, combined_detections, prompt_colors
    )
    annotated_b64 = base64.b64encode(annotated_bytes).decode("utf-8")
    annotated_img = html.Img(
        src=f"data:image/jpeg;base64,{annotated_b64}",
        style={"maxWidth": "100%", "height": "auto", "border": "2px solid #ccc"},
    )

    results_div = html.Div(
        [
            html.Pre("".join(status)),
            html.Div(detection_table, style={"marginTop": "20px"}),
        ]
    )

    store = {
        "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
        "detection_table": table_data,
    }

    return (results_div, annotated_img, store)


@app.callback(
    Output("download-button", "style"),
    [Input("detection_store", "data"), Input("detect-button", "n_clicks")],
)
def toggle_download(detect_data, clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return {"display": "none"}
    tid = ctx.triggered[0]["prop_id"].split(".")[0]
    if tid == "detect-button":
        return {"display": "none"}
    if detect_data:
        return {"width": "100%", "marginTop": "1rem"}
    return {"display": "none"}


@app.callback(
    Output("download-results", "data"),
    Input("download-button", "n_clicks"),
    State("detection_store", "data"),
    prevent_initial_call=True,
)
def download_results(n, data):
    if not data:
        return dash.no_update

    ann = data.get("annotated_image", "")
    tbl = data.get("detection_table", [])

    # CSV
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["Label", "Score", "Bounding Box"])
    w.writeheader()
    w.writerows(tbl)
    csv_bytes = buf.getvalue().encode("utf-8")

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        if ann.startswith("data:image"):
            img_b = base64.b64decode(ann.split(",")[1])
        else:
            img_b = b""
        zf.writestr("annotated_image.jpg", img_b)
        zf.writestr("results.csv", csv_bytes)
    zip_buf.seek(0)
    return dcc.send_bytes(zip_buf.getvalue(), "results.zip")


if __name__ == "__main__":
    app.run_server(debug=True)
