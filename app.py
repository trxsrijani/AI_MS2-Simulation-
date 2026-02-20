from flask import Flask, render_template, Response, jsonify, request
import cv2
import json
from ultralytics import YOLO

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
model = YOLO(r"/home/srijani/AI SMARTSHIP/AI_MS2_SIMULATION/best5.pt")

app = Flask(__name__)

# -------------------------------------------------
# LOAD SENSOR JSON
# -------------------------------------------------
with open("/home/srijani/AI SMARTSHIP/AI_MS2_SIMULATION/sensor_data.json") as f:
    fusion_data = json.load(f)

targets = fusion_data["targets"]

# -------------------------------------------------
# GLOBALS
# -------------------------------------------------
current_objects = []
selected_object_id = None
current_display_data = {}

# Track management
track_metadata = {}      # track_id -> {last_seen, sim_id}
frame_counter = 0
STALE_THRESHOLD = 30     # frames (~1 sec if 30fps)

# -------------------------------------------------
# VIDEO + YOLO TRACKING
# -------------------------------------------------
def generate_frames():

    global current_objects, selected_object_id
    global current_display_data, track_metadata
    global frame_counter

    cap = cv2.VideoCapture("/home/srijani/AI SMARTSHIP/AI_MS2_SIMULATION/naval_dock.mp4")

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_counter += 1
        frame_objects = []
        active_track_ids = set()

        results = model.track(frame, persist=True, conf=0.5)

        for r in results:
            for box in r.boxes:

                if box.id is None:
                    continue

                track_id = int(box.id[0])
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                active_track_ids.add(track_id)

                # -------------------------------------------------
                # NEW TRACK → ASSIGN SIM TARGET
                # -------------------------------------------------
                if track_id not in track_metadata:

                    sim_target_id = None

                    for t in targets:
                        if t["class"].lower() == class_name.lower():
                            if t["object_id"] not in [v["sim_id"] for v in track_metadata.values()]:
                                sim_target_id = t["object_id"]
                                break

                    track_metadata[track_id] = {
                        "last_seen": frame_counter,
                        "sim_id": sim_target_id
                    }

                else:
                    # Update last seen
                    track_metadata[track_id]["last_seen"] = frame_counter

                frame_objects.append({
                    "id": track_id,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                })

                # Draw box
                color = (0, 255, 255)
                if track_id == selected_object_id:
                    color = (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame,
                            f"ID {track_id} - {class_name}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2)

        current_objects = frame_objects

        # -------------------------------------------------
        # CLEANUP STALE TRACKS
        # -------------------------------------------------
        stale_ids = []

        for track_id, data in track_metadata.items():
            if frame_counter - data["last_seen"] > STALE_THRESHOLD:
                stale_ids.append(track_id)

        for sid in stale_ids:
            del track_metadata[sid]

            if sid == selected_object_id:
                selected_object_id = None
                current_display_data = {}

        # -------------------------------------------------
        # SENSOR DATA FETCH
        # -------------------------------------------------
        if selected_object_id is not None:

            if selected_object_id in track_metadata:

                sim_id = track_metadata[selected_object_id]["sim_id"]

                target = next((t for t in targets if t["object_id"] == sim_id), None)

                if target:
                    current_display_data = {
                        "id": selected_object_id,
                        "class": target["class"],
                        "speed": target["speed_kts"],
                        "course": target["course_deg_T"],
                        "range": target["range_m"],
                        "cpa": target["cpa_m"],
                        "threat": target["threat_level"],
                        "collision": target["collision_status"]
                    }

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


# -------------------------------------------------
# ROUTES
# -------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/objects')
def objects():
    return jsonify(current_objects)

@app.route('/sensor_data')
def sensor_data():
    return jsonify(current_display_data)

@app.route('/select_object', methods=['POST'])
def select_object():
    global selected_object_id
    selected_object_id = int(request.json["id"])
    return jsonify({"status": "ok"})

# -------------------------------------------------
# RUN
# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, threaded=True)