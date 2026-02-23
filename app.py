from flask import Flask, render_template, Response, jsonify, request
import cv2
import json
from ultralytics import YOLO

# -------------------------------------------------
# LOAD YOLO MODEL
# -------------------------------------------------
model = YOLO(r"/home/tractrix/Desktop/AI_SmartShip/AI_MS2-Simulation-/best5.pt")

app = Flask(__name__)

# -------------------------------------------------
# LOAD SENSOR JSON
# -------------------------------------------------
with open("/home/tractrix/Desktop/AI_SmartShip/AI_MS2-Simulation-/sensor_data.json") as f:
    fusion_data = json.load(f)

targets = fusion_data["targets"]

# -------------------------------------------------
# GROUP TARGETS BY CLASS
# -------------------------------------------------
targets_by_class = {}

for t in targets:
    cls = t["class"].lower()
    if cls not in targets_by_class:
        targets_by_class[cls] = []
    targets_by_class[cls].append(t)

# Track how many of each class assigned
class_assignment_counter = {cls: 0 for cls in targets_by_class.keys()}

# -------------------------------------------------
# GLOBALS
# -------------------------------------------------
object_tracker = {}
next_object_id = 1
current_objects = []
selected_object_id = None
current_display_data = {}

# -------------------------------------------------
# VIDEO + DETECTION STREAM
# -------------------------------------------------
def generate_frames():

    global object_tracker, next_object_id
    global current_objects, selected_object_id
    global current_display_data
    global class_assignment_counter

    # cap = cv2.VideoCapture("/home/srijani/AI SMARTSHIP/AI_MS2_SIMULATION/naval_dock.mp4")
    video_paths = [
   
        "/home/tractrix/Desktop/AI_SmartShip/AI_MS2-Simulation-/static/Maritime_Surveillance_Footage_Generation.mp4",
        "/home/tractrix/Desktop/AI_SmartShip/AI_MS2-Simulation-/static/Maritime_Surveillance_Footage_Generation (1).mp4",
        "/home/tractrix/Desktop/AI_SmartShip/AI_MS2-Simulation-/static/grok-video-0714534f-8b65-4cc6-8aee-e661b1eaad37 (8).mp4",
        "/home/tractrix/Desktop/AI_SmartShip/AI_MS2-Simulation-/static/grok-video-0714534f-8b65-4cc6-8aee-e661b1eaad37 (7).mp4",
        "/home/tractrix/Desktop/AI_SmartShip/AI_MS2-Simulation-/static/grok-video-0714534f-8b65-4cc6-8aee-e661b1eaad37 (6).mp4",
        "/home/tractrix/Desktop/AI_SmartShip/AI_MS2-Simulation-/static/grok-video-0714534f-8b65-4cc6-8aee-e661b1eaad37 (3).mp4",
        "/home/tractrix/Desktop/AI_SmartShip/AI_MS2-Simulation-/static/Maritime_Surveillance_Feed_Generation.mp4",
        "/home/tractrix/Desktop/AI_SmartShip/AI_MS2-Simulation-/naval_dock.mp4",
        "/home/tractrix/Desktop/AI_SmartShip/AI_MS2-Simulation-/static/Naval_Corridor_EOIR_Video_Generation.mp4"
        # "/home/tractrix/Desktop/AI_SmartShip/AI_MS2-Simulation-/static/grok1.mp4",
        # "/home/tractrix/Desktop/AI_SmartShip/AI_MS2-Simulation-/static/grok2.mp4",
        # "/home/tractrix/Desktop/AI_SmartShip/AI_MS2-Simulation-/static/grok3.mp4",
        # "/home/tractrix/Desktop/AI_SmartShip/AI_MS2-Simulation-/static/grok4.mp4",
        # "/home/tractrix/Desktop/AI_SmartShip/AI_MS2-Simulation-/static/grok5.mp4",
        # "/home/tractrix/Desktop/AI_SmartShip/AI_MS2-Simulation-/static/grok5 (2).mp4",
        # "/home/tractrix/Desktop/AI_SmartShip/AI_MS2-Simulation-/static/grok52.mp4",
        # "/home/tractrix/Desktop/AI_SmartShip/AI_MS2-Simulation-/static/grok6.mp4",
        # "/home/tractrix/Desktop/AI_SmartShip/AI_MS2-Simulation-/static/grok7.mp4"

   
   
   
]
    for video_path in video_paths:

        cap = cv2.VideoCapture(video_path)
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_objects = []

            results = model(frame, imgsz=640, conf=0.5, verbose=False)

            for r in results:
                for box in r.boxes:

                    cls_id = int(box.cls[0])
                    class_name = model.names[cls_id]
                    class_lower = class_name.lower()

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    assigned_id = None

                    # ---------------------------------
                    # SIMPLE TRACKING
                    # ---------------------------------
                    for obj_id, data in object_tracker.items():
                        prev_x, prev_y = data["center"]
                        if abs(center_x - prev_x) < 50 and abs(center_y - prev_y) < 50:
                            assigned_id = obj_id
                            object_tracker[obj_id]["center"] = (center_x, center_y)
                            break

                    # ---------------------------------
                    # NEW OBJECT
                    # ---------------------------------
                    if assigned_id is None:

                        assigned_id = next_object_id

                        sim_target = None

                        # Assign from same class list
                        if class_lower in targets_by_class:

                            class_index = class_assignment_counter[class_lower]

                            if class_index < len(targets_by_class[class_lower]):
                                sim_target = targets_by_class[class_lower][class_index]
                                class_assignment_counter[class_lower] += 1

                        object_tracker[assigned_id] = {
                            "class": class_name,
                            "center": (center_x, center_y),
                            "sim_target": sim_target
                        }

                        next_object_id += 1

                    frame_objects.append({
                        "id": assigned_id,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    })

                    # DRAW BOX
                    color = (0, 255, 255)
                    if assigned_id == selected_object_id:
                        color = (0, 255, 0)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame,
                                f"ID {assigned_id} - {class_name}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                color,
                                2)

            current_objects = frame_objects

            # -------------------------------------------------
            # FETCH SENSOR DATA FROM CLASS-MATCHED TARGET
            # -------------------------------------------------
            if selected_object_id is not None:

                if selected_object_id in object_tracker:

                    sim_target = object_tracker[selected_object_id]["sim_target"]

                    if sim_target:

                        # current_display_data = {
                        #     "id": selected_object_id,
                        #     "class": sim_target["class"],
                        #     "speed": sim_target["speed_kts"],
                        #     "course": sim_target["course_deg_T"],
                        #     "range": sim_target["range_m"],
                        #     "cpa": sim_target["cpa_m"],
                        #     "threat": sim_target["threat_level"],
                        #     "collision": sim_target["collision_status"]
                        # }
                        current_display_data = {
                            "id": selected_object_id,
                            "class": sim_target["class"],
                            "mmsi":sim_target["mmsi"],
                            
                            "bearing_degree":sim_target["bearing_deg_T"],
                            "bearing_relative_degree":sim_target["bearing_relative_deg"],

                            "speed": sim_target["speed_kts"],
                            "course": sim_target["course_deg_T"],

                            "range": sim_target["range_m"],
                            "cpa": sim_target["cpa_m"],
                            "threat": sim_target["threat_level"],
                            "collision": sim_target["collision_status"],
                            "navy_type": sim_target["navy_type"]
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