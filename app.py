from flask import Flask, render_template, Response, jsonify, request
import cv2
import json
from ultralytics import YOLO
import time
import math

import requests
import threading
from enum import Enum

from flask_cors import CORS

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
model = YOLO(r"/home/tractrix/Desktop/AI_SmartShip/AI_MS2-Simulation-/best5.pt")

app = Flask(__name__)
CORS(app)
# -------------------------------------------------
# LOAD SENSOR JSON
# -------------------------------------------------
with open("/home/tractrix/Desktop/AI_SmartShip/AI_MS2-Simulation-/sensor_data.json") as f:
    fusion_data = json.load(f)

targets = fusion_data["targets"]
# own_ship = fusion_data["own_ship"]
# visibility = fusion_data["simulation_meta"]["visibility"]

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
# OWN SHIP GLOBAL STATE
# -------------------------------------------------
own_ship_state = {
    "lat": None,
    "lon": None,
    "heading_deg_T": None,
    "cog_deg": None,
    "sog_kts": None,
    "rate_of_turn": None,
    "last_update": None
}

ROUTE_SIM_API = "http://192.168.59.100:5002/route_simulation_state"

def update_own_ship_loop():
    global own_ship_state

    while True:
        try:
            response = requests.get(ROUTE_SIM_API, timeout=0.2)

            data = response.json()

            live = data.get("live_state", {})
            pos = live.get("position", {})
            nav = live.get("navigation", {})

            own_ship_state.update({
                "lat": pos.get("lat_dms"),
                "lon":pos.get("lon_dms"),
                "heading_deg_T": nav.get("heading"),
                "cog_deg": nav.get("cog"),
                "sog_kts": nav.get("sog"),
                "rate_of_turn": nav.get("rate_of_turn"),
                "last_update": time.time()
            })
            print(own_ship_state)

        except Exception as e:
            print("Own ship API error:", e)

        time.sleep(0.1)  # 100ms refresh


class VesselType(Enum):
    NUC = 1
    RAM = 2
    CBD = 3
    FISHING = 4
    SAILING = 5
    POWER = 6



def rule18_priority(own_type, target_type):

    own_val = VesselType[own_type].value
    tgt_val = VesselType[target_type].value

    if own_val > tgt_val:
        return "GIVE_WAY"
    elif own_val < tgt_val:
        return "STAND_ON"
    else:
        return None



def latlon_to_xy(lat1, lon1, lat2, lon2):
    R = 6371000  # meters
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = lat2_rad - lat1_rad
    dlon = math.radians(lon2 - lon1)

    x = dlon * math.cos((lat1_rad + lat2_rad)/2) * R
    y = dlat * R
    return x, y




def normalize_angle_rad(angle):
    """Normalize angle to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def relative_bearing(own, target):
    """
    Returns relative bearing in degrees.
    +ve = starboard
    -ve = port
    """

    # --- Position difference in meters ---
    dx, dy = latlon_to_xy(
        own["position_latlon"]["lat"],
        own["position_latlon"]["lon"],
        target["position_latlon"]["lat"],
        target["position_latlon"]["lon"]
    )

    # --- True bearing from own ship to target ---
    # atan2(East, North) to match marine convention
    true_bearing = math.atan2(dx, dy)

    # --- Own ship heading ---
    own_heading = math.radians(own["cog_deg"])

    # --- Relative bearing ---
    rel_bearing_rad = normalize_angle_rad(true_bearing - own_heading)

    return math.degrees(rel_bearing_rad)
def compute_cpa_tcpa(own, target):

    # --- Position in meters ---
    dx, dy = latlon_to_xy(
        own["position_latlon"]["lat"],
        own["position_latlon"]["lon"],
        target["position_latlon"]["lat"],
        target["position_latlon"]["lon"]
    )

    # --- Convert speeds ---
    own_speed = own["sog_kts"] * 0.514444
    tgt_speed = target["sog_kts"] * 0.514444

    own_cog = math.radians(own["cog_deg"])
    tgt_cog = math.radians(target["cog_deg"])

    own_vx = own_speed * math.sin(own_cog)
    own_vy = own_speed * math.cos(own_cog)

    tgt_vx = tgt_speed * math.sin(tgt_cog)
    tgt_vy = tgt_speed * math.cos(tgt_cog)

    # --- Relative vectors ---
    rvx = tgt_vx - own_vx
    rvy = tgt_vy - own_vy

    r_dot_v = dx * rvx + dy * rvy
    v_sq = rvx**2 + rvy**2

    if v_sq == 0:
        return math.hypot(dx, dy), float("inf")

    tcpa_sec = -r_dot_v / v_sq
    cpa_x = dx + rvx * tcpa_sec
    cpa_y = dy + rvy * tcpa_sec

    dcpa = math.hypot(cpa_x, cpa_y)

    return dcpa, tcpa_sec / 60.0  # return minutes



def colregs_decision(sim_target, own_ship,rel_bearing,visibility):

    dcpa = sim_target["cpa_m"]
    tcpa = sim_target["tcpa_min"]
    # dcpa, tcpa = compute_cpa_tcpa(own_ship, sim_target)
    # rel_bearing = sim_target["bearing_relative_deg"]
    # rel_bearing =relative_bearing(own_ship, sim_target)

    own_heading = own_ship["heading_deg_T"]
    # target_course = sim_target["course_deg_T"]
    target_course = sim_target["heading_deg_T"]

    D_SAFE = 1000
    T_SAFE = 15

    if not (dcpa < D_SAFE and 0 < tcpa < T_SAFE):
        return "✓ NO COLLISION RISK – MAINTAIN COURSE"
    



    heading_diff = abs((target_course - own_heading + 180) % 360 - 180)


    if 112.5 < rel_bearing < 247.5:
        encounter = "OVERTAKING"

    # elif heading_diff > 150 and (rel_bearing < 10 or rel_bearing > 350):
    #     encounter = "HEAD_ON"
    elif heading_diff > 150 and rel_bearing < 10:
        encounter = "HEAD_ON"

    else:
        encounter = "CROSSING"



    # ----------------------------
    # 4. Rule 18 Priority
    # ----------------------------
    role = rule18_priority(own_ship["vessel_type"], sim_target["vessel_type"])

    # If same priority, use geometry rules
    if role is None:

        if encounter in ["OVERTAKING", "HEAD_ON"]:
            role = "GIVE_WAY"
        elif encounter == "CROSSING":
            if rel_bearing > 0:
                role = "GIVE_WAY"
            else:
                role = "STAND_ON"


    # if encounter in ["OVERTAKING", "HEAD_ON"]:
    #     role = "GIVE_WAY"

    # elif encounter == "CROSSING":
    #     if 0 < rel_bearing < 180:
    #         role = "GIVE_WAY"
    #     else:
    #         role = "STAND_ON"

    if role == "GIVE_WAY":
        return f"⚠ {encounter} – GIVE WAY – ALTER COURSE STARBOARD (≥20°)"

    else:
        if dcpa < 500 and 0 < tcpa < 5:
            return "⚠ STAND-ON – TAKE ACTION (Rule 17 Emergency)"
        else:
            return "✓ STAND-ON – MAINTAIN COURSE"
        

# -------------------------------------------------
# VIDEO + YOLO TRACKING
# -------------------------------------------------
def generate_frames():

    global current_objects, selected_object_id
    global current_display_data, track_metadata
    global frame_counter
    global visibility
    video_paths = [

        # "/home/tractrix/Desktop/AI_SmartShip/AI_MS2-Simulation-/static/Maritime_Surveillance_Footage_Generation.mp4",
        # "/home/tractrix/Desktop/AI_SmartShip/AI_MS2-Simulation-/static/Maritime_Surveillance_Footage_Generation (1).mp4",
        # "/home/tractrix/Desktop/AI_SmartShip/AI_MS2-Simulation-/static/grok-video-0714534f-8b65-4cc6-8aee-e661b1eaad37 (8).mp4",
        # "/home/tractrix/Desktop/AI_SmartShip/AI_MS2-Simulation-/static/grok-video-0714534f-8b65-4cc6-8aee-e661b1eaad37 (7).mp4",
        # "/home/tractrix/Desktop/AI_SmartShip/AI_MS2-Simulation-/static/grok-video-0714534f-8b65-4cc6-8aee-e661b1eaad37 (6).mp4",
        # "/home/tractrix/Desktop/AI_SmartShip/AI_MS2-Simulation-/static/grok-video-0714534f-8b65-4cc6-8aee-e661b1eaad37 (3).mp4",
        # "/home/tractrix/Desktop/AI_SmartShip/AI_MS2-Simulation-/static/Maritime_Surveillance_Feed_Generation.mp4",
        # "/home/tractrix/Desktop/AI_SmartShip/AI_MS2-Simulation-/naval_dock.mp4",
        # "/home/tractrix/Desktop/AI_SmartShip/AI_MS2-Simulation-/static/Naval_Corridor_EOIR_Video_Generation.mp4"
        "/home/tractrix/Desktop/AI_SmartShip/AI_MS2-Simulation-/static/grok1.mp4",
        "/home/tractrix/Desktop/AI_SmartShip/AI_MS2-Simulation-/static/grok2.mp4",
        "/home/tractrix/Desktop/AI_SmartShip/AI_MS2-Simulation-/static/grok3.mp4",
        "/home/tractrix/Desktop/AI_SmartShip/AI_MS2-Simulation-/static/grok4.mp4",
        "/home/tractrix/Desktop/AI_SmartShip/AI_MS2-Simulation-/static/grok5.mp4",
        "/home/tractrix/Desktop/AI_SmartShip/AI_MS2-Simulation-/static/grok6.mp4",
        "/home/tractrix/Desktop/AI_SmartShip/AI_MS2-Simulation-/static/grok7.mp4"
       
    ]

    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            time.sleep(0.07)
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

                    # if target:
                    #     current_display_data = {
                    #         "id": selected_object_id,
                    #         "class": target["class"],
                    #         "mmsi":target["mmsi"],

                    #         "bearing_degree":target["bearing_deg_T"],
                    #         "bearing_relative_degree":target["bearing_relative_deg"],

                    #         "speed": target["speed_kts"],
                    #         "course": target["course_deg_T"],

                    #         "range": target["range_m"],
                    #         "cpa": target["cpa_m"],
                    #         "threat": target["threat_level"],
                    #         "collision": target["collision_status"],
                    #         "navy_type": target["navy_type"]
                    #     }
                    if target:
                        

                        dcpa, tcpa = compute_cpa_tcpa(own_ship, target)

                        target_with_cpa = target.copy()
                        target_with_cpa["cpa_m"] = dcpa
                        target_with_cpa["tcpa_min"] = tcpa

                        # rel_bearing = sim_target["bearing_relative_deg"]
                        rel_bearing =relative_bearing(own_ship,target)
                        action = colregs_decision(target_with_cpa, own_ship,rel_bearing,visibility)

                        current_display_data = {

                            # OWN SHIP
                            "own_heading": own_ship["heading_deg_T"],
                            "own_speed": own_ship["speed_kts"],

                            # TARGET
                            "id": selected_object_id,
                            "class": target["class"],
                            "mmsi": target["mmsi"],
                            "navy_type": target["navy_type"],

                            "bearing_degree": target["bearing_deg_T"],
                            "bearing_relative_degree": target["bearing_relative_deg"],

                            "speed": target["speed_kts"],
                            "course": target["course_deg_T"],

                            "range": target["range_m"],
                            "cpa": target["cpa_m"],
                            "threat": target["threat_level"],
                            "collision": target["collision_status"],

                            # ACTION
                            "recommended_action": action
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


@app.route('/own_ship', methods=['GET'])
def get_own_ship():
    return jsonify(own_ship_state)

@app.route('/update_own_ship', methods=['POST'])
def update_own_ship():

    global own_ship_state

    data = request.json
    print(data)
    try:
        live = data.get("live_state", {})
        pos = live.get("position", {})
        nav = live.get("navigation", {})

        own_ship_state.update({
            "lat": pos.get("lat"),
            "lon": pos.get("lon"),
            "heading_deg_T": nav.get("heading"),
            "cog_deg": nav.get("cog"),
            "sog_kts": nav.get("sog"),
            "rate_of_turn": nav.get("rate_of_turn"),
            "last_update": time.time()
        })

        return jsonify({"status": "stored"})

    except Exception as e:
        return jsonify({"error": str(e)}), 400
# -------------------------------------------------
# RUN
# -------------------------------------------------
if __name__ == "__main__":
    threading.Thread(target=update_own_ship_loop, daemon=True).start()
    app.run(debug=True,port="5002", threaded=True)