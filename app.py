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

RELINK_RATIO = 0.08
RELINK_DISTANCE = int(RELINK_RATIO * 1280)
lost_tracks = {}  # old_track_id -> {center, sim_id, last_seen}
RELINK_TIME = 60  # frames (~4 sec at 15 FPS)
next_display_id = 1

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
    "last_update": None,
    "vessel_type": "POWER"   
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
                "lat": pos.get("lat"),
                "lon":pos.get("lon"),
                "heading_deg_T": nav.get("heading"),
                "cog_deg": nav.get("cog"),
                "sog_kts": nav.get("sog"),
                "rate_of_turn": nav.get("rate_of_turn"),
                "last_update": time.time()
            })

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

def own_ship_ready():
    return (
        own_ship_state["lat"] is not None and
        own_ship_state["lon"] is not None and
        own_ship_state["cog_deg"] is not None and
        own_ship_state["sog_kts"] is not None
    )

def colregs_decision(sim_target, own_ship,rel_bearing):

    dcpa = sim_target["cpa_m"]
    tcpa = sim_target["tcpa_min"]
    # dcpa, tcpa = compute_cpa_tcpa(own_ship, sim_target)
    # rel_bearing = sim_target["bearing_relative_deg"]
    # rel_bearing =relative_bearing(own_ship, sim_target)

    own_heading = own_ship["heading_deg_T"]
    # target_course = sim_target["course_deg_T"]
    target_course = sim_target["heading_deg_T"]

    D_SAFE = 500
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


def bbox_center(x1, y1, x2, y2):
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def euclidean(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5        

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = boxAArea + boxBArea - interArea

    if union == 0:
        return 0

    return interArea / union

# -------------------------------------------------
# VIDEO + YOLO TRACKING
# -------------------------------------------------

# def generate_frames():

#     global current_objects, selected_object_id
#     global current_display_data, track_metadata
#     global frame_counter, lost_tracks, next_display_id
#     global own_ship_state

#     video_paths = [
#         "/home/tractrix/Desktop/AI_SmartShip/AI_MS2-Simulation-/static/grok1.mp4",
#         "/home/tractrix/Desktop/AI_SmartShip/AI_MS2-Simulation-/static/grok2.mp4",
#         "/home/tractrix/Desktop/AI_SmartShip/AI_MS2-Simulation-/static/grok3.mp4",
#         "/home/tractrix/Desktop/AI_SmartShip/AI_MS2-Simulation-/static/grok4.mp4",
#         "/home/tractrix/Desktop/AI_SmartShip/AI_MS2-Simulation-/static/grok5.mp4",
#         "/home/tractrix/Desktop/AI_SmartShip/AI_MS2-Simulation-/static/grok6.mp4",
#         "/home/tractrix/Desktop/AI_SmartShip/AI_MS2-Simulation-/static/grok7.mp4"
#     ]

#     for video_path in video_paths:
#         cap = cv2.VideoCapture(video_path)

#         while True:
#             success, frame = cap.read()
#             if not success:
#                 break

#             frame_counter += 1
#             frame_objects = []
#             frame_boxes = []

#             results = model.track(
#                 frame,
#                 persist=True,
#                 conf=0.35,
#                 iou=0.7,
#                 verbose=False
#             )

#             # ===============================
#             # DETECTION LOOP
#             # ===============================
#             for r in results:
#                 for box in r.boxes:

#                     if box.id is None:
#                         continue

#                     track_id = int(box.id[0])
#                     cls_id = int(box.cls[0])
#                     class_name = model.names[cls_id]

#                     x1, y1, x2, y2 = map(int, box.xyxy[0])
#                     new_box = (x1, y1, x2, y2)

#                     # Duplicate suppression (same frame)
#                     duplicate = False
#                     new_center = bbox_center(x1, y1, x2, y2)

#                     for existing_box in frame_boxes:
#                         existing_center = bbox_center(*existing_box)
#                         if iou(new_box, existing_box) > 0.6 or \
#                            euclidean(new_center, existing_center) < 50:
#                             duplicate = True
#                             break

#                     if duplicate:
#                         continue

#                     frame_boxes.append(new_box)
#                     center = new_center

#                     # =====================================================
#                     # NEW TRACK
#                     # =====================================================
#                     if track_id not in track_metadata:

#                         # -------- Try relinking ----------
#                         relinked = False

#                         for lost_id, lost_data in list(lost_tracks.items()):

#                             if frame_counter - lost_data["last_seen"] > RELINK_TIME:
#                                 del lost_tracks[lost_id]
#                                 continue

#                             if euclidean(center, lost_data["center"]) < RELINK_DISTANCE:

#                                 track_metadata[track_id] = {
#                                     "display_id": lost_data["display_id"],
#                                     "sim_id": lost_data["sim_id"],
#                                     "last_seen": frame_counter,
#                                     "center": center,
#                                     "class_history": [class_name],
#                                     "confirmed_class": class_name,
#                                     "age": 1,
#                                     "confirmed": False
#                                 }

#                                 del lost_tracks[lost_id]
#                                 relinked = True
#                                 break

#                         # -------- If not relinked ----------
#                         if not relinked:

#                             # Assign sim target FIRST
#                             sim_target_id = None
#                             for t in targets:
#                                 if t["class"].lower() == class_name.lower():
#                                     if t["object_id"] not in [
#                                         v["sim_id"] for v in track_metadata.values()
#                                     ]:
#                                         sim_target_id = t["object_id"]
#                                         break

#                             display_id = next_display_id
#                             next_display_id += 1

#                             track_metadata[track_id] = {
#                                 "display_id": display_id,
#                                 "sim_id": sim_target_id,
#                                 "last_seen": frame_counter,
#                                 "center": center,
#                                 "class_history": [class_name],
#                                 "confirmed_class": class_name,
#                                 "age": 1,
#                                 "confirmed": False
#                             }

#                     # =====================================================
#                     # EXISTING TRACK
#                     # =====================================================
#                     else:
#                         meta = track_metadata[track_id]
#                         meta["last_seen"] = frame_counter
#                         meta["center"] = center
#                         meta["age"] += 1

#                         # Confirm after 5 frames
#                         if meta["age"] >= 5:
#                             meta["confirmed"] = True

#                         # Sliding window class smoothing
#                         meta["class_history"].append(class_name)
#                         if len(meta["class_history"]) > 15:
#                             meta["class_history"].pop(0)

#                         meta["confirmed_class"] = max(
#                             set(meta["class_history"]),
#                             key=meta["class_history"].count
#                         )

#                     meta = track_metadata[track_id]

#                     # Only show confirmed tracks
#                     if not meta["confirmed"]:
#                         continue

#                     frame_objects.append({
#                         "id": track_id,
#                         "x1": x1,
#                         "y1": y1,
#                         "x2": x2,
#                         "y2": y2
#                     })

#                     # Draw
#                     color = (0, 255, 0) if track_id == selected_object_id else (0, 255, 255)
#                     display_id = meta["display_id"]

#                     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                     cv2.putText(
#                         frame,
#                         f"ID {display_id} - {meta['confirmed_class']}",
#                         (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.6,
#                         color,
#                         2
#                     )

#             current_objects = frame_objects

#             # =====================================================
#             # CLEANUP STALE TRACKS
#             # =====================================================
#             stale_ids = []

#             for tid, data in track_metadata.items():
#                 if frame_counter - data["last_seen"] > STALE_THRESHOLD:
#                     stale_ids.append(tid)

#             for sid in stale_ids:
#                 lost_tracks[sid] = {
#                     "center": track_metadata[sid]["center"],
#                     "sim_id": track_metadata[sid]["sim_id"],
#                     "display_id": track_metadata[sid]["display_id"],
#                     "last_seen": frame_counter
#                 }
#                 del track_metadata[sid]

#                 if sid == selected_object_id:
#                     selected_object_id = None
#                     current_display_data = {}

#             # Encode frame
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame_bytes = buffer.tobytes()

#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' +
#                    frame_bytes + b'\r\n')

#         cap.release()


def generate_frames():

    global current_objects, selected_object_id
    global current_display_data, track_metadata
    global frame_counter, lost_tracks, next_display_id
    global own_ship_state

    video_paths = [
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

            frame_counter += 1
            frame_objects = []
            frame_boxes = []

            results = model.track(
                frame,
                persist=True,
                conf=0.35,
                iou=0.7,
                verbose=False
            )

            # ===============================
            # DETECTION LOOP
            # ===============================
            for r in results:
                for box in r.boxes:

                    if box.id is None:
                        continue

                    track_id = int(box.id[0])
                    cls_id = int(box.cls[0])
                    class_name = model.names[cls_id]

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    new_box = (x1, y1, x2, y2)

                    # Duplicate suppression (same frame)
                    duplicate = False
                    new_center = bbox_center(x1, y1, x2, y2)

                    for existing_box in frame_boxes:
                        existing_center = bbox_center(*existing_box)
                        if iou(new_box, existing_box) > 0.6 or \
                           euclidean(new_center, existing_center) < 50:
                            duplicate = True
                            break

                    if duplicate:
                        continue

                    frame_boxes.append(new_box)
                    center = new_center

                    # =====================================================
                    # NEW TRACK
                    # =====================================================
                    if track_id not in track_metadata:

                        # -------- Try relinking ----------
                        relinked = False

                        for lost_id, lost_data in list(lost_tracks.items()):

                            if frame_counter - lost_data["last_seen"] > RELINK_TIME:
                                del lost_tracks[lost_id]
                                continue

                            if euclidean(center, lost_data["center"]) < RELINK_DISTANCE:

                                track_metadata[track_id] = {
                                    "display_id": lost_data["display_id"],
                                    "sim_id": lost_data["sim_id"],
                                    "last_seen": frame_counter,
                                    "center": center,
                                    "class_history": [class_name],
                                    "confirmed_class": class_name,
                                    "age": 1,
                                    "confirmed": False
                                }

                                del lost_tracks[lost_id]
                                relinked = True
                                break

                        # -------- If not relinked ----------
                        if not relinked:

                            # Assign sim target FIRST
                            sim_target_id = None
                            for t in targets:
                                if t["class"].lower() == class_name.lower():
                                    if t["object_id"] not in [
                                        v["sim_id"] for v in track_metadata.values()
                                    ]:
                                        sim_target_id = t["object_id"]
                                        break

                            display_id = next_display_id
                            next_display_id += 1

                            track_metadata[track_id] = {
                                "display_id": display_id,
                                "sim_id": sim_target_id,
                                "last_seen": frame_counter,
                                "center": center,
                                "class_history": [class_name],
                                "confirmed_class": class_name,
                                "age": 1,
                                "confirmed": False
                            }

                    # =====================================================
                    # EXISTING TRACK
                    # =====================================================
                    else:
                        meta = track_metadata[track_id]
                        meta["last_seen"] = frame_counter
                        meta["center"] = center
                        meta["age"] += 1

                        # Confirm after 5 frames
                        if meta["age"] >= 5:
                            meta["confirmed"] = True

                        # Sliding window class smoothing
                        meta["class_history"].append(class_name)
                        if len(meta["class_history"]) > 15:
                            meta["class_history"].pop(0)

                        meta["confirmed_class"] = max(
                            set(meta["class_history"]),
                            key=meta["class_history"].count
                        )

                    meta = track_metadata[track_id]

                    # Only show confirmed tracks
                    if not meta["confirmed"]:
                        continue

                    frame_objects.append({
                        "id": track_id,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    })

                    # Draw
                    color = (0, 255, 0) if track_id == selected_object_id else (0, 255, 255)
                    display_id = meta["display_id"]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"ID {display_id} - {meta['confirmed_class']}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )

            current_objects = frame_objects

            # =====================================================
            # CLEANUP STALE TRACKS
            # =====================================================
            stale_ids = []

            for tid, data in track_metadata.items():
                if frame_counter - data["last_seen"] > STALE_THRESHOLD:
                    stale_ids.append(tid)

            for sid in stale_ids:
                lost_tracks[sid] = {
                    "center": track_metadata[sid]["center"],
                    "sim_id": track_metadata[sid]["sim_id"],
                    "display_id": track_metadata[sid]["display_id"],
                    "last_seen": frame_counter
                }
                del track_metadata[sid]

                if sid == selected_object_id:
                    selected_object_id = None
                    current_display_data = {}
            
            if selected_object_id is not None:

                if selected_object_id in track_metadata:
                    # print("------------------------------------>",track_metadata)
                    sim_id = track_metadata[selected_object_id]["sim_id"]
                    print("----------------------------track_metadata_id",sim_id)
                    target = next((t for t in targets if t["object_id"] == sim_id), None)
                    
                    if target and own_ship_ready():

                        # --- Position difference ---
                        dx, dy = latlon_to_xy(
                            own_ship_state["lat"],
                            own_ship_state["lon"],
                            target["position_latlon"]["lat"],
                            target["position_latlon"]["lon"]
                        )

                        range_m = math.hypot(dx, dy)

                        # --- Bearings ---
                        rel_bearing = relative_bearing(
                            {
                                "position_latlon": {"lat": own_ship_state["lat"], "lon": own_ship_state["lon"]},
                                "cog_deg": own_ship_state["cog_deg"]
                            },
                            target
                        )

                        true_bearing = (rel_bearing + own_ship_state["heading_deg_T"]) % 360

                        # --- CPA / TCPA ---
                        own_for_cpa = {
                            "position_latlon": {"lat": own_ship_state["lat"], "lon": own_ship_state["lon"]},
                            "cog_deg": own_ship_state["cog_deg"],
                            "sog_kts": own_ship_state["sog_kts"],
                            "heading_deg_T": own_ship_state["heading_deg_T"],
                            "vessel_type": own_ship_state["vessel_type"]
                        }

                        target_for_cpa = target.copy()
                        dcpa, tcpa = compute_cpa_tcpa(own_for_cpa, target_for_cpa)

                        # --- Collision level ---
                        collision = "GREEN"
                        if dcpa < 500 and 0 < tcpa < 5:
                            collision = "RED"
                        elif dcpa < 1000 and 0 < tcpa < 15:
                            collision = "YELLOW"

                        # --- Threat level ---
                        if collision == "RED":
                            threat = "HIGH"
                        elif collision == "YELLOW":
                            threat = "MEDIUM"
                        else:
                            threat = "LOW"

                        # --- COLREG action ---
                        target_with_cpa = target.copy()
                        target_with_cpa["cpa_m"] = dcpa
                        target_with_cpa["tcpa_min"] = tcpa

                        action = colregs_decision(
                            target_with_cpa,
                            own_for_cpa,
                            rel_bearing,
                            
                        )

                        # --- Final data for UI ---
                        current_display_data = {
                            "id": selected_object_id,
                            "class": target["class"],
                            "mmsi": target["mmsi"],
                            "navy_type": target["navy_type"],

                            "bearing_degree": true_bearing,
                            "bearing_relative_degree": rel_bearing,

                            "speed": target["sog_kts"],
                            "course": target["cog_deg"],

                            "range": range_m,
                            "cpa": dcpa,

                            "threat": threat,
                            "collision": collision,

                            "recommended_action": action
                        }
            # Encode frame
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   frame_bytes + b'\r\n')

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