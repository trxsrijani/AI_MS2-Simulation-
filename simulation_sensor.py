"""
Maritime Sensor Fusion – Realistic Sensor Physics  (Milestone 2 Enhanced v2)
═══════════════════════════════════════════════════════════════════════════════
Each sensor outputs EXACTLY what it produces in the real world:

  GPS           → own position (x,y), fix quality, HDOP
  Gyro Compass  → own heading (°T), rate-of-turn (°/min)
  Radar ARPA    → range, rel-bearing, true-bearing, course, speed,
                  CPA, TCPA, BCR (bow-crossing range), BCT (bow-crossing time),
                  derived position
  AIS           → MMSI, vessel name, type, callsign, flag, nav-status,
                  position, COG, SOG, true-heading, ROT, destination, ETA,
                  range (derived), CPA (derived), TCPA (derived)
  LRF           → range ONLY  (very accurate, no bearing from sensor)
  LIDAR         → range ONLY  (most accurate, no bearing)
  CameraDepth   → range ONLY  (least accurate; plug depth_estimation_algo here)
  CVClassifier  → ship class, confidence, navy, type, length, displacement

Cross-Validator → compares every overlapping field across sensors,
                  flags mismatches, assigns per-source trust scores

Fusion Engine   → Kalman filter on position; best-source selection for
                  course/speed/range; unified CPA/TCPA/BCR/BCT from fused track

Units convention:
  distances  : metres (m)   | displayed also in nautical miles (nm) where noted
  speed      : m/s internal | displayed in knots (kts)
  bearings   : degrees True (°T)
  CPA        : metres
  TCPA       : minutes (maritime standard)
  BCR        : metres
  BCT        : minutes
"""

import numpy as np
from math import degrees, radians, atan2, cos, sin, sqrt, pi, isfinite
np.random.seed(42)

# ─── unit helpers ──────────────────────────────────────────────────────────
MS_TO_KTS  = 1.9438445       # 1 m/s = 1.9438 kts
KTS_TO_MS  = 0.5144444       # 1 kts = 0.5144 m/s
M_TO_NM    = 1 / 1852.0      # 1 m = 0.000540 nm
MIN_TO_SEC = 60.0

def mps_to_kts(v):  return round(v * MS_TO_KTS, 1)
def kts_to_mps(v):  return v * KTS_TO_MS
def m_to_nm(d):     return round(d * M_TO_NM, 3)
def td(deg):        return round(deg % 360, 1)     # true-degrees, clamped 0-360

# ─── SHIP DATABASE (navy, type, length, displacement) ──────────────────────
SHIP_DB = {
    "Warship": ("NAV","Warship",150,5000),
    "Boat":    ("CIV","Boat",20,50),
    "Vessel":  ("COM","Vessel",250,80000),
}
NON_THREAT = {"Boat", "Vessel"}

# AIS vessel type codes (abbreviated)
AIS_TYPE = {
    "Warship": (35,"Naval Vessel"),
    "Boat":    (30,"Fishing"),
    "Vessel":  (70,"Cargo"),
}
NAV_STATUS = {0:"Underway using engine",1:"At anchor",2:"Not under command",
              3:"Restricted manoeuvrability",7:"Engaged in fishing",
              8:"Underway sailing",15:"Not defined"}
DESTINATIONS = ["MUMBAI PORT","KARACHI OUTER","COLOMBO","SINGAPORE","DUBAI",
                "ADEN GULF","GULF OF OMAN","MALACCA STR","PORT LOUIS","DJIBOUTI"]

# Vessel name map (representative)
VESSEL_NAMES = {
    "Warship": "INS WARSHIP",
    "Boat":    "FV SMALL BOAT",
    "Vessel":  "MV MERCHANT VESSEL",
}

# ═══════════════════════════════════════════════════════════════════════════
# SENSOR CLASSES
# ═══════════════════════════════════════════════════════════════════════════

class GPSSensor:
    """
    Outputs own-ship position.
    Real output: lat/lon, fix-quality, HDOP, COG, SOG (from GPS).
    Here we work in local flat-earth metres (x East, y North).
    """
    NOISE_M = 5.0   # 1-sigma position error (m)

    def measure(self, true_own_pos):
        noise = np.random.normal(0, self.NOISE_M, 2)
        pos   = true_own_pos + noise
        hdop  = round(np.random.uniform(0.8, 1.4), 2)
        return {
            "sensor":       "GPS",
            "pos_xy_m":     [round(float(pos[0]),1), round(float(pos[1]),1)],
            "fix_quality":  "RTK-DGPS",
            "hdop":         hdop,
            "accuracy_m":   round(self.NOISE_M * hdop, 1),
        }


class GyroCompass:
    """
    Outputs own-ship true heading.
    Real output: heading (°T), ROT (°/min).
    Gyro drift ~0.1-0.5° typical.
    """
    DRIFT_DEG = 0.2   # 1-sigma heading error

    def measure(self, true_heading_deg, true_rot_deg_min=0.0):
        hdg = true_heading_deg + np.random.normal(0, self.DRIFT_DEG)
        rot = true_rot_deg_min + np.random.normal(0, 0.5)
        return {
            "sensor":         "Gyro Compass",
            "heading_true_T": td(hdg),
            "rot_deg_min":    round(rot, 2),    # Rate of Turn (+ = turning right)
            "mode":           "Gyro (True North)",
        }


class RadarARPA:
    """
    ARPA Radar – outputs exactly what an ARPA radar display shows.

    Real outputs per track:
      range_m, range_nm       – slant range to target
      bearing_rel_T           – relative bearing (°) from ship's head
      bearing_true_T          – true bearing (°T)
      course_true_T           – target's true course (°T)  [ARPA computed]
      speed_kts               – target's speed (knots)     [ARPA computed]
      cpa_m, cpa_nm           – Closest Point of Approach
      tcpa_min                – Time to CPA (min) ; negative = past CPA
      bcr_m, bcr_nm           – Bow Crossing Range
      bct_min                 – Bow Crossing Time (min)
      pos_xy_m                – derived Cartesian position (own as origin)
      vector_mode             – "True" (true motion vector shown)
    """
    RANGE_NOISE_M    = 20.0    # 1-sigma range error
    BEARING_NOISE_D  = 0.5     # 1-sigma bearing error (degrees)

    def measure(self, own_pos, own_vel, own_hdg_deg,
                tgt_pos, tgt_vel, track_id=1):
        # ── True geometry ──────────────────────────────────────────────
        rel_true  = tgt_pos - own_pos
        rng_true  = float(np.linalg.norm(rel_true))
        brg_true  = degrees(atan2(rel_true[1], rel_true[0]))  # from E axis
        # Convert to navigational convention (°T from North, clockwise)
        brg_nav   = (90.0 - brg_true) % 360.0

        # ── Add sensor noise ───────────────────────────────────────────
        rng_m  = rng_true  + np.random.normal(0, self.RANGE_NOISE_M)
        rng_m  = max(rng_m, 1.0)
        brg_d  = brg_nav   + np.random.normal(0, self.BEARING_NOISE_D)

        # ── Derived position from noisy range+bearing ──────────────────
        brg_math = radians(90.0 - brg_d)
        pos_xy   = own_pos + rng_m * np.array([cos(brg_math), sin(brg_math)])

        # ── ARPA course/speed (from true velocity with noise) ──────────
        tgt_crs_rad = atan2(tgt_vel[1], tgt_vel[0])
        tgt_crs_nav = (90.0 - degrees(tgt_crs_rad)) % 360.0   # °T nav convention
        tgt_spd_kts = mps_to_kts(float(np.linalg.norm(tgt_vel)))
        crs_noise   = float(np.random.normal(0, 1.5))   # ARPA course error ~1-2°
        spd_noise   = float(np.random.normal(0, 0.3))   # ARPA speed error ~0.3 kts
        arpa_crs    = td(tgt_crs_nav + crs_noise)
        arpa_spd    = max(0.0, tgt_spd_kts + spd_noise)

        # Noisy target velocity for CPA/TCPA/BCR computation
        tgt_vel_noisy = kts_to_mps(arpa_spd) * np.array([
            cos(radians(90.0 - arpa_crs)),
            sin(radians(90.0 - arpa_crs))
        ])

        # ── CPA / TCPA ─────────────────────────────────────────────────
        cpa_m, tcpa_min = _cpa_tcpa_full(own_pos, own_vel,
                                          tgt_pos + np.random.normal(0,15,2),
                                          tgt_vel_noisy)

        # ── BCR / BCT ──────────────────────────────────────────────────
        bcr_m, bct_min = _bcr_bct(own_pos, own_vel, own_hdg_deg,
                                   tgt_pos, tgt_vel_noisy)

        # ── Relative bearing ───────────────────────────────────────────
        brg_rel = td(brg_d - own_hdg_deg)

        return {
            "sensor":           "Radar ARPA",
            "track_id":         track_id,
            "range_m":          round(rng_m, 0),
            "range_nm":         m_to_nm(rng_m),
            "bearing_rel_deg":  td(brg_rel),
            "bearing_true_T":   td(brg_d),
            "course_true_T":    arpa_crs,
            "speed_kts":        arpa_spd,
            "cpa_m":            cpa_m,
            "cpa_nm":           m_to_nm(cpa_m),
            "tcpa_min":         tcpa_min,
            "bcr_m":            bcr_m,
            "bcr_nm":           m_to_nm(abs(bcr_m)),
            "bct_min":          bct_min,
            "pos_xy_m":         [round(float(pos_xy[0]),1), round(float(pos_xy[1]),1)],
            "vector_mode":      "True Motion",
        }


class AISSensor:
    """
    AIS Transponder data – outputs exactly what an AIS receiver decodes.

    Real outputs (from Class A transponder):
      mmsi, vessel_name, callsign, flag
      ais_type_code, vessel_type_str
      nav_status
      pos_xy_m          – position (converted from lat/lon; small GPS noise)
      cog_T             – course over ground (°T)   from GPS
      sog_kts           – speed over ground (knots) from GPS
      heading_true_T    – true heading from gyro on target
      rot_deg_min       – rate of turn
      destination, eta
      range_m           – derived from own GPS and target AIS position
      cpa_m, tcpa_min   – derived/ARPA-computed on bridge display
    """
    POS_NOISE_M  = 5.0     # AIS position (GPS quality on target)
    COG_NOISE_D  = 0.3     # COG error (GPS heading noise)
    SOG_NOISE_KT = 0.1     # SOG error

    def measure(self, own_pos, own_vel, tgt_pos, tgt_vel, tgt_hdg_deg,
                mmsi, vessel_cls, destination_idx=0):
        info    = SHIP_DB.get(vessel_cls, ("UNK","Unknown",0,0))
        vtype   = info[1]
        ais_t   = AIS_TYPE.get(vessel_cls, (99,"Other"))
        nav_s   = 0 if np.linalg.norm(tgt_vel) > 0.5 else 1

        # GPS-quality position
        pos  = tgt_pos + np.random.normal(0, self.POS_NOISE_M, 2)
        spd  = mps_to_kts(float(np.linalg.norm(tgt_vel))) + np.random.normal(0,self.SOG_NOISE_KT)
        spd  = max(0.0, spd)

        crs_rad  = atan2(tgt_vel[1], tgt_vel[0])
        cog_nav  = td(90.0 - degrees(crs_rad) + np.random.normal(0, self.COG_NOISE_D))
        hdg_nav  = td(tgt_hdg_deg + np.random.normal(0, 0.5))
        rot      = round(float(np.random.normal(0, 0.3)), 2)

        rng_m    = float(np.linalg.norm(pos - own_pos))
        cpa_m, tcpa_min = _cpa_tcpa_full(own_pos, own_vel,
                                          pos, tgt_vel + np.random.normal(0,0.1,2))

        dest = DESTINATIONS[destination_idx % len(DESTINATIONS)]
        eta  = f"2026-{np.random.randint(2,12):02d}-{np.random.randint(1,28):02d} {np.random.randint(0,23):02d}:{np.random.randint(0,59):02d} UTC"

        vessel_name = VESSEL_NAMES.get(vessel_cls, vessel_cls) + f" {np.random.randint(100,999)}"

        return {
            "sensor":           "AIS",
            "mmsi":             mmsi,
            "vessel_name":      vessel_name,
            "callsign":         f"{info[0]}{np.random.randint(100,999)}",
            "flag":             info[0],
            "ais_type_code":    ais_t[0],
            "vessel_type":      ais_t[1],
            "nav_status":       NAV_STATUS[nav_s],
            "pos_xy_m":         [round(float(pos[0]),1), round(float(pos[1]),1)],
            "cog_T":            cog_nav,
            "sog_kts":          round(spd, 1),
            "heading_true_T":   hdg_nav,
            "rot_deg_min":      rot,
            "destination":      dest,
            "eta":              eta,
            "range_m":          round(rng_m, 1),
            "range_nm":         m_to_nm(rng_m),
            "cpa_m":            cpa_m,
            "cpa_nm":           m_to_nm(cpa_m),
            "tcpa_min":         tcpa_min,
        }


class LRFSensor:
    """
    Laser Range Finder – measures slant RANGE ONLY.
    No bearing, no velocity, no position from this sensor alone.
    Accuracy ~1 m (1-sigma).
    """
    NOISE_M = 1.0

    def measure(self, own_pos, tgt_pos):
        true_rng = float(np.linalg.norm(tgt_pos - own_pos))
        meas_rng = true_rng + np.random.normal(0, self.NOISE_M)
        return {
            "sensor":           "LRF",
            "range_m":          round(max(0.5, meas_rng), 2),
            "note":             "Range only – no bearing output",
        }


class LIDARSensor:
    """
    LIDAR – measures slant RANGE ONLY (most precise).
    No bearing, no velocity, no position from this sensor alone.
    Accuracy ~0.1 m (1-sigma).
    """
    NOISE_M = 0.1

    def measure(self, own_pos, tgt_pos):
        true_rng = float(np.linalg.norm(tgt_pos - own_pos))
        meas_rng = true_rng + np.random.normal(0, self.NOISE_M)
        return {
            "sensor":           "LIDAR",
            "range_m":          round(max(0.5, meas_rng), 2),
            "note":             "Range only – highest precision",
        }


class CameraDepthSensor:
    """
    Monocular depth-estimation camera – range ONLY estimate.
    Accuracy ~3-5% of range (1-sigma) for a well-trained model.
    ──────────────────────────────────────────────────────────
    INTEGRATION POINT: Replace the noise model with:
        depth_m = depth_estimation_algo(image_frame, focal_length, baseline)
    """
    NOISE_FRAC = 0.05     # 5% of range, 1-sigma

    def measure(self, own_pos, tgt_pos):
        true_rng  = float(np.linalg.norm(tgt_pos - own_pos))
        noise_m   = true_rng * self.NOISE_FRAC
        meas_rng  = true_rng + np.random.normal(0, noise_m)
        return {
            "sensor":           "CameraDepth",
            "range_m":          round(max(1.0, meas_rng), 1),
            "accuracy_pct":     f"±{self.NOISE_FRAC*100:.0f}%",
            "note":             "Replace with depth_estimation_algo() in production",
        }


class CVClassificationSensor:
    """
    Computer-vision ship-class classifier (trained CNN/ViT).
    ────────────────────────────────────────────────────────
    INTEGRATION POINT: Replace get_classification() body with:
        logits = model.predict(preprocess(image_frame))
        pred_idx = logits.argmax()
        pred_cls = CLASS_NAMES[pred_idx]
        confidence = softmax(logits)[pred_idx]
    """
    ACCURACY = 0.85     # top-1 accuracy (simulated)
    ALL_CLS  = ["Warship", "Boat", "Vessel"]

    def measure(self, true_cls):
        if np.random.rand() < self.ACCURACY:
            pred       = true_cls
            confidence = float(np.random.uniform(0.78, 0.99))
        else:
            pool  = [c for c in self.ALL_CLS if c != true_cls]
            pred  = np.random.choice(pool) if pool else true_cls
            confidence = float(np.random.uniform(0.40, 0.74))

        info = SHIP_DB.get(pred, ("UNK","UNK",0,0))
        return {
            "sensor":           "CVClassification",
            "cls":              pred,
            "confidence_pct":   round(confidence*100, 1),
            "navy":             info[0],
            "ship_type":        info[1],
            "length_m":         info[2],
            "displacement_t":   info[3],
            "is_warship":       info[1] not in NON_THREAT,
        }


# ═══════════════════════════════════════════════════════════════════════════
# GEOMETRY HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _cpa_tcpa_full(own_pos, own_vel, tgt_pos, tgt_vel):
    """
    Returns (CPA_m, TCPA_min).
    TCPA negative → already past CPA (diverging).
    CPA is current range when TCPA < 0.
    """
    rel_p = tgt_pos - own_pos
    rel_v = tgt_vel - own_vel
    v2    = float(np.dot(rel_v, rel_v))
    if v2 < 1e-8:
        return round(float(np.linalg.norm(rel_p)), 1), 0.0
    tcpa_s  = float(-np.dot(rel_p, rel_v) / v2)
    if tcpa_s < 0:
        return round(float(np.linalg.norm(rel_p)), 1), round(tcpa_s/60, 2)
    cpa_pos = rel_p + rel_v * tcpa_s
    return round(float(np.linalg.norm(cpa_pos)), 1), round(tcpa_s/60, 2)


def _bcr_bct(own_pos, own_vel, own_hdg_deg, tgt_pos, tgt_vel):
    """
    BCR – Bow Crossing Range (m):
        Signed distance along own bow axis where target crosses the bow line.
        +ve = target crosses AHEAD (bow);  –ve = target crosses ASTERN.
        |BCR| < ship_length → extremely dangerous.

    BCT – Bow Crossing Time (min):
        Time until target crosses own ship's bow heading line.
        +ve = future crossing;  –ve = already crossed.
        'INF' if target track is parallel to bow (will never cross).

    Derivation (relative motion frame, own-ship as origin):
      Let h  = unit vector along own heading (forward)
          s  = unit vector to starboard (h rotated 90° CW)
      Relative position:  r = tgt_pos – own_pos
      Relative velocity:  v = tgt_vel – own_vel
      Position along starboard axis:  y(t) = dot(r+v*t, s)
      Bow crossing when y(t) = 0:
          t_BCT = –dot(r, s) / dot(v, s)
      BCR = dot(r + v*t_BCT, h)        [distance ahead(+) or astern(–)]
    """
    hdg_rad = radians(own_hdg_deg)
    # Forward (bow) unit vector: N=0° so x=sin, y=cos in nav coords
    # But we work in math coords (x=East, y=North)
    h = np.array([sin(hdg_rad), cos(hdg_rad)])   # bow direction
    s = np.array([cos(hdg_rad), -sin(hdg_rad)])  # starboard direction

    r = tgt_pos - own_pos
    v = tgt_vel - own_vel

    denom = float(np.dot(v, s))
    if abs(denom) < 1e-6:
        # Track parallel to bow – will never cross bow line
        return float("inf"), float("inf")

    t_bct_s = -float(np.dot(r, s)) / denom
    bcr_m   = float(np.dot(r + v * t_bct_s, h))

    return round(bcr_m, 1), round(t_bct_s / 60.0, 2)


# ═══════════════════════════════════════════════════════════════════════════
# CROSS-VALIDATOR
# ═══════════════════════════════════════════════════════════════════════════

RANGE_TOL_PCT   = 5.0    # >5%  difference in range → flag
CPA_TOL_M       = 300.0  # >300 m difference in CPA → flag
TCPA_TOL_MIN    = 2.0    # >2 min difference in TCPA → flag
POS_TOL_M       = 150.0  # >150 m position discrepancy → flag

def cross_validate(radar, ais, lrf, lidar, cam_depth, cv_cls, own_hdg_deg):
    """
    Cross-checks all overlapping fields between sensors.
    Returns:
      issues   : list of flagged discrepancy strings
      trust    : dict sensor → range trust score (0-1, higher = more trusted)
      range_best : best-estimate range with source tag
    """
    issues = []
    ranges = {}

    # ── Collect ranges ─────────────────────────────────────────────────────
    if lidar:     ranges["LIDAR"]       = lidar["range_m"]
    if lrf:       ranges["LRF"]         = lrf["range_m"]
    if radar:     ranges["Radar"]       = radar["range_m"]
    if cam_depth: ranges["CameraDepth"] = cam_depth["range_m"]
    if ais:       ranges["AIS"]         = ais["range_m"]

    # Trust hierarchy weights (purely based on sensor accuracy specs)
    TRUST_W = {"LIDAR":1.0, "LRF":0.95, "Radar":0.75, "AIS":0.65, "CameraDepth":0.40}

    # ── Range cross-check ─────────────────────────────────────────────────
    ref_src = min(ranges, key=lambda k: TRUST_W.get(k,0.5) - 1) if ranges else None
    for s1, r1 in ranges.items():
        for s2, r2 in ranges.items():
            if s1 >= s2: continue
            diff_pct = abs(r1 - r2) / max(r1, r2, 1.0) * 100
            if diff_pct > RANGE_TOL_PCT:
                issues.append(
                    f"RANGE MISMATCH  {s1}={r1:.0f}m vs {s2}={r2:.0f}m "
                    f"(Δ={abs(r1-r2):.0f}m / {diff_pct:.1f}%)"
                )

    # ── CPA cross-check ───────────────────────────────────────────────────
    cpas = {}
    if radar: cpas["Radar"] = (radar["cpa_m"], radar["tcpa_min"])
    if ais:   cpas["AIS"]   = (ais["cpa_m"],   ais["tcpa_min"])
    for s1, (c1,t1) in cpas.items():
        for s2, (c2,t2) in cpas.items():
            if s1 >= s2: continue
            if abs(c1-c2) > CPA_TOL_M:
                issues.append(
                    f"CPA MISMATCH    {s1}={c1:.0f}m vs {s2}={c2:.0f}m "
                    f"(Δ={abs(c1-c2):.0f}m)"
                )
            if abs(t1-t2) > TCPA_TOL_MIN:
                issues.append(
                    f"TCPA MISMATCH   {s1}={t1:.1f}min vs {s2}={t2:.1f}min "
                    f"(Δ={abs(t1-t2):.1f}min)"
                )

    # ── Position cross-check (Radar vs AIS) ───────────────────────────────
    if radar and ais:
        rp = np.array(radar["pos_xy_m"])
        ap = np.array(ais["pos_xy_m"])
        pos_diff = float(np.linalg.norm(rp - ap))
        if pos_diff > POS_TOL_M:
            issues.append(
                f"POSITION MISMATCH  Radar vs AIS = {pos_diff:.0f}m apart "
                f"(>{POS_TOL_M:.0f}m threshold)"
            )

    # ── Classification cross-check (AIS type vs CV model) ────────────────
    if cv_cls and ais:
        cv_type  = cv_cls["ship_type"]
        ais_type = ais["vessel_type"]
        if ais_type == "Naval Vessel" and cv_type in NON_THREAT:
            issues.append(
                f"ID CONFLICT     AIS type='{ais_type}' but CV model says '{cv_type}' "
                f"[conf={cv_cls['confidence_pct']}%] → verify"
            )
        elif ais_type == "Cargo" and cv_cls["is_warship"]:
            issues.append(
                f"ID CONFLICT     AIS type='Cargo' but CV model classifies as WARSHIP "
                f"[{cv_cls['cls']} @ {cv_cls['confidence_pct']}%] → verify"
            )

    # ── Best range selection ──────────────────────────────────────────────
    if ranges:
        best_src = min(ranges, key=lambda k: 1.0 - TRUST_W.get(k, 0.5))
        range_best = {"range_m": ranges[best_src], "source": best_src,
                      "trust_score": TRUST_W.get(best_src, 0.5)}
    else:
        range_best = {"range_m": None, "source": "None", "trust_score": 0}

    if not issues:
        issues.append("✓ All sensor cross-checks PASSED – No discrepancies detected")

    return issues, TRUST_W, range_best


# ═══════════════════════════════════════════════════════════════════════════
# KALMAN FILTER
# ═══════════════════════════════════════════════════════════════════════════

class CVKalmanFilter:
    """Constant-velocity KF for target position tracking."""
    def __init__(self, dt=1.0):
        self.dt   = dt
        self.init = False
        self.x    = np.zeros((4, 1))
        self.P    = np.diag([500., 500., 50., 50.])
        self.F    = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]],dtype=float)
        self.Q    = np.diag([2., 2., 0.5, 0.5])

    def initialize(self, pos, vel=None):
        v = vel if vel is not None else np.zeros(2)
        self.x = np.array([[pos[0]],[pos[1]],[v[0]],[v[1]]])
        self.P = np.diag([100., 100., 25., 25.])
        self.init = True

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z, R):
        H = np.array([[1,0,0,0],[0,1,0,0]],dtype=float)
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

    def state(self):
        return self.x.flatten()


# ═══════════════════════════════════════════════════════════════════════════
# FUSION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class SensorFusion:
    """
    Fuses all sensor inputs into a single unambiguous track.
    Position KF uses: Radar (R=400), AIS (R=25), LRF+bearing (R=1, bearing from Radar/AIS)
    Course/speed: best from AIS(COG/SOG) > Radar(ARPA)
    Range: best from LIDAR > LRF > Radar > AIS > CamDepth
    CPA/TCPA/BCR/BCT: recomputed from fused track (most accurate)
    """
    def __init__(self):
        self.kf   = CVKalmanFilter(dt=1.0)
        self.step = 0
        # Sensor instances
        self.gps    = GPSSensor()
        self.gyro   = GyroCompass()
        self.radar  = RadarARPA()
        self.ais    = AISSensor()
        self.lrf    = LRFSensor()
        self.lidar  = LIDARSensor()
        self.cam    = CameraDepthSensor()
        self.cv     = CVClassificationSensor()

    def fuse(self, active_sensors, own_pos, own_vel, own_hdg_deg,
             tgt_pos, tgt_vel, tgt_hdg_deg,
             mmsi, vessel_cls, dest_idx=0, object_id=""):
        self.step += 1

        # ── Own-ship sensors (always active) ────────────────────────────
        gps_out   = self.gps.measure(own_pos)
        gyro_out  = self.gyro.measure(own_hdg_deg)
        own_pos_m = np.array(gps_out["pos_xy_m"])       # best own position
        own_hdg_m = gyro_out["heading_true_T"]           # measured heading

        # ── Target sensors (as specified per scenario) ──────────────────
        radar_out = None; ais_out   = None
        lrf_out   = None; lidar_out = None
        cam_out   = None; cv_out    = None

        if "Radar"       in active_sensors:
            radar_out = self.radar.measure(own_pos_m, own_vel, own_hdg_m,
                                           tgt_pos, tgt_vel)
        if "AIS"         in active_sensors:
            ais_out   = self.ais.measure(own_pos_m, own_vel, tgt_pos, tgt_vel,
                                         tgt_hdg_deg, mmsi, vessel_cls, dest_idx)
        if "LRF"         in active_sensors:
            lrf_out   = self.lrf.measure(own_pos_m, tgt_pos)
        if "LIDAR"       in active_sensors:
            lidar_out = self.lidar.measure(own_pos_m, tgt_pos)
        if "CameraDepth" in active_sensors:
            cam_out   = self.cam.measure(own_pos_m, tgt_pos)
        if "CVModel"     in active_sensors:
            cv_out    = self.cv.measure(vessel_cls)

        # ── Cross-validation ────────────────────────────────────────────
        issues, trust_w, range_best = cross_validate(
            radar_out, ais_out, lrf_out, lidar_out, cam_out, cv_out, own_hdg_m
        )

        # ── Kalman Filter – position fusion ─────────────────────────────
        #   Build position measurements with appropriate covariances
        pos_meas = []
        if radar_out:
            pos_meas.append((np.array(radar_out["pos_xy_m"]), np.eye(2)*400.))
        if ais_out:
            pos_meas.append((np.array(ais_out["pos_xy_m"]), np.eye(2)*25.))
        if not pos_meas:
            # No position-bearing sensor → use range-only (offset from last known bearing)
            # For range-only sensors we project along last known bearing from own ship
            # Using radar/AIS bearing if ever known; fall back to rough estimate
            pass

        # Seed or predict
        if not self.kf.init and pos_meas:
            # Seed velocity from AIS COG/SOG if available
            seed_vel = None
            if ais_out:
                spd_ms = kts_to_mps(ais_out["sog_kts"])
                crs_r  = radians(90.0 - ais_out["cog_T"])  # nav→math
                seed_vel = np.array([spd_ms*cos(crs_r), spd_ms*sin(crs_r)])
            elif radar_out:
                spd_ms = kts_to_mps(radar_out["speed_kts"])
                crs_r  = radians(90.0 - radar_out["course_true_T"])
                seed_vel = np.array([spd_ms*cos(crs_r), spd_ms*sin(crs_r)])
            self.kf.initialize(pos_meas[0][0], seed_vel)
        elif self.kf.init:
            self.kf.predict()

        for z, R in pos_meas:
            self.kf.update(z.reshape(2,1), R)

        # ── Extract fused state ─────────────────────────────────────────
        if self.kf.init:
            st         = self.kf.state()
            fused_pos  = st[0:2]
            fused_vel  = st[2:4]
        elif pos_meas:
            fused_pos  = pos_meas[0][0]
            fused_vel  = np.zeros(2)
        else:
            fused_pos  = tgt_pos.copy()   # fallback
            fused_vel  = tgt_vel.copy()

        fused_spd_ms  = float(np.linalg.norm(fused_vel))
        fused_spd_kts = mps_to_kts(fused_spd_ms)

        # Course: prefer AIS COG (GPS-quality) over KF estimate
        if ais_out and ais_out["sog_kts"] > 0.3:
            fused_crs_T    = ais_out["cog_T"]
            fused_spd_kts  = ais_out["sog_kts"]
            fused_spd_ms   = kts_to_mps(fused_spd_kts)
            crs_r          = radians(90.0 - fused_crs_T)
            fused_vel      = fused_spd_ms * np.array([cos(crs_r), sin(crs_r)])
            crs_source     = "AIS COG/SOG (GPS-grade)"
        elif radar_out:
            fused_crs_T    = radar_out["course_true_T"]
            fused_spd_kts  = radar_out["speed_kts"]
            fused_spd_ms   = kts_to_mps(fused_spd_kts)
            crs_r          = radians(90.0 - fused_crs_T)
            fused_vel      = fused_spd_ms * np.array([cos(crs_r), sin(crs_r)])
            crs_source     = "Radar ARPA"
        else:
            crs_math       = degrees(atan2(fused_vel[1], fused_vel[0]))
            fused_crs_T    = td(90.0 - crs_math)
            crs_source     = "KF estimate"

        # ── Fused CPA / TCPA / BCR / BCT ──────────────────────────────
        own_pos_meas = np.array(gps_out["pos_xy_m"])
        fused_cpa, fused_tcpa = _cpa_tcpa_full(own_pos_meas, own_vel,
                                                fused_pos, fused_vel)
        fused_bcr, fused_bct  = _bcr_bct(own_pos_meas, own_vel, own_hdg_m,
                                          fused_pos, fused_vel)

        # ── Relative bearing from own ship to fused position ────────────
        diff        = fused_pos - own_pos_meas
        brg_math    = degrees(atan2(diff[1], diff[0]))
        brg_true    = td(90.0 - brg_math)
        brg_rel     = td(brg_true - own_hdg_m)
        fused_rng_m = float(np.linalg.norm(diff))

        # ── Collision / threat assessment ─────────────────────────────
        col_sev = _collision_severity(fused_cpa, fused_tcpa)
        thr_sev = _threat_severity(fused_rng_m)

        # ── Classification fusion ─────────────────────────────────────
        if cv_out and cv_out["confidence_pct"] >= 75:
            final_cls  = cv_out["cls"]
            cls_src    = f"CVModel @ {cv_out['confidence_pct']}%"
        elif ais_out:
            final_cls  = vessel_cls
            cls_src    = f"AIS (CV {'low-conf '+str(cv_out['confidence_pct'])+'%' if cv_out else 'not active'})"
        elif cv_out:
            final_cls  = cv_out["cls"]
            cls_src    = f"CVModel @ {cv_out['confidence_pct']}% (no AIS)"
        else:
            final_cls  = "UNKNOWN"
            cls_src    = "No classification source"

        info       = SHIP_DB.get(final_cls, ("UNK","UNK",0,0))
        is_warship = info[1] not in NON_THREAT

        # ── BCR threat interpretation ──────────────────────────────────
        if not isfinite(fused_bcr):
            bcr_interp = "No bow crossing (parallel track)"
        elif fused_bcr > 0 and fused_bcr < 1000 and fused_bct > 0:
            bcr_interp = f"⚠ CROSSING AHEAD in {fused_bct:.1f} min at {fused_bcr:.0f}m (STAND-ON risk)"
        elif fused_bcr > 0:
            bcr_interp = f"Crossing ahead at {fused_bcr:.0f}m in {fused_bct:.1f} min"
        elif fused_bcr < 0 and fused_bct > 0:
            bcr_interp = f"Crossing ASTERN in {fused_bct:.1f} min at {abs(fused_bcr):.0f}m (safe)"
        else:
            bcr_interp = f"Crossed {'ahead' if fused_bcr>0 else 'astern'} {abs(fused_bct):.1f} min ago"

        output = {
            # Own ship
            "own_pos_xy_m":       gps_out["pos_xy_m"],
            "own_heading_T":      gyro_out["heading_true_T"],
            "own_rot_deg_min":    gyro_out["rot_deg_min"],
            # Fused track
            "fused_pos_xy_m":     [round(float(fused_pos[0]),1), round(float(fused_pos[1]),1)],
            "fused_range_m":      round(fused_rng_m, 1),
            "fused_range_nm":     m_to_nm(fused_rng_m),
            "fused_bearing_T":    round(brg_true, 1),
            "fused_bearing_rel":  round(brg_rel, 1),
            "fused_course_T":     round(fused_crs_T, 1),
            "fused_speed_kts":    round(fused_spd_kts, 1),
            "course_speed_src":   crs_source,
            # Safety
            "cpa_m":              fused_cpa,
            "cpa_nm":             m_to_nm(fused_cpa),
            "tcpa_min":           fused_tcpa,
            "bcr_m":              round(fused_bcr, 1) if isfinite(fused_bcr) else "∞",
            "bcr_nm":             m_to_nm(abs(fused_bcr)) if isfinite(fused_bcr) else "∞",
            "bct_min":            round(fused_bct, 2) if isfinite(fused_bct) else "∞",
            "bcr_interpretation": bcr_interp,
            "collision_severity": col_sev,
            "threat_severity":    thr_sev,
            # Classification
            "mmsi":               mmsi,
            "classification":     final_cls,
            "cls_source":         cls_src,
            "navy":               info[0],
            "ship_type":          info[1],
            "length_m":           info[2],
            "displacement_t":     info[3],
            "threat_flag":        "⚠ WARSHIP – MAINTAIN WATCH" if is_warship else "✓ Non-Threat",
            # Best range
            "best_range_m":       range_best["range_m"],
            "best_range_source":  range_best["source"],
            # Object ID
            "object_id":          object_id,
        }

        return {
            "own": {"gps": gps_out, "gyro": gyro_out},
            "target_sensors": {
                "radar":    radar_out,
                "ais":      ais_out,
                "lrf":      lrf_out,
                "lidar":    lidar_out,
                "cam_depth":cam_out,
                "cv_model": cv_out,
            },
            "cross_validation": issues,
            "fused_output":     output,
        }


def _collision_severity(cpa_m, tcpa_min):
    if tcpa_min < 0:
        return "GREEN  – Diverging / past CPA"
    if cpa_m < 500  and tcpa_min < 3:
        return "RED    ⚠ COLLISION IMMINENT"
    if cpa_m < 1000 and tcpa_min < 6:
        return "RED    ⚠ COLLISION IMMINENT"
    if cpa_m < 1500 or tcpa_min < 8:
        return "YELLOW – Collision Likely – Take action"
    return "GREEN  – Safe Passage"

def _threat_severity(rng_m):
    if rng_m < 1000:  return "HIGH  – FPM Threat Zone (<1 nm)"
    if rng_m < 5556:  return "MEDIUM – Inner CPA Zone (<3 nm)"
    return "LOW   – Outer Watch Zone"


# ═══════════════════════════════════════════════════════════════════════════
# PRETTY PRINTER
# ═══════════════════════════════════════════════════════════════════════════

def print_step(step_num, result):
    W = 110
    ts = result["target_sensors"]
    fus = result["fused_output"]
    own = result["own"]

    print(f"\n  ┌{'─'*(W-4)}┐")
    print(f"  │  STEP {step_num}  {'─'*(W-14)}│")
    print(f"  └{'─'*(W-4)}┘")

    # Own-ship sensors
    g = own["gyro"];  gp = own["gps"]
    print(f"\n  ◈ OWN SHIP  (from GPS + Gyro Compass)")
    print(f"    GPS  │ pos={gp['pos_xy_m']} m  fix={gp['fix_quality']}  HDOP={gp['hdop']}  acc≈{gp['accuracy_m']}m")
    print(f"    Gyro │ hdg={g['heading_true_T']}°T  ROT={g['rot_deg_min']}°/min  mode={g['mode']}")

    # Radar
    if ts["radar"]:
        r = ts["radar"]
        print(f"\n  ◈ RADAR ARPA  (track #{r['track_id']})")
        print(f"    Range      : {r['range_m']:.0f} m  ({r['range_nm']} nm)")
        print(f"    Bearing    : {r['bearing_rel_deg']}° Rel  |  {r['bearing_true_T']}°T")
        print(f"    Course     : {r['course_true_T']}°T   Speed : {r['speed_kts']} kts")
        print(f"    CPA        : {r['cpa_m']:.0f} m  ({r['cpa_nm']} nm)   TCPA : {r['tcpa_min']} min")
        print(f"    BCR        : {r['bcr_m']:.0f} m  ({r['bcr_nm']} nm)   BCT  : {r['bct_min']} min")
        print(f"    Position   : {r['pos_xy_m']} m  [{r['vector_mode']}]")

    # AIS
    if ts["ais"]:
        a = ts["ais"]
        print(f"\n  ◈ AIS  (Class A transponder)")
        print(f"    MMSI       : {a['mmsi']}   Name : {a['vessel_name']}")
        print(f"    Callsign   : {a['callsign']}   Flag : {a['flag']}   Type : {a['ais_type_code']} ({a['vessel_type']})")
        print(f"    Nav Status : {a['nav_status']}")
        print(f"    Position   : {a['pos_xy_m']} m")
        print(f"    COG        : {a['cog_T']}°T   SOG  : {a['sog_kts']} kts   Hdg : {a['heading_true_T']}°T   ROT : {a['rot_deg_min']}°/min")
        print(f"    Range      : {a['range_m']:.0f} m  ({a['range_nm']} nm)")
        print(f"    CPA        : {a['cpa_m']:.0f} m  ({a['cpa_nm']} nm)   TCPA : {a['tcpa_min']} min")
        print(f"    Destination: {a['destination']}   ETA : {a['eta']}")

    # Range-only sensors
    range_line = []
    if ts["lrf"]:    range_line.append(f"LRF={ts['lrf']['range_m']} m")
    if ts["lidar"]:  range_line.append(f"LIDAR={ts['lidar']['range_m']} m")
    if ts["cam_depth"]: range_line.append(f"CamDepth={ts['cam_depth']['range_m']} m ({ts['cam_depth']['accuracy_pct']})")
    if range_line:
        print(f"\n  ◈ RANGE-ONLY SENSORS (no bearing / position output)")
        print(f"    {' | '.join(range_line)}")

    # CV Model
    if ts["cv_model"]:
        cv = ts["cv_model"]
        flag = "⚠ WARSHIP" if cv["is_warship"] else "✓ Non-Threat"
        print(f"\n  ◈ CV CLASSIFICATION MODEL")
        print(f"    Class      : {cv['cls']}  ({cv['confidence_pct']}% confidence)")
        print(f"    Navy/Type  : {cv['navy']} / {cv['ship_type']}   {flag}")

    # Cross-validation
    print(f"\n  ◈ CROSS-VALIDATION REPORT")
    for issue in result["cross_validation"]:
        prefix = "    ✗ " if "MISMATCH" in issue or "CONFLICT" in issue else "    "
        print(f"{prefix}{issue}")

    # Fused output
    print(f"\n  ◈ FUSED TRACK (PDS 5-1(b) Single Unambiguous Track)")
    f = fus
    print(f"    Object ID  : {f['object_id']}")
    print(f"    Position   : {f['fused_pos_xy_m']} m")
    print(f"    Range      : {f['fused_range_m']} m  ({f['fused_range_nm']} nm)   Best from: {f['best_range_source']} ({f['best_range_m']} m)")
    print(f"    Bearing    : {f['fused_bearing_T']}°T  ({f['fused_bearing_rel']}° Rel)")
    print(f"    Course     : {f['fused_course_T']}°T   Speed : {f['fused_speed_kts']} kts   [src: {f['course_speed_src']}]")
    print(f"    CPA        : {f['cpa_m']} m  ({f['cpa_nm']} nm)   TCPA : {f['tcpa_min']} min")
    print(f"    BCR        : {f['bcr_m']} m  ({f['bcr_nm']} nm)   BCT  : {f['bct_min']} min")
    print(f"    BCR Note   : {f['bcr_interpretation']}")
    print(f"    Collision  : {f['collision_severity']}")
    print(f"    Threat     : {f['threat_severity']}")
    print(f"    ID/Class   : {f['classification']}  [{f['cls_source']}]")
    print(f"    Navy/Type  : {f['navy']} / {f['ship_type']}  L={f['length_m']}m  Disp={f['displacement_t']}t")
    print(f"    MMSI       : {f['mmsi']}   Threat Flag: {f['threat_flag']}")


def print_final(result, object_id, cls, mmsi):
    fus = result["fused_output"]
    print(f"\nFinal Fused Data for Object ID: {object_id} ({cls} {mmsi})")
    f = fus
    print(f"    Position   : {f['fused_pos_xy_m']} m")
    print(f"    Range      : {f['fused_range_m']} m  ({f['fused_range_nm']} nm)   Best from: {f['best_range_source']} ({f['best_range_m']} m)")
    print(f"    Bearing    : {f['fused_bearing_T']}°T  ({f['fused_bearing_rel']}° Rel)")
    print(f"    Course     : {f['fused_course_T']}°T   Speed : {f['fused_speed_kts']} kts   [src: {f['course_speed_src']}]")
    print(f"    CPA        : {f['cpa_m']} m  ({f['cpa_nm']} nm)   TCPA : {f['tcpa_min']} min")
    print(f"    BCR        : {f['bcr_m']} m  ({f['bcr_nm']} nm)   BCT  : {f['bct_min']} min")
    print(f"    BCR Note   : {f['bcr_interpretation']}")
    print(f"    Collision  : {f['collision_severity']}")
    print(f"    Threat     : {f['threat_severity']}")
    print(f"    ID/Class   : {f['classification']}  [{f['cls_source']}]")
    print(f"    Navy/Type  : {f['navy']} / {f['ship_type']}  L={f['length_m']}m  Disp={f['displacement_t']}t")
    print(f"    MMSI       : {f['mmsi']}   Threat Flag: {f['threat_flag']}")


def run_test(targets, active_sensors, steps, own_pos_init, own_vel_init, own_hdg_deg, skip_steps=True):
    own_pos = np.array(own_pos_init, dtype=float)
    own_vel = np.array(own_vel_init, dtype=float)

    tgt_list = []
    for i, t in enumerate(targets):
        object_id = f"target_{i+1:03d}"
        tgt_list.append({
            "pos": np.array(t["pos_init"], dtype=float),
            "vel": np.array(t["vel_init"], dtype=float),
            "hdg_deg": t["hdg_deg"],
            "cls": t["cls"],
            "mmsi": t["mmsi"],
            "dest_idx": t["dest_idx"],
            "object_id": object_id,
        })

    W = 112
    print(f"\n{'═'*W}")
    print(f"  SIMULATION WITH {len(targets)} TARGETS")
    print(f"  Sensors     : {active_sensors}")
    own_pos_fmt = [round(float(v), 1) for v in own_pos]
    own_vel_fmt = [round(float(v), 2) for v in own_vel]
    print(f"  Own ship    : pos={own_pos_fmt} m  vel={own_vel_fmt} m/s ({mps_to_kts(float(np.linalg.norm(own_vel)))} kts)  hdg={own_hdg_deg}°T")
    for i, t in enumerate(targets):
        tgt_pos_fmt = [round(float(v), 1) for v in t["pos_init"]]
        tgt_vel_fmt = [round(float(v), 2) for v in t["vel_init"]]
        print(f"  Target {i+1} (ID: {tgt_list[i]['object_id']}) : {t['cls']}  MMSI={t['mmsi']}")
        print(f"                pos={tgt_pos_fmt} m  vel={tgt_vel_fmt} m/s ({mps_to_kts(float(np.linalg.norm(t['vel_init'])))} kts)  hdg={t['hdg_deg']}°T")
    print(f"{'═'*W}")

    fusions = [SensorFusion() for _ in targets]
    final_results = [None] * len(targets)

    for step in range(steps):
        own_pos = own_pos + own_vel
        for i, tgt in enumerate(tgt_list):
            tgt["pos"] = tgt["pos"] + tgt["vel"]
            result  = fusions[i].fuse(
                active_sensors, own_pos, own_vel, own_hdg_deg,
                tgt["pos"], tgt["vel"], tgt["hdg_deg"],
                tgt["mmsi"], tgt["cls"], tgt["dest_idx"], tgt["object_id"]
            )
            if not skip_steps:
                print(f"\n── STEP {step+1} ──")
                print(f"\nTarget {i+1} ({tgt['cls']} {tgt['mmsi']}) ID: {tgt['object_id']}")
                print_step(step+1, result)
            if step == steps - 1:
                final_results[i] = result

    if skip_steps:
        print("\n── FINAL SIMULATION DATA ──")
        for i, result in enumerate(final_results):
            print_final(result, tgt_list[i]["object_id"], tgt_list[i]["cls"], tgt_list[i]["mmsi"])


if __name__ == "__main__":
    # Hardcoded targets: Change this list to modify the simulation (e.g., 2 warships, 3 boats, 1 vessel)
    targets = []

    # Example: 2 Warships
    for i in range(5):
        cls = "Warship"
        angle = np.random.uniform(0, 2*pi)
        dist = np.random.uniform(1000, 10000)  # within range
        tgt_pos_init = [dist * cos(angle), dist * sin(angle)]

        spd = np.random.uniform(5, 15)  # kts
        dir = np.random.uniform(0, 360)
        tgt_vel_init = kts_to_mps(spd) * np.array([cos(radians(90 - dir)), sin(radians(90 - dir))])

        tgt_hdg_deg = dir  # assume heading matches course for simplicity

        mmsi = f"{cls.upper()}-{np.random.randint(100,999)}"

        dest_idx = np.random.randint(0, len(DESTINATIONS))

        targets.append({
            "pos_init": tgt_pos_init,
            "vel_init": tgt_vel_init,
            "hdg_deg": tgt_hdg_deg,
            "cls": cls,
            "mmsi": mmsi,
            "dest_idx": dest_idx,
        })

    # Example: Add 3 Boats (uncomment and adjust as needed)
    for i in range(5):
        cls = "Boat"
        angle = np.random.uniform(0, 2*pi)
        dist = np.random.uniform(1000, 10000)
        tgt_pos_init = [dist * cos(angle), dist * sin(angle)]
        spd = np.random.uniform(5, 15)
        dir = np.random.uniform(0, 360)
        tgt_vel_init = kts_to_mps(spd) * np.array([cos(radians(90 - dir)), sin(radians(90 - dir))])
        tgt_hdg_deg = dir
        mmsi = f"{cls.upper()}-{np.random.randint(100,999)}"
        dest_idx = np.random.randint(0, len(DESTINATIONS))
        targets.append({
            "pos_init": tgt_pos_init,
            "vel_init": tgt_vel_init,
            "hdg_deg": tgt_hdg_deg,
            "cls": cls,
            "mmsi": mmsi,
            "dest_idx": dest_idx,
        })

    # Example: Add 1 Vessel (uncomment and adjust as needed)
    cls = "Vessel"
    angle = np.random.uniform(0, 2*pi)
    dist = np.random.uniform(1000, 10000)
    tgt_pos_init = [dist * cos(angle), dist * sin(angle)]
    spd = np.random.uniform(5, 15)
    dir = np.random.uniform(0, 360)
    tgt_vel_init = kts_to_mps(spd) * np.array([cos(radians(90 - dir)), sin(radians(90 - dir))])
    tgt_hdg_deg = dir
    mmsi = f"{cls.upper()}-{np.random.randint(100,999)}"
    dest_idx = np.random.randint(0, len(DESTINATIONS))
    targets.append({
        "pos_init": tgt_pos_init,
        "vel_init": tgt_vel_init,
        "hdg_deg": tgt_hdg_deg,
        "cls": cls,
        "mmsi": mmsi,
        "dest_idx": dest_idx,
    })

    # Own ship parameters (fixed for simulation)
    own_pos_init = [0, 0]
    own_vel_init = kts_to_mps(10) * np.array([0, 1])  # 10 kts north
    own_hdg_deg = 0  # north

    active_sensors = ["Radar", "AIS", "LRF", "LIDAR", "CameraDepth", "CVModel"]
    steps = 3

    run_test(targets, active_sensors, steps, own_pos_init, own_vel_init, own_hdg_deg, skip_steps=True)

