Plan: Concrete Stack & Outputs
Ingest, preprocess, and undistort
FFmpeg/PyAV to stream MP4s from videos/input; trim/compress via utils/trim_compress_video.py.
OpenCV fisheye undistortion (camera_matrix, dist_coeffs saved per venue in config/venues/{venue}.yaml).
Artifact: videos/output/undistorted.mp4.
Calibration & geometry
Use click_points.py to capture image_points_px → docs/calibration/pitch_calibration.json.
Define court_points_m for futsal standard (e.g., 40x20m, boxes, center circle 3m radius); compute homography; store in config/venues/{venue}.yaml.
Artifact: docs/calibration/pitch_calibration.json; diagnostics overlay with court lines.
Detection & tracking
Players: YOLOv8x (Ultralytics) → ByteTrack MOT.
Ball: TrackNetV3 (3-frame input) → peak detection → Kalman filter; fallback: small YOLOv8n-ball head + optical flow to bridge gaps < 30 frames.
Artifacts: videos/output/debug_boxes.mp4; tracks JSON (per-frame bboxes/IDs/ball).
Team assignment & jersey OCR
Team clustering: K-Means (k=2) on HSV histograms of player crops (optionally k=3 to isolate referees).
Jersey OCR: PaddleOCR; frame-wise votes; confirm number after ≥5 consistent reads; link to track ID; non-numbered stay anonymous but tracked.
Artifacts: analytics/player_id_map.json; videos/output/debug_ids.mp4.
Events & analytics (v1)
Possession: ball within 1.5 m of player for ≥5 frames.
Pass: same-team possession change, ball traveled ≥2 m within ≤3 s.
Shot: ball speed ≥8 m/s toward goal in attacking half.
Goal: ball crosses goal line between posts; confirm via homography coordinates.
Tackles/interceptions/blocks: possession change triggered by opponent proximity <1.5 m at intercept frame.
Analytics: per-player distance/speed, heatmaps; per-team pass accuracy, shots, goals, possession.
Artifacts: analytics/events.json, analytics/player_stats.json, analytics/team_stats.json.
Visualization & packaging
Overlays: videos/output/overlay.mp4 (boxes, IDs, teams, ball trail, events).
Bird’s-eye: videos/output/birdseye.mp4 (mapped trajectories, events).
Reports: reports/match_report.json + optional HTML/PDF via Jinja2/WeasyPrint.
CLI: python main.py --config config/venues/{venue}.yaml --input GoPro_Futsal_part1.mp4 --output videos/output.
Tech Stack (pin now)
Detection: ultralytics>=8, torch>=2.
Ball: TrackNetV3 (onnx/pt) + OpenCV/NumPy; fallback YOLOv8n-ball.
Tracking: ByteTrack (lap, cython_bbox or pure Python impl).
OCR: paddleocr>=2.6.0; paddlepaddle-gpu matching CUDA.
Vision: opencv-python[contrib]>=4.8; filterpy for Kalman.
Video I/O: av>=10.0.0; ffmpeg on PATH.
Data/plots: numpy, pandas, pydantic, pyyaml, matplotlib.
Reports: jinja2, weasyprint.
Directory/Modules to create
src/pipeline.py (orchestrator)
src/detection/player_detector.py, ball_detector.py, tracking.py
src/geometry/fisheye.py, homography.py, calibration_io.py
src/identification/team_cluster.py, jersey_ocr.py, reid.py (numbered only)
src/events/possession.py, passes.py, shots.py, goals.py, tackles.py
src/analytics/heatmaps.py, distance_speed.py, stats.py
src/output/overlay.py, birdseye.py, report.py
config/default.yaml, config/venues/{venue}.yaml
docs/calibration/pitch_calibration.json (from click_points.py)
Why these choices (anchored)
YOLOv8 + ByteTrack: proven MOT; aligns with prior Roboflow-style pipeline.
TrackNetV3: addresses “YOLO ball tracking … not accurate and consistent” (requirements.md).
K-Means HSV: “separating teams using clustering worked perfectly” (requirements.md prior experiment).
PaddleOCR: “Assigning the number to the name was accurate and persistent” (requirements.md prior experiment).
Fisheye + missing corners: explicitly called out in requirements.md; click_points.py already defines point order for homography input.