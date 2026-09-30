import warnings
warnings.filterwarnings('ignore')

import os
import cv2
import csv
import torch
import argparse
import glob
import numpy as np
from collections import deque
from torchvision.transforms import ToTensor, Compose

from models.segformer.segformer import Seg

# -----------------------------
# Constants
# -----------------------------
# Confidence Thresholds
CONF_TH_GENERAL = 0.70        
CONF_TH_SKY_OUTDOOR = 0.95    
CONF_TH_SKY_INDOOR = 0.97     

FRAME_STEP = 1

# Transition Capture Settings
TRANSITION_WINDOW_SEC = 1.5   

# Class IDs
ROAD_ID = 0
SIDEWALK_ID = 1
BUILDING_ID = 2
WALL_ID = 3
FENCE_ID = 4
VEGETATION_ID = 8 
TERRAIN_ID = 9
SKY_ID = 10
VEHICLE_MERGED_ID = 13 

# Indices to merge into 'vehicle'
VEHICLE_SOURCE_INDICES = [13, 14, 15, 16, 17, 18]

# 🟢 UPDATED: Added VEHICLE_MERGED_ID to this list.
# Now Vehicles, Roads, Sidewalks, and Terrain are forbidden indoors.
OUTDOOR_ONLY_SURFACES = [ROAD_ID, SIDEWALK_ID, TERRAIN_ID, VEHICLE_MERGED_ID]

NUM_CLASSES = 19

# Geometry Config
TARGET_W = 2048
TARGET_H = 400
SMOOTH_SCALE = 0.60           

# --- PERSPECTIVE CONFIG ---
VANISHING_X_PCT = 0.45    
HORIZON_Y_PCT   = 0.42    
BOTTOM_ROAD_X_PCT = 0.75  

NAME_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'vehicle', 'truck', 'bus', 'train',
    'motorcycle', 'bicycle'
]

# Color Map
COLOR_MAP = np.array([
    [128,  0,128],   # 0 road - purple
    [255,  0,  0],   # 1 sidewalk - red
    [ 70, 70,255],   # 2 building - strong blue
    [  0,  0,  0],   # 3 wall - black
    [  0,180,180],   # 4 fence - teal
    [255,  0,255],   # 5 pole - magenta
    [  0,255,255],   # 6 traffic light - cyan
    [255, 64, 64],   # 7 traffic sign - bright red
    [200,100,  0],   # 8 vegetation - dark orange
    [  0,255,128],   # 9 terrain - mint
    [  0,255,  0],   # 10 sky - bright green
    [255,  0,128],   # 11 person - pink
    [255,128,255],   # 12 rider - light magenta
    [  0, 64,255],   # 13 vehicle - DEEP BLUE
    [  0, 64,255],   # 14 -> vehicle
    [  0, 64,255],   # 15 -> vehicle
    [  0, 64,255],   # 16 -> vehicle
    [  0, 64,255],   # 17 -> vehicle
    [  0, 64,255],   # 18 -> vehicle
], dtype=np.uint8)

# -----------------------------
# Helpers
# -----------------------------
def remove_module_prefix(state_dict):
    new_state = {}
    for k, v in state_dict.items():
        new_k = k[7:] if k.startswith("module.") else k
        new_state[new_k] = v
    return new_state

def colorize_mask(mask):
    h, w = mask.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in range(NUM_CLASSES):
        img[mask == cls] = COLOR_MAP[cls]
    return img

def overlay_mask_on_image(image_tensor, mask_rgb, alpha=0.5):
    img = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    mask = cv2.resize(mask_rgb, (img.shape[1], img.shape[0]))
    return cv2.addWeighted(mask, alpha, img, 1 - alpha, 0)

def compute_object_stats(mask, prob_map):
    h, w = mask.shape
    total_pixels = h * w
    results = []
    
    for cls in range(NUM_CLASSES):
        if cls in VEHICLE_SOURCE_INDICES and cls != VEHICLE_MERGED_ID:
            continue
            
        pixels = (mask == cls)
        area = int(pixels.sum())
        if area == 0: continue
        
        confidence = 0.0
        if prob_map is not None:
            if cls == VEHICLE_MERGED_ID:
                 confidence = 0.8 
            else:
                 confidence = float(prob_map[cls][pixels].mean())
            
        results.append({
            "class_id": cls,
            "class_name": NAME_CLASSES[cls],
            "pixel_area": area,
            "area_percent": 100.0 * area / total_pixels,
            "confidence": confidence
        })
    return results

def smooth_frame_rgb(frame_rgb, scale=0.75):
    if scale >= 0.999: return frame_rgb
    h, w, _ = frame_rgb.shape
    sw = max(2, int(w * scale))
    sh = max(2, int(h * scale))
    small = cv2.resize(frame_rgb, (sw, sh), interpolation=cv2.INTER_AREA)
    back = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return back

def clean_mask(mask, min_area=500):
    cleaned = mask.copy()
    for cls in [VEHICLE_MERGED_ID, ROAD_ID, SIDEWALK_ID]:
        binary = (cleaned == cls).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                cleaned[labels == i] = SIDEWALK_ID 
    return cleaned

def smooth_low_confidence_pixels(pred, conf_map, threshold=0.70):
    is_not_sky = (pred != SKY_ID)
    is_weak = conf_map < threshold
    target_mask = is_not_sky & is_weak
    
    if not np.any(target_mask):
        return pred

    pred_uint8 = pred.astype(np.uint8)
    smoothed_pred = cv2.medianBlur(pred_uint8, 7)
    out_pred = pred.copy()
    out_pred[target_mask] = smoothed_pred[target_mask]
    return out_pred

# -----------------------------
# Sky Cleaning
# -----------------------------
def clean_sky_strict(pred, probs, is_indoor_mode):
    cleaned = pred.copy()
    sky_prob_map = probs[SKY_ID, :, :]
    required_conf = CONF_TH_SKY_INDOOR if is_indoor_mode else CONF_TH_SKY_OUTDOOR
    
    weak_sky = (cleaned == SKY_ID) & (sky_prob_map < required_conf)
    cleaned[weak_sky] = BUILDING_ID

    binary_sky = (cleaned == SKY_ID).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_sky, 8)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if is_indoor_mode and area < 5000:
            cleaned[labels == i] = BUILDING_ID
            
    return cleaned

# -----------------------------
# INDOOR FLOOR CLEANING (Updated for Cars)
# -----------------------------
def clean_indoor_floor(pred, is_indoor_mode):
    if not is_indoor_mode:
        return pred
        
    cleaned = pred.copy()
    # 🟢 Uses the updated list: [ROAD, SIDEWALK, TERRAIN, VEHICLE]
    invalid_surfaces = np.isin(cleaned, OUTDOOR_ONLY_SURFACES)
    cleaned[invalid_surfaces] = BUILDING_ID
    
    return cleaned

# -----------------------------
# Logic Components
# -----------------------------

def refine_road_sidewalk_angular(probs, width, height, power=0.6):
    horizon_y = int(height * HORIZON_Y_PCT)
    vanishing_x = int(width * VANISHING_X_PCT)
    bottom_x = int(width * BOTTOM_ROAD_X_PCT)
    
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    valid_y_mask = (yy >= horizon_y)
    y_progress = np.zeros_like(yy, dtype=np.float32)
    if height > horizon_y:
        y_progress[valid_y_mask] = (yy[valid_y_mask] - horizon_y) / (height - horizon_y)
    
    x_line = vanishing_x + (bottom_x - vanishing_x) * y_progress
    dist_from_line = (xx - x_line) / (width * 0.5) 
    
    k = 8.0  
    road_w = 1.0 / (1.0 + np.exp(k * dist_from_line))
    sidewalk_w = 1.0 - road_w
    
    suppress_road_on_right = 1.0 - (sidewalk_w * valid_y_mask * power) 
    suppress_sidewalk_on_left = 1.0 - (road_w * valid_y_mask * power)
    
    probs[ROAD_ID, :, :]     *= suppress_road_on_right
    probs[SIDEWALK_ID, :, :] *= suppress_sidewalk_on_left
    
    return probs

def connect_vertical_road_gaps(mask):
    refined = mask.copy()
    road_mask = (refined == ROAD_ID).astype(np.uint8)
    kernel = np.ones((41, 1), np.uint8) 
    closed_road = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
    allowed_area = (refined == SIDEWALK_ID) | (refined == TERRAIN_ID)
    update_mask = (closed_road == 1) & allowed_area & (refined != ROAD_ID)
    refined[update_mask] = ROAD_ID
    return refined

# -----------------------------
# 🟢 UPDATED: Logic to Determine Scene Type (Using Vegetation & Cars)
# -----------------------------
def determine_scene_type(current_sky_pct, current_veg_pct, current_car_pct, prev_sky_pct, is_currently_outdoor):
    new_status = is_currently_outdoor
    
    # 1. INDOOR Logic (Trying to switch to INDOOR)
    if is_currently_outdoor:
        # Sky Conditions: Either extremely low OR dropped significantly
        sky_is_low = (current_sky_pct < 5.0) or ((current_sky_pct < 10.0) and (current_sky_pct < prev_sky_pct * 0.6))
        
        # We only switch to INDOOR if sky, vegetation AND cars are all low.
        veg_is_low = (current_veg_pct < 5.0)
        cars_are_low = (current_car_pct < 2.0) # 🟢 New condition
        
        if sky_is_low and veg_is_low and cars_are_low:
            new_status = False 

    # 2. OUTDOOR Logic (Trying to switch to OUTDOOR)
    elif not is_currently_outdoor:
        # Switch if Sky is high OR Vegetation is high OR Cars are present (>5%)
        if current_sky_pct >= 15.0 or current_veg_pct > 15.0 or current_car_pct > 5.0:
            new_status = True
            
    return new_status

# -----------------------------
# 🟢 UPDATED: Smart Initialization (Frame 0)
# -----------------------------
def detect_initial_state_smart(pred, probs, height):
    total_px = pred.size
    
    # 1. Vegetation Check: High vegetation -> Outdoor
    veg_px = np.sum(pred == VEGETATION_ID)
    veg_pct = (veg_px / total_px) * 100.0
    if veg_pct > 15.0:
        return True # Outdoor (nature)

    # 2. Car Check: High vehicle presence -> Outdoor (street) 🟢 New
    car_px = np.sum(pred == VEHICLE_MERGED_ID)
    car_pct = (car_px / total_px) * 100.0
    if car_pct > 5.0:
        return True # Outdoor (street with cars)

    # 3. Sky Checks
    top_limit = int(height * 0.10)
    top_slice = pred[:top_limit, :]
    sky_in_top = np.sum(top_slice == SKY_ID)
    total_top = top_slice.size
    top_sky_ratio = sky_in_top / total_top

    sky_mask = (pred == SKY_ID)
    if np.sum(sky_mask) == 0:
        return False # No sky -> Indoor

    avg_sky_conf = probs[SKY_ID][sky_mask].mean()

    if avg_sky_conf < CONF_TH_SKY_OUTDOOR:
        return False
    if top_sky_ratio < 0.25:
        return False

    return True

# -----------------------------
# Visuals
# -----------------------------
def draw_classification_border(img, is_outdoor, sky_percent):
    h, w, _ = img.shape
    color = (0, 255, 0) if is_outdoor else (0, 0, 255) 
    label = "OUTDOOR" if is_outdoor else "INDOOR"
    cv2.rectangle(img, (0, 0), (w-1, h-1), color, 15)
    cv2.rectangle(img, (0, 0), (450, 60), color, -1)
    text = f"{label} (Sky: {sky_percent:.1f}%)"
    cv2.putText(img, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    return img

def draw_legend(img, present_classes):
    h, w, _ = img.shape
    box_size = 20
    line_height = 35
    start_x = w - 220
    start_y = 20
    present_ids = sorted(list(set(present_classes)))
    overlay = img.copy()
    cv2.rectangle(overlay, (start_x - 10, 0), (w, start_y + len(present_ids) * line_height + 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    for i, cls_id in enumerate(present_ids):
        cls_name = NAME_CLASSES[cls_id]
        color_rgb = COLOR_MAP[cls_id]
        color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
        y = start_y + i * line_height
        cv2.rectangle(img, (start_x, y), (start_x + box_size, y + box_size), color_bgr, -1)
        cv2.putText(img, cls_name, (start_x + box_size + 10, y + box_size - 3), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img

# -----------------------------
# Input Generator
# -----------------------------
def get_input_generator(source_path):
    fps = 30.0 # Default fallback
    if os.path.isdir(source_path):
        files = sorted(glob.glob(os.path.join(source_path, "*")))
        valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [f for f in files if os.path.splitext(f)[1].lower() in valid_exts]
        print(f"[INFO] Folder mode: {len(image_files)} images.")
        for fpath in image_files:
            frame_bgr = cv2.imread(fpath)
            if frame_bgr is None: continue
            yield os.path.basename(fpath), frame_bgr, fps
            
    else:
        print(f"[INFO] Video mode: {source_path}")
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened(): raise RuntimeError("Failed to open video")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        idx = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret: break
            yield idx, frame_bgr, fps
            idx += 1
        cap.release()

# -----------------------------
# Main Inference Logic
# -----------------------------
def run_inference(input_source, model, device, output_dir, output_fps=None):
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames")
    transitions_base_dir = os.path.join(output_dir, "transitions")
    
    # 🟢 NEW: Add a directory specifically for colored masks
    masks_only_dir = os.path.join(output_dir, "colored_masks")
    
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(transitions_base_dir, exist_ok=True)
    os.makedirs(masks_only_dir, exist_ok=True)

    print(f"[INFO] Starting inference. Output: {output_dir}")
    
    csv_path = os.path.join(output_dir, "frame_object_stats.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame_id", "class_id", "class_name", "pixel_area", "area_percent", "confidence"])

    # 🟢 NEW: Add status CSV file to log Indoor/Outdoor transitions for the Mapper
    status_csv_path = os.path.join(output_dir, "indoor_outdoor_status.csv")
    status_file = open(status_csv_path, "w", newline="")
    status_writer = csv.writer(status_file)
    status_writer.writerow(["frame", "is_indoor"])

    transform = Compose([ToTensor()])
    
    writer = None
    is_outdoor = None 
    prev_sky_percent = 0.0
    
    history_buffer = deque() 
    recording_active = False
    record_countdown = 0
    transition_event_counter = 0
    
    process_counter = 0
    input_gen = get_input_generator(input_source)

    # 🟢 BLOCK START: Safe exit
    try:
        for frame_id, frame_bgr, source_fps in input_gen:
            
            final_fps = output_fps if output_fps is not None else 30
            buffer_len = int(final_fps * TRANSITION_WINDOW_SEC)
            
            if len(history_buffer) > buffer_len:
                history_buffer.popleft()

            if process_counter % FRAME_STEP != 0:
                process_counter += 1
                continue
            
            if process_counter % (FRAME_STEP * 10) == 0:
                print(f"[INFO] Processing frame {frame_id}")

            # --- PREDICTION ---
            frame_bgr_resized = cv2.resize(frame_bgr, (TARGET_W, TARGET_H))
            frame_rgb = cv2.cvtColor(frame_bgr_resized, cv2.COLOR_BGR2RGB)
            frame_rgb = smooth_frame_rgb(frame_rgb, scale=SMOOTH_SCALE)
            
            img_tensor = transform(frame_rgb).unsqueeze(0).to(device)

            with torch.no_grad():
                logits, _ = model(img_tensor)
            
            probs = torch.softmax(logits, dim=1)
            probs_np = probs.squeeze(0).cpu().numpy() 

            probs_np = refine_road_sidewalk_angular(probs_np, TARGET_W, TARGET_H)

            vehicle_sum_prob = np.sum(probs_np[VEHICLE_SOURCE_INDICES], axis=0)
            probs_np[VEHICLE_MERGED_ID] = vehicle_sum_prob
            for v_idx in VEHICLE_SOURCE_INDICES:
                if v_idx != VEHICLE_MERGED_ID:
                    probs_np[v_idx] = 0.0

            pred = np.argmax(probs_np, axis=0)
            conf_map = np.max(probs_np, axis=0) 
            pred = smooth_low_confidence_pixels(pred, conf_map, threshold=CONF_TH_GENERAL)
            pred = clean_mask(pred, min_area=400)

            # --- LOGIC ---
            if is_outdoor is None:
                is_outdoor = detect_initial_state_smart(pred, probs_np, TARGET_H)
                sky_px = np.sum(pred == SKY_ID)
                prev_sky_percent = (sky_px / (TARGET_W * TARGET_H)) * 100.0
                print(f"[INFO] Start State: {'OUTDOOR' if is_outdoor else 'INDOOR'}")

            pred = clean_sky_strict(pred, probs_np, is_indoor_mode=(not is_outdoor))
            pred = connect_vertical_road_gaps(pred)
            pred[pred >= 13] = VEHICLE_MERGED_ID
            
            # 🟢 NOTE: This is where Vehicles are now cleaned indoors
            pred = clean_indoor_floor(pred, is_indoor_mode=(not is_outdoor))

            # 🟢 NEW: SAVE PURE COLORED MASK FOR GPS MAPPER
            pure_mask_rgb = colorize_mask(pred)
            pure_mask_bgr = cv2.cvtColor(pure_mask_rgb, cv2.COLOR_RGB2BGR)
            # Apply same brightness scaling as overlay to ensure consistency
            pure_mask_bgr = cv2.convertScaleAbs(pure_mask_bgr, alpha=1.3, beta=0)
            out_mask_name = f"mask_{frame_id:06d}.png" if isinstance(frame_id, int) else f"{os.path.splitext(frame_id)[0]}_mask.png"
            cv2.imwrite(os.path.join(masks_only_dir, out_mask_name), pure_mask_bgr)

            stats = compute_object_stats(pred, probs_np) 
            
            current_sky_percent = 0.0
            current_veg_percent = 0.0
            current_car_percent = 0.0
            present_classes = []
            
            for s in stats:
                present_classes.append(s["class_id"])
                if s["class_id"] == SKY_ID:
                    current_sky_percent = s["area_percent"]
                if s["class_id"] == VEGETATION_ID:
                    current_veg_percent = s["area_percent"]
                if s["class_id"] == VEHICLE_MERGED_ID:
                    current_car_percent = s["area_percent"]
                    
                csv_writer.writerow([frame_id, s["class_id"], s["class_name"], s["pixel_area"],
                                     round(s["area_percent"], 4), round(s["confidence"], 4)])

            new_is_outdoor = determine_scene_type(
                current_sky_percent, 
                current_veg_percent, 
                current_car_percent, 
                prev_sky_percent, 
                is_outdoor
            )
            
            # 🟢 NEW: RECORD INDOOR/OUTDOOR STATUS
            status_writer.writerow([frame_id, not new_is_outdoor])
            
            if new_is_outdoor != is_outdoor:
                transition_event_counter += 1
                type_str = "IN_to_OUT" if new_is_outdoor else "OUT_to_IN"
                folder_name = f"event_{transition_event_counter:03d}_{type_str}"
                current_transition_dir = os.path.join(transitions_base_dir, folder_name)
                os.makedirs(current_transition_dir, exist_ok=True)
                print(f"[EVENT] Transition detected: {type_str}")
                
                for hist_id, hist_img in history_buffer:
                    fname = f"history_{hist_id}.png" if isinstance(hist_id, str) else f"frame_{hist_id:06d}.png"
                    cv2.imwrite(os.path.join(current_transition_dir, fname), hist_img)
                
                recording_active = True
                record_countdown = buffer_len 

            is_outdoor = new_is_outdoor
            prev_sky_percent = current_sky_percent

            # Visualization
            mask_rgb = colorize_mask(pred)
            mask_rgb = cv2.convertScaleAbs(mask_rgb, alpha=1.3, beta=0) 
            overlay = overlay_mask_on_image(img_tensor, mask_rgb, alpha=0.5)
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            
            overlay_bgr = draw_classification_border(overlay_bgr, is_outdoor, current_sky_percent)
            overlay_bgr = draw_legend(overlay_bgr, present_classes)

            history_buffer.append((frame_id, overlay_bgr))
            
            if recording_active:
                fname = f"future_{frame_id}.png" if isinstance(frame_id, str) else f"frame_{frame_id:06d}.png"
                cv2.imwrite(os.path.join(current_transition_dir, fname), overlay_bgr)
                record_countdown -= 1
                if record_countdown <= 0:
                    recording_active = False

            if writer is None:
                h_out, w_out, _ = overlay_bgr.shape
                print(f"[INFO] Creating video with FPS: {final_fps}")
                writer = cv2.VideoWriter(
                    os.path.join(output_dir, "segmented_video.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    final_fps, 
                    (w_out, h_out)
                )

            writer.write(overlay_bgr)
            out_name = f"frame_{frame_id:06d}.png" if isinstance(frame_id, int) else f"{os.path.splitext(frame_id)[0]}_out.png"
            cv2.imwrite(os.path.join(frames_dir, out_name), overlay_bgr)
            
            process_counter += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Saving current progress...")

    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")

    finally:
        # 🟢 BLOCK END: Safe closing
        if writer is not None:
            writer.release()
            print("[INFO] Video writer released.")
        
        if csv_file:
            csv_file.close()
            print("[INFO] Frame stats CSV file closed.")
            
        if status_file:
            status_file.close()
            print("[INFO] Indoor/Outdoor status CSV file closed.")
            
        print(f"Done. Partial or full results saved to: {output_dir}")

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_source", required=True, help="Path to video file OR directory of images")
    parser.add_argument("--output_dir", default="./video_results")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output_fps", type=float, default=None, help="Force specific FPS for the output video")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seg(backbone="mit_b2", num_classes=NUM_CLASSES, embedding_dim=512, pretrained=True)
    state_dict = remove_module_prefix(torch.load(args.weights, map_location="cpu"))
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()

    run_inference(args.input_source, model, device, args.output_dir, args.output_fps)

if __name__ == "__main__":
    main()