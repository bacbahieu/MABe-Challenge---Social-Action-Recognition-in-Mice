import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patheffects as pe
from matplotlib.widgets import TextBox
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

# =============================================================================
# 1. CONFIGURATION (GIỮ NGUYÊN CM/FRAME)
# =============================================================================
class Config:
    LAB_ID = "UppityFerret"
    VIDEO_ID = 1960237444
    
    BASE_DIR = r"MABe-mouse-behavior-detection"
    TRACKING_FILE = f"{BASE_DIR}\\train_tracking\\{LAB_ID}\\{VIDEO_ID}.parquet"
    ANN_FOLDER = Path(f"{BASE_DIR}\\train_annotation\\{LAB_ID}\\{VIDEO_ID}.parquet")
    TRAIN_CSV = f"{BASE_DIR}\\train.csv"
    
    ID_MAPPING = {1: 1, 2: 2, 3: 3, 4: 4}
    
    FRAME_MIN = 0
    FRAME_MAX = None
    SIGMA = 1 
    
    PIXELS_PER_CM = None
    VIDEO_FPS = None
    
    VELOCITY_BODYPART = "body_center"
    
    # --- CẤU HÌNH CHO CM/FRAME ---
    MAX_SPEED = 2.5        
    GRAPH_Y_LIMIT = 2.5    
    GRAPH_WINDOW_SIZE = 100 
    
    INITIAL_FPS = 60
    POINT_SIZE = 40 # Kích thước điểm tròn nền
    SKELETON_ALPHA = 0.6
    
    VISIBLE_BODYPARTS = [
        "nose",
        "body_center",
        "tail_base", 
        "lateral_left", "lateral_right", 
        "ear_left", "ear_right",
        "neck","hip_left", "hip_right",
        "tail_tip",
        "tail_middle_1",
        "tail_middle_2",
        "spine_1",
        "spine_2"
    ]
    
    SKELETON_CONNECTIONS = [
        {"name": "nose_body",       "points": ["nose", "body_center"],    "style": {"ls": "--", "lw": 0.5}},
       {"name": "body_tail",       "points": ["tail_base", "body_center"],  "style": {"ls": "--", "lw": 1}},
       {"name": "tail_full",       "points": ["tail_base", "tail_midpoint"],"style": {"ls": "-",  "lw": 1}},
       {"name": "ears",            "points": ["ear_left", "ear_right"],      "style": {"ls": "-",  "lw": 0.5}},
       {"name": "ears1_nose",      "points": ["ear_left", "nose"],      "style": {"ls": "-",  "lw": 1}},
       {"name": "ears2_nose",      "points": ["ear_right", "nose"],      "style": {"ls": "-",  "lw": 1}},
       {"name": "ears1_body",      "points": ["ear_left", "body_center"],      "style": {"ls": "-",  "lw": 1}},
       {"name": "ears2_body",      "points": ["ear_right", "body_center"],      "style": {"ls": "-",  "lw": 1}},
       {"name": "laterals",        "points": ["lateral_left", "lateral_right"], "style": {"ls": ":", "lw": 0.5}},
        {"name": "nose_neck",       "points": ["neck", "nose"],                  "style": {"ls": "--",  "lw": 0.5}},
       {"name": "neck_tailbase",   "points": ["neck", "tail_base"],      "style": {"ls": "-",  "lw": 1}},
       {"name": "hip1_neck",        "points": ["hip_left", "neck"], "style": {"ls": ":", "lw": 0.5}},
       {"name": "hip1_hip2",        "points": ["hip_left", "hip_right"], "style": {"ls": ":", "lw": 0.5}},
        {"name": "hip2_neck",        "points": ["hip_right", "neck"], "style": {"ls": ":", "lw": 0.5}},
       {"name": "hip2_ear2",        "points": ["hip_right", "ear_right"], "style": {"ls": ":", "lw": 0.5}},
       {"name": "hip1_ear1",        "points": ["hip_left", "ear_left"], "style": {"ls": ":", "lw": 0.5}},
    #    {"name": "tail1_tail2",        "points": ["tail_base", "tail_tip"], "style": {"ls": ":", "lw": 0.5}},
    ]

# =============================================================================
# 2. DATA PROCESSOR
# =============================================================================
class MouseDataProcessor:
    @staticmethod
    def preprocess_trajectory(group):
        group = group.sort_values("video_frame").copy()
        group['x'] = group['x'].astype(float)
        group['y'] = group['y'].astype(float)
        group[['x', 'y']] = group[['x', 'y']].ffill().bfill()
        
        if group['x'].isnull().any():
            group['velocity_cm_per_s'] = 0.0
            group['velocity_cm_per_frame'] = 0.0
            return group

        if Config.SIGMA > 0:
            try:
                group['x'] = gaussian_filter1d(group['x'].values, sigma=Config.SIGMA, mode='nearest')
                group['y'] = gaussian_filter1d(group['y'].values, sigma=Config.SIGMA, mode='nearest')
            except Exception as e:
                print(f"Lỗi làm mượt: {e}")
        
        dx = group['x'].diff()
        dy = group['y'].diff()
        distance_px = np.sqrt(dx**2 + dy**2)
        
        group['velocity_cm_per_frame'] = (distance_px / Config.PIXELS_PER_CM).fillna(0)
        group['velocity_cm_per_s'] = (group['velocity_cm_per_frame'] * Config.VIDEO_FPS).fillna(0)
        return group

    @staticmethod
    def process_dataframe(df):
        print(f"Đang xử lý dữ liệu (Gaussian Sigma={Config.SIGMA})...")
        print(f"Tính vận tốc: Scale={Config.PIXELS_PER_CM} px/cm, FPS={Config.VIDEO_FPS}")
        return df.groupby(["mouse_id", "bodypart"], group_keys=False).apply(MouseDataProcessor.preprocess_trajectory)

# =============================================================================
# 3. DATA LOADER
# =============================================================================
class DataLoader:
    @staticmethod
    def load_metadata():
        try:
            meta_df = pd.read_csv(Config.TRAIN_CSV)
            row = meta_df[(meta_df["lab_id"] == Config.LAB_ID) & (meta_df["video_id"] == Config.VIDEO_ID)]
            if row.empty: return 1024, 768, 16.0, 30.0
            row = row.iloc[0]
            return row["video_width_pix"], row["video_height_pix"], row["pix_per_cm_approx"], row["frames_per_second"]
        except Exception: return 1024, 768, 16.0, 30.0

    @staticmethod
    def load_tracking_data():
        print(f"Đọc tracking: {Config.TRACKING_FILE}")
        try:
            df = pd.read_parquet(Config.TRACKING_FILE, columns=["video_frame", "mouse_id", "bodypart", "x", "y"])
        except: return pd.DataFrame(), 0, 0
        f_min, f_max = int(df["video_frame"].min()), int(df["video_frame"].max())
        start = max(Config.FRAME_MIN, f_min)
        end = min(Config.FRAME_MAX if Config.FRAME_MAX else f_max, f_max)
        df = df[(df["video_frame"] >= start) & (df["video_frame"] <= end)].copy()
        return df, start, end

    @staticmethod
    def load_annotations(start_frame, end_frame):
        print(f"Đọc annotations: {Config.ANN_FOLDER}")
        ann_files = list(Config.ANN_FOLDER.glob("*.parquet"))
        if not ann_files: 
            if Config.ANN_FOLDER.is_file(): ann_files = [Config.ANN_FOLDER]
            else: return pd.DataFrame()
        df_list = [pd.read_parquet(p) for p in ann_files]
        if not df_list: return pd.DataFrame()
        df = pd.concat(df_list, ignore_index=True)
        return df[(df["stop_frame"] >= start_frame) & (df["start_frame"] <= end_frame)]

# =============================================================================
# 4. VISUALIZER
# =============================================================================
class BehaviorVisualizer:
    def __init__(self, df_tracking, df_annotations, video_size):
        self.df = df_tracking
        self.ann_df = df_annotations
        self.width, self.height = video_size
        
        self.frames = sorted(self.df["video_frame"].unique())
        self.frame_to_idx = {f: i for i, f in enumerate(self.frames)}
        self.mouse_ids = sorted(self.df["mouse_id"].unique())
        self.bodyparts_map = {bp: i+1 for i, bp in enumerate(sorted(self.df["bodypart"].unique()))}
        
        self.colors = {m_id: plt.cm.tab10(i % 10) for i, m_id in enumerate(self.mouse_ids)}
        self.id_map = Config.ID_MAPPING if Config.ID_MAPPING else {m: m for m in self.mouse_ids}

        # --- CACHE GRAPH DATA (CM/FRAME) ---
        print("Đang chuẩn bị dữ liệu đồ thị (cm/frame)...")
        self.graph_data = {}
        full_frames_series = pd.DataFrame({'video_frame': self.frames})
        
        for m_id in self.mouse_ids:
            m_df = self.df[(self.df['mouse_id'] == m_id) & (self.df['bodypart'] == Config.VELOCITY_BODYPART)]
            merged = pd.merge(full_frames_series, m_df[['video_frame', 'velocity_cm_per_frame']], on='video_frame', how='left')
            merged['velocity_cm_per_frame'] = merged['velocity_cm_per_frame'].fillna(0)
            self.graph_data[m_id] = merged['velocity_cm_per_frame'].values

        self.state = {
            "frame_idx": 0,
            "playing": True,
            "fps": Config.INITIAL_FPS
        }
        
        self._setup_figure()
        self._init_artists()
        self._setup_interaction()

    def _setup_figure(self):
        self.fig = plt.figure(figsize=(20, 9))
        gs = self.fig.add_gridspec(1, 3, width_ratios=[1.4, 3, 1.6], wspace=0.15)
        self.fig.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.08)

        self.ax_list = self.fig.add_subplot(gs[0])
        self.ax_viz = self.fig.add_subplot(gs[1])
        
        gs_right = gs[2].subgridspec(2, 1, height_ratios=[0.8, 1.2], hspace=0.2)
        self.ax_vel = self.fig.add_subplot(gs_right[0])
        self.ax_graph = self.fig.add_subplot(gs_right[1])

        # Viz Setup
        self.ax_viz.set_xlim(0, self.width)
        self.ax_viz.set_ylim(0, self.height)
        self.ax_viz.set_aspect("equal")
        self.ax_viz.set_xlabel("X (px)")
        self.ax_viz.set_ylabel("Y (px)")
        rect = plt.Rectangle((0, 0), self.width, self.height, fill=False, edgecolor='gray', ls='--', lw=1)
        self.ax_viz.add_patch(rect)

        self.ax_list.set_axis_off()
        self.ax_vel.set_axis_off()

        # Graph Setup (CM/FRAME)
        self.ax_graph.set_title("Velocity (cm/frame)", fontsize=10, fontweight='bold')
        self.ax_graph.set_ylim(0, Config.GRAPH_Y_LIMIT)
        self.ax_graph.set_xlabel("Frame")
        self.ax_graph.grid(True, linestyle=':', alpha=0.6)

    def _init_artists(self):
        self.artists = {
            "scatters": {},
            "skeletons": {m: {} for m in self.mouse_ids},
            "labels": {m: [] for m in self.mouse_ids}, # LIST ĐỂ CHỨA SỐ (TEXT)
            "vel_texts": {},
            "graph_lines": {},
            "graph_cursor": None
        }

        for m_id in self.mouse_ids:
            color = self.colors[m_id]
            self.artists["scatters"][m_id] = self.ax_viz.scatter(
                [], [], s=Config.POINT_SIZE, color=color, 
                edgecolors='white', linewidth=0.5, alpha=0.9, label=f"Mouse {m_id}"
            )
            for conn in Config.SKELETON_CONNECTIONS:
                line, = self.ax_viz.plot([], [], color=color, alpha=Config.SKELETON_ALPHA, **conn["style"])
                self.artists["skeletons"][m_id][conn["name"]] = line
            
            # Text hiển thị vận tốc ngay cạnh chuột
            vel_txt = self.ax_viz.text(0, 0, "", color="white", fontsize=8, fontweight='bold', ha='left', va='center', zorder=20)
            vel_txt.set_path_effects([pe.withStroke(linewidth=2, foreground='black')])
            self.artists["vel_texts"][m_id] = vel_txt

            # Graph Line
            g_line, = self.ax_graph.plot(self.frames, self.graph_data[m_id], color=color, lw=1.5, alpha=0.8, label=f"M{m_id}")
            self.artists["graph_lines"][m_id] = g_line
            
        self.ax_viz.legend(loc="upper left", fontsize=9)
        self.artists["graph_cursor"] = self.ax_graph.axvline(x=self.frames[0], color='black', ls='--', lw=1)

    def _setup_interaction(self):
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        axbox = plt.axes([0.4, 0.05, 0.15, 0.04])
        self.text_box = TextBox(axbox, "Frame:", initial=str(self.frames[0]))
        self.text_box.on_submit(self._on_submit_frame)

    def _update_frame(self, frame_idx):
        current_video_frame = self.frames[frame_idx]
        df_frame = self.df[self.df["video_frame"] == current_video_frame]
        
        self.ax_viz.set_title(f"Video: {Config.VIDEO_ID} | Frame: {current_video_frame} | FPS: {self.state['fps']}")

        # Lấy giá trị cm/frame từ cache
        mouse_velocities = {}
        for m_id in self.mouse_ids:
            vel_frame = self.graph_data[m_id][frame_idx] 
            mouse_velocities[m_id] = vel_frame

        # --- 1. LIST HÀNH ĐỘNG ---
        self.ax_list.clear()
        self.ax_list.set_axis_off()
        self.ax_list.set_title("Active Behaviors", fontsize=14, fontweight='bold', pad=15)
        
        col_action = (0.00, 0.39) 
        col_agent = (0.4, 0.28)    
        col_target = (0.69, 0.28)  
        
        self.ax_list.text(col_action[0], 0.90, "ACTION", fontweight='bold', fontsize=10, transform=self.ax_list.transAxes, ha='left')
        self.ax_list.text(col_agent[0], 0.90, "AGENT", fontweight='bold', fontsize=10, transform=self.ax_list.transAxes, ha='left')
        self.ax_list.text(col_target[0], 0.90, "TARGET", fontweight='bold', fontsize=10, transform=self.ax_list.transAxes, ha='left')
        self.ax_list.plot([0, 1], [0.88, 0.88], color='black', linewidth=1, transform=self.ax_list.transAxes, clip_on=False)

        if not self.ann_df.empty:
            current_acts = self.ann_df[(self.ann_df["start_frame"] <= current_video_frame) & (self.ann_df["stop_frame"] >= current_video_frame)]
            y_pos = 0.82
            if not current_acts.empty:
                for _, row in current_acts.iterrows():
                    mapped_agent = self.id_map.get(row['agent_id'], row['agent_id'])
                    mapped_target = self.id_map.get(row['target_id'], row['target_id'])
                    c_agent = self.colors.get(mapped_agent, 'black')
                    c_target = self.colors.get(mapped_target, 'black')
                    self.ax_list.text(col_action[0], y_pos, row['action'].upper(), fontsize=9, transform=self.ax_list.transAxes)
                    self.ax_list.text(col_agent[0], y_pos, f"M{row['agent_id']}", color=c_agent, fontweight='bold', fontsize=9, transform=self.ax_list.transAxes)
                    self.ax_list.text(col_target[0], y_pos, f"M{row['target_id']}", color=c_target, fontweight='bold', fontsize=9, transform=self.ax_list.transAxes)
                    y_pos -= 0.08
            else:
                self.ax_list.text(0.53, 0.53, "(No Action)", ha='center', color='gray', style='italic', transform=self.ax_list.transAxes)

        self.ax_list.plot([0, 1], [0.50, 0.50], color='black', linewidth=1, transform=self.ax_list.transAxes, clip_on=False)
        self.ax_list.text(0.5, 0.47, "Body Part", ha='center', fontweight='bold', fontsize=12, transform=self.ax_list.transAxes)
        legend_txt = "\n".join([f"{idx}: {bp}" for bp, idx in self.bodyparts_map.items()])
        self.ax_list.text(0.1, 0.43, legend_txt, va="top", ha="left", fontsize=9, family='monospace', transform=self.ax_list.transAxes, bbox=dict(facecolor='white', alpha=0.6))

        # --- 2. BẢNG VẬN TỐC (CM/FRAME) ---
        self.ax_vel.clear()
        self.ax_vel.set_axis_off()
        self.ax_vel.set_title("Instant Velocity", fontsize=12, fontweight='bold', color='darkred', pad=5)
        
        self.ax_vel.text(0.02, 0.90, "Mouse", fontweight='bold', fontsize=9, transform=self.ax_vel.transAxes)
        self.ax_vel.text(0.25, 0.90, "cm/frame", fontweight='bold', fontsize=9, transform=self.ax_vel.transAxes)
        self.ax_vel.text(0.70, 0.90, "Level", fontweight='bold', fontsize=9, transform=self.ax_vel.transAxes)
        self.ax_vel.plot([0, 1], [0.85, 0.85], color='black', linewidth=1, transform=self.ax_vel.transAxes, clip_on=False)
        
        y_vel = 0.75
        for m_id in self.mouse_ids:
            val = mouse_velocities.get(m_id, 0.0)
            color = self.colors[m_id]
            self.ax_vel.text(0.02, y_vel, f"M{m_id}", color=color, fontweight='bold', fontsize=10, transform=self.ax_vel.transAxes)
            self.ax_vel.text(0.25, y_vel, f"{val:.3f}", color='black', fontsize=10, transform=self.ax_vel.transAxes)
            
            bar_width = min(1.0, (val / Config.MAX_SPEED)) * 0.35
            bar_bg = plt.Rectangle((0.65, y_vel - 0.03), 0.35, 0.06, transform=self.ax_vel.transAxes, facecolor='lightgray', edgecolor='black', linewidth=0.5)
            self.ax_vel.add_patch(bar_bg)
            if bar_width > 0:
                bar = plt.Rectangle((0.65, y_vel - 0.03), bar_width, 0.06, transform=self.ax_vel.transAxes, facecolor=color, edgecolor='none', alpha=0.8)
                self.ax_vel.add_patch(bar)
            y_vel -= 0.15

        # --- 3. ĐỒ THỊ (SCROLLING) ---
        window_half = Config.GRAPH_WINDOW_SIZE // 2
        x_min = current_video_frame - window_half
        x_max = current_video_frame + window_half
        
        self.ax_graph.set_xlim(x_min, x_max)
        self.artists["graph_cursor"].set_xdata([current_video_frame])

        # --- 4. VISUAL UPDATE (ĐÃ SỬA LỖI MẤT SỐ) ---
        for m in self.mouse_ids:
            for txt in self.artists["labels"][m]: txt.remove()
            self.artists["labels"][m].clear()

        full_coords_cache = {m: {} for m in self.mouse_ids}

        for m_id in self.mouse_ids:
            mouse_data = df_frame[df_frame["mouse_id"] == m_id]
            
            # Text vận tốc (cm/frame)
            bc_data = mouse_data[mouse_data["bodypart"] == Config.VELOCITY_BODYPART]
            vel_txt = self.artists["vel_texts"][m_id]
            val = mouse_velocities.get(m_id, 0.0)
            
            if not bc_data.empty and not np.isnan(bc_data.iloc[0]['x']):
                bx, by = bc_data.iloc[0]['x'], bc_data.iloc[0]['y']
                vel_txt.set_position((bx + 15, by - 15))
                vel_txt.set_text(f"{val:.3f}") 
            else:
                vel_txt.set_text("")
            
            # --- VẼ CHẤM VÀ SỐ (KHÔI PHỤC ĐOẠN NÀY) ---
            if not mouse_data.empty:
                valid_data = mouse_data[mouse_data["bodypart"].isin(Config.VISIBLE_BODYPARTS)].dropna(subset=['x', 'y'])
                if not valid_data.empty:
                    self.artists["scatters"][m_id].set_offsets(valid_data[["x", "y"]].to_numpy())
                    
                    # ===> ĐOẠN ĐƯỢC THÊM LẠI <===
                    for _, row in valid_data.iterrows():
                        if row["bodypart"] == "body_center": continue
                        idx = self.bodyparts_map.get(row["bodypart"], 0)
                        txt = self.ax_viz.text(row["x"], row["y"], str(idx), fontsize=7, color='white', 
                                               ha='center', va='center', weight='bold', clip_on=True)
                        txt.set_path_effects([pe.withStroke(linewidth=2, foreground=self.colors[m_id])])
                        self.artists["labels"][m_id].append(txt)
                    # ==============================
                else:
                    self.artists["scatters"][m_id].set_offsets(np.empty((0, 2)))
            else:
                self.artists["scatters"][m_id].set_offsets(np.empty((0, 2)))

            full_coords_cache[m_id] = {r["bodypart"]: (r["x"], r["y"]) for _, r in mouse_data.dropna(subset=['x', 'y']).iterrows()}

            for conn in Config.SKELETON_CONNECTIONS:
                line = self.artists["skeletons"][m_id][conn["name"]]
                pts = conn["points"]
                coords = full_coords_cache[m_id]
                if all(p in coords for p in pts):
                    line.set_data([coords[p][0] for p in pts], [coords[p][1] for p in pts])
                else:
                    line.set_data([], [])

    def _anim_loop(self, _):
        if self.state["playing"]:
            self.state["frame_idx"] = (self.state["frame_idx"] + 1) % len(self.frames)
        return self._update_frame(self.state["frame_idx"])

    def _on_key_press(self, e):
        if e.key == " ": self.state["playing"] = not self.state["playing"]
        elif e.key == "right": 
            self.state["frame_idx"] = (self.state["frame_idx"] + 1) % len(self.frames)
            self._update_frame(self.state["frame_idx"])
            self.fig.canvas.draw_idle()
        elif e.key == "left":
            self.state["frame_idx"] = (self.state["frame_idx"] - 1) % len(self.frames)
            self._update_frame(self.state["frame_idx"])
            self.fig.canvas.draw_idle()
        elif e.key in [str(i) for i in range(1, 10)]:
            self.state["fps"] = int(e.key) * 10
            if hasattr(self, 'anim'): self.anim.event_source.interval = 1000 / self.state["fps"]

    def _on_submit_frame(self, text):
        try:
            val = int(text)
            if val in self.frame_to_idx:
                self.state["frame_idx"] = self.frame_to_idx[val]
                self._update_frame(self.state["frame_idx"])
                self.fig.canvas.draw_idle()
        except: pass

    def run(self):
        self.anim = animation.FuncAnimation(self.fig, self._anim_loop, interval=1000/self.state["fps"], cache_frame_data=False)
        plt.show()

if __name__ == "__main__":
    w, h, pix_per_cm, fps = DataLoader.load_metadata()
    Config.PIXELS_PER_CM = pix_per_cm
    Config.VIDEO_FPS = fps
    
    print(f"Video: {w}x{h} | Scale: {pix_per_cm} | FPS: {fps}")
    
    df_raw, start, end = DataLoader.load_tracking_data()
    if df_raw.empty: 
        print("Không tìm thấy dữ liệu.")
        exit()
    
    df_proc = MouseDataProcessor.process_dataframe(df_raw)
    df_ann = DataLoader.load_annotations(start, end)
    
    print("Khởi động Visualizer...")
    app = BehaviorVisualizer(df_proc, df_ann, (w, h))
    app.run()