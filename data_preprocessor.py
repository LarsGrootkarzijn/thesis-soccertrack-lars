import pandas as pd
import numpy as np
import sportslabkit as slk

FRAME_RATE = 30  # frames per second

class DataPreprocessor:
    def __init__(self, annotation_csv_path):
        self.annotation_csv_path = annotation_csv_path
        self.df = None
        self.pass_frames = []
        self.deflection_frames = []
        self.filtered_angle_change_frames_df = None

    def load_and_preprocess(self):
        print(f"Loading annotation CSV: {self.annotation_csv_path}")
        df = slk.load_df(self.annotation_csv_path)

        # Extract ball columns and flatten
        ball_cols = [col for col in df.columns if col[0] == 'BALL']
        ball_df = df[ball_cols].copy()
        ball_df.insert(0, 'frame', range(1, len(ball_df) + 1))
        new_columns = ['frame'] + [col[2] for col in ball_cols]
        ball_df.columns = new_columns
        self.df = ball_df

    def compute_features(self):
        df = self.df
        df['dx'] = df['bb_left'].diff()
        df['dy'] = df['bb_top'].diff()
        df['time_seconds'] = df['frame'] / FRAME_RATE
        df['Vx'] = df['dx'] / (1 / FRAME_RATE)
        df['Vy'] = df['dy'] / (1 / FRAME_RATE)
        df['velocity'] = np.sqrt(df['Vx'] ** 2 + df['Vy'] ** 2)
        df['dV'] = df['velocity'].diff()
        df['dt'] = df['time_seconds'].diff()
        df['acceleration'] = df['dV'] / df['dt']

    def detect_events(self):
        df = self.df

        # Thresholds
        velocity_threshold = 300.0
        acceleration_threshold = 3000
        deflection_velocity_threshold = -300
        deflection_acceleration_threshold = -4000

        pass_frames = []
        deflection_frames = []

        for i in range(1, len(df)):
            if df['velocity'][i] < deflection_velocity_threshold or df['acceleration'][i] < deflection_acceleration_threshold:
                deflection_frames.append(df['frame'][i])
                continue

            if df['velocity'][i] > velocity_threshold and df['acceleration'][i] > acceleration_threshold:
                if abs(df['Vx'][i] - df['Vx'][i - 1]) > 0 or abs(df['Vy'][i] - df['Vy'][i - 1]) > 0:
                    if (i == 1) or (df['velocity'][i - 1] <= velocity_threshold or df['acceleration'][i - 1] <= acceleration_threshold):
                        pass_frames.append(df['frame'][i])

        self.pass_frames = pass_frames
        self.deflection_frames = deflection_frames

    def analyze_angles(self):
        df = self.df
        df['angle'] = np.arctan2(df['dy'], df['dx']) * (180 / np.pi)
        df['angle_change'] = df['angle'].diff().abs()

        angle_min_change_threshold = 25.0
        angle_max_change_threshold = 180.0

        angle_change_frames = df[df['angle_change'] > angle_min_change_threshold]

        filtered_frames = []
        last_time = None
        min_time_gap = 0

        for index, row in angle_change_frames.iterrows():
            current_time = row['time_seconds']
            if last_time is None or (current_time - last_time) >= min_time_gap:
                filtered_frames.append(row)
                last_time = current_time

        filtered_df = pd.DataFrame(filtered_frames)
        self.filtered_angle_change_frames_df = filtered_df[
            (filtered_df['angle_change'] >= angle_min_change_threshold) &
            (filtered_df['angle_change'] <= angle_max_change_threshold)
        ]
