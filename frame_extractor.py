import cv2
import os

class FrameExtractor:
    def __init__(self, video_path, pass_frames, deflection_frames, output_dir):
        self.video_path = video_path
        self.pass_frames = pass_frames
        self.deflection_frames = deflection_frames
        self.output_dir = output_dir

    def extract_and_save_frames(self):
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            print(f"ERROR: Cannot open video {self.video_path}")
            return []

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        extracted_frame_paths = []

        event_frames = [('pass', frame) for frame in self.pass_frames] + [('deflection', frame) for frame in self.deflection_frames]

        if not event_frames:
            print(f"No events detected in {self.video_path}.")
            return []

        for event_type, event_frame in event_frames:
            frame_index = int(event_frame) - 1  # OpenCV 0-indexed

            if frame_index < 0 or frame_index >= frame_count:
                print(f"WARNING: Frame {frame_index} out of range.")
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()

            if ret:
                filename = f"{event_type}_frame_{frame_index}.png"
                output_path = os.path.join(self.output_dir, filename)
                cv2.imwrite(output_path, frame)
                extracted_frame_paths.append(output_path)
            else:
                print(f"ERROR: Failed to read frame {frame_index}.")

        cap.release()
        return extracted_frame_paths
