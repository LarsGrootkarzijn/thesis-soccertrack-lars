import os
from data_preprocessor import DataPreprocessor
from graph_generator import GraphGenerator
from frame_extractor import FrameExtractor

BASE_PATH = './soccertrack/top_view/'
ANNOTATIONS_DIR = os.path.join(BASE_PATH, 'annotations')
VIDEOS_DIR = os.path.join(BASE_PATH, 'videos')
OUTPUT_FRAMES_DIR = './extracted_frames'
FRAMES_TXT_PATH = './extracted_frames_list.txt'

os.makedirs(OUTPUT_FRAMES_DIR, exist_ok=True)

def process_file(annotation_file, video_file, output_prefix):
    # Data Prep
    preprocessor = DataPreprocessor(annotation_file)
    preprocessor.load_and_preprocess()
    preprocessor.compute_features()
    preprocessor.detect_events()
    preprocessor.analyze_angles()

    # Graphs
    graph = GraphGenerator(
        preprocessor.df,
        preprocessor.pass_frames,
        preprocessor.deflection_frames,
        preprocessor.filtered_angle_change_frames_df,
        output_prefix
    )
    graph.plot_velocity_acceleration()
    graph.plot_trajectory()

    # Frames
    video_output_dir = os.path.join(OUTPUT_FRAMES_DIR, os.path.basename(output_prefix))
    os.makedirs(video_output_dir, exist_ok=True)

    extractor = FrameExtractor(video_file, preprocessor.pass_frames, preprocessor.deflection_frames, video_output_dir)
    frame_paths = extractor.extract_and_save_frames()

    return frame_paths

def main():
    all_extracted_frames = []

    for filename in os.listdir(ANNOTATIONS_DIR):
        if filename.endswith('.csv'):
            annotation_file = os.path.join(ANNOTATIONS_DIR, filename)
            video_filename = filename.replace('.csv', '.mp4')
            video_file = os.path.join(VIDEOS_DIR, video_filename)

            if not os.path.exists(video_file):
                print(f"Video file {video_file} not found, skipping.")
                continue

            output_prefix = os.path.join('./graphs', os.path.splitext(filename)[0])
            os.makedirs('./graphs', exist_ok=True)

            print(f"\nProcessing {annotation_file} and {video_file}")
            frame_paths = process_file(annotation_file, video_file, output_prefix)

            all_extracted_frames.extend(frame_paths)
            all_extracted_frames.append("")  # blank line separator

    # Save all frame paths to a txt file
    with open(FRAMES_TXT_PATH, 'w') as f:
        for path in all_extracted_frames:
            f.write(f"{path}\n")

    print(f"Extraction completed. Frame list saved at {FRAMES_TXT_PATH}")

if __name__ == "__main__":
    main()
