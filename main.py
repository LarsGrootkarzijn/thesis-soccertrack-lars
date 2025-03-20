import os
import pandas as pd  # Important for combining dataframes!
from data_preprocessor import DataPreprocessor
from graph_generator import GraphGenerator
from frame_extractor import FrameExtractor

# Paths
BASE_PATH = './soccertrack/top_view/'
ANNOTATIONS_DIR = os.path.join(BASE_PATH, 'annotations')
VIDEOS_DIR = os.path.join(BASE_PATH, 'videos')
OUTPUT_FRAMES_DIR = './extracted_frames'
FRAMES_TXT_PATH = './extracted_frames_list.txt'
COMBINED_CSV_PATH = './combined_ball_data.csv'  # Final combined CSV output

# Make sure output directories exist
os.makedirs(OUTPUT_FRAMES_DIR, exist_ok=True)
os.makedirs('./graphs', exist_ok=True)

def process_file(annotation_file, video_file, output_prefix):
    # Data Preprocessing
    preprocessor = DataPreprocessor(annotation_file)
    preprocessor.load_and_preprocess()
    preprocessor.compute_features()
    preprocessor.detect_events()
    preprocessor.analyze_angles()

    # Graph generation
    graph = GraphGenerator(
        preprocessor.df,
        preprocessor.pass_frames,
        preprocessor.deflection_frames,
        preprocessor.filtered_angle_change_frames_df,
        output_prefix
    )
    graph.plot_velocity_acceleration()
    graph.plot_trajectory()

    # Frame extraction
    video_output_dir = os.path.join(OUTPUT_FRAMES_DIR, os.path.basename(output_prefix))
    os.makedirs(video_output_dir, exist_ok=True)

    extractor = FrameExtractor(
        video_file,
        preprocessor.pass_frames,
        preprocessor.deflection_frames,
        video_output_dir
    )
    frame_paths = extractor.extract_and_save_frames()

    # ----------------------
    # Create "event" column
    # ----------------------
    extracted_frames_set = set(preprocessor.pass_frames + preprocessor.deflection_frames)

    # Mark frames with event = 1 or 0
    preprocessor.df['event'] = preprocessor.df['frame'].apply(
        lambda x: 1 if x in extracted_frames_set else 0
    )

    # Return both frame paths and the updated ball dataframe
    return frame_paths, preprocessor.df.copy()

def main():
    all_extracted_frames = {}  # dictionary: annotation_filename -> list of frame paths
    combined_dfs = []  # list to collect all the ball dataframes

    # Iterate over annotation files
    for filename in os.listdir(ANNOTATIONS_DIR):
        if not filename.endswith('.csv'):
            continue

        annotation_file = os.path.join(ANNOTATIONS_DIR, filename)
        video_filename = filename.replace('.csv', '.mp4')
        video_file = os.path.join(VIDEOS_DIR, video_filename)

        if not os.path.exists(video_file):
            print(f"Video file {video_file} not found, skipping.")
            continue

        annotation_filename = filename  # full filename (e.g., F_20200220_1_0000_0030.csv)
        output_prefix = os.path.join('./graphs', os.path.splitext(annotation_filename)[0])

        print(f"\nProcessing {annotation_file} and {video_file}")

        frame_paths, ball_df = process_file(annotation_file, video_file, output_prefix)

        # Add annotation filename to the ball dataframe
        ball_df['annotation_filename'] = annotation_filename

        # Append it to our list of dataframes
        combined_dfs.append(ball_df)

        # Store the frame paths by filename
        all_extracted_frames[annotation_filename] = frame_paths

    # Save all frame numbers to the text file grouped by annotation file name
    with open(FRAMES_TXT_PATH, 'w') as f:
        for annotation_filename, frame_paths in all_extracted_frames.items():
            f.write(f"{annotation_filename}\n\n")  # Write annotation file name as header

            for path in frame_paths:
                # Extract frame number from filename
                frame_number = os.path.splitext(os.path.basename(path))[0].split('_')[-1]
                f.write(f"{frame_number}\n")

            f.write("\n")  # Blank line between different annotation files

    # Combine all the ball dataframes into one big dataframe
    if combined_dfs:
        combined_ball_data_df = pd.concat(combined_dfs, ignore_index=True)
        combined_ball_data_df.to_csv(COMBINED_CSV_PATH, index=False)
        print(f"\nCombined ball data saved to {COMBINED_CSV_PATH}")
    else:
        print("No data was processed, combined CSV not created.")

    print(f"\nExtraction completed. Frame list saved at {FRAMES_TXT_PATH}")

if __name__ == "__main__":
    main()
