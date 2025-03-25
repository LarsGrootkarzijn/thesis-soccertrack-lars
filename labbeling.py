import os
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import messagebox
from tkinter import filedialog
import sportslabkit as slk
import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
import csv


# Define the frame rate
FRAME_RATE = 30  # frames per second

class DataPreprocessor:
    def __init__(self, annotation_csv_path):
        self.annotation_csv_path = annotation_csv_path
        self.df = None
        self.pass_frames = []
        self.deflection_frames = []
        self.filtered_angle_change_frames_df = None

    def load_and_preprocess(self):
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

        # Calculate angle based on the ball's movement
        df['angle'] = np.arctan2(df['dy'], df['dx']) * (180 / np.pi)  # Angle in degrees
        self.df = df  # Ensure that angle is part of the DataFrame

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

class LabelingUI:
    def __init__(self, annotation_folder):
        self.annotation_folder = annotation_folder
        self.file_list = sorted(os.listdir(annotation_folder))  # List of annotation files
        self.current_file_index = 0
        self.df = None
        self.pass_frames = []
        self.deflection_frames = []
        self.preprocessor = None
        self.event_times = {}  # Dictionary to store event times for each file
        self.event_types = ["Pass", "Deflection", "Shot", "Other"]  # Event types to choose from
        self.selected_event_types = {}  # Dictionary to store event types for each event time

        
        # Load previously saved event types if available

        # Initialize Tkinter
        self.window = tk.Tk()
        self.window.title("Soccer Annotation Tool")

        # Create the top bar frame (only for the graph area)
        self.top_bar_frame = tk.Frame(self.window)
        self.top_bar_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        # Create the main content frame (for plot and sidebar)
        self.main_frame = tk.Frame(self.window)
        self.main_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")

        # Create the plot frame (left side of the window)
        self.plot_frame = tk.Frame(self.main_frame)
        self.plot_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Create the sidebar frame (right side of the window)
        self.sidebar_frame = tk.Frame(self.main_frame, width=200, bg="lightgray")
        self.sidebar_frame.grid(row=0, column=1, sticky="ns", padx=10, pady=10)

        # Load File, Previous, and Next buttons on the top bar
        self.load_file_button = tk.Button(self.top_bar_frame, text="Load File", command=self.load_file)
        self.load_file_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.prev_button = tk.Button(self.top_bar_frame, text="Previous", command=self.prev_file)
        self.prev_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.next_button = tk.Button(self.top_bar_frame, text="Next", command=self.next_file)
        self.next_button.pack(side=tk.LEFT, padx=10, pady=10)

        # Add Export Annotations button
        self.export_button = tk.Button(self.top_bar_frame, text="Export Annotations", command=self.export_annotations_to_csv)
        self.export_button.pack(side=tk.LEFT, padx=10, pady=10)
        # Add Load Annotations button
        self.load_annotations_button = tk.Button(self.top_bar_frame, text="Load Annotations", command=self.load_annotations_from_csv)
        self.load_annotations_button.pack(side=tk.LEFT, padx=10, pady=10)

        # Label to show current filename in the top bar
        self.filename_label = tk.Label(self.top_bar_frame, text="", font=("Helvetica", 14))
        self.filename_label.pack(side=tk.LEFT, padx=10)

        # Create figure and axis for plotting
        self.fig, self.axs = plt.subplots(3, 1, figsize=(14, 12))

        # Add click event to the plot (to add events manually)
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        self.load_file()
        
    def export_annotations_to_csv(self):
        # Create a Tkinter root window (it will not be shown)
        root = tk.Tk()
        root.withdraw()  # Hide the root window since we just need the dialog

        # Ask the user for a file path to save the CSV
        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv",  # Default extension
            filetypes=[("CSV Files", "*.csv")],  # File type options
            title="Save Annotations As"
        )

        rows = []

        for filename, events in self.event_times.items():
        # Iterate through each event (time and event type) in the list of events for the current file
            for event in events:
                event_time, event_type = event
                
                # Find the frame corresponding to this event time
                event_frame = self.df[self.df['time_seconds'] >= event_time].iloc[0]
                frame_number = event_frame['frame']
                
                # Append the frame number, event type, and filename to the rows list
                rows.append([frame_number, event_type, filename])
        
        df = pd.DataFrame(rows, columns=["frame", "event_type", "filename"])

        if save_path:  # Only save if a path was selected
            df.to_csv(save_path, index=False)

    def load_annotations_from_csv(self):
        # Create a Tkinter root window (it will not be shown)
        root = tk.Tk()
        root.withdraw()  # Hide the root window since we just need the dialog

        # Ask the user for a file path to open
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV Files", "*.csv")],  # File type options
            title="Open Annotations File"
        )
        
        # Reload the file to reflect new events
        if file_path:  # If a file was selected
            df = pd.read_csv(file_path)  # Load the CSV file as a DataFrame
            
            # Ensure that self.event_times is initialized correctly if it is not already
            if not hasattr(self, 'event_times'):
                self.event_times = {}

            # Iterate through each row of the DataFrame
            for _, row in df.iterrows():
                current_file = row['filename']
                event_time = row['frame'] / 30  # Convert frame to event time (assuming 30 fps)
                event_type = row['event_type']

                # If the current file is not already in self.event_times, initialize it
                if current_file not in self.event_times:
                    self.event_times[current_file] = []

                # Append the event (event_time, event_type) to the corresponding file's list
                self.event_times[current_file].append([event_time, event_type])

                # Also, add/update the selected_event_types dictionary
                self.selected_event_types[event_time] = event_type
            
            # After loading the events, you might want to perform additional operations, like refreshing the UI
            self.load_file()  # Assuming this reloads or updates the file content with the new annotations


    def load_event_types(self):
        # Load previously saved event types from file if it exists
        if os.path.exists(self.event_types_file):
            with open(self.event_types_file, "r") as file:
                self.selected_event_types = json.load(file)

    def save_event(self, event_time, event_type):
        self.selected_event_types[event_time] = event_type
        self.save_event_types()  # Save to the file


    def load_file(self):
        if self.current_file_index < 0 or self.current_file_index >= len(self.file_list):
            messagebox.showinfo("End of Files", "No more files to load.")
            return

        # Load the current file path
        current_file = self.file_list[self.current_file_index]
        file_path = os.path.join(self.annotation_folder, current_file)

        # Update the filename label
        self.filename_label.config(text=f"Current File: {current_file}")

        # Initialize the preprocessor and process the file
        self.preprocessor = DataPreprocessor(file_path)
        self.preprocessor.load_and_preprocess()  # Load and preprocess the data
        self.df = self.preprocessor.df
        self.pass_frames = self.preprocessor.pass_frames
        self.deflection_frames = self.preprocessor.deflection_frames

        # Compute features like velocity, acceleration, etc.
        self.preprocessor.compute_features()

        # Detect events like passes and deflections
        self.preprocessor.detect_events()

        # If we already have event times for this file, load them, otherwise, initialize an empty list
        file_event_times = self.event_times.get(current_file, [])

        # Now that the data is loaded and processed, you can plot
        self.plot_trajectory(file_event_times)

        # Refresh the sidebar to show events for this file only
        self.refresh_sidebar(current_file)

    def plot_trajectory(self, file_event_times):
        # Clear previous plots
        df = self.df
        self.axs[0].cla()  # Clear the ball trajectory plot
        self.axs[1].cla()  # Clear the velocity vs time plot
        self.axs[2].cla()  # Clear the acceleration vs time plot

        # Ball Trajectory
        self.axs[0].plot(df['bb_left'], df['bb_top'], label='Ball Trajectory', linewidth=2)

        # Add vertical lines (dotted) at clicked times for Ball Trajectory
        for event_time in file_event_times:
            event_index = df[df['time_seconds'] >= event_time[0]].index[0]
            self.axs[0].plot([df['bb_left'][event_index], df['bb_left'][event_index]], [0, df['bb_top'].max()], 'r:', label="Event Marker" if event_time[0] == file_event_times[0][0] else "")

        self.axs[0].set_title('Ball Trajectory')
        self.axs[0].set_xlabel('X Position (pixels)')
        self.axs[0].set_ylabel('Y Position (pixels)')
        self.axs[0].invert_yaxis()  # Invert y-axis as per the plot requirement
        self.axs[0].legend()
        self.axs[0].grid(True)

        # Velocity vs Time Plot
        self.axs[1].plot(df['time_seconds'], df['velocity'], label='Velocity', color='blue', linewidth=2)

        # Add vertical lines at clicked times for Velocity vs Time
        for event_time in file_event_times:
            self.axs[1].axvline(x=event_time[0], color='red', linestyle=':', label="Event Marker" if event_time[0] == file_event_times[0][0] else "")

        self.axs[1].set_title('Velocity vs Time')
        self.axs[1].set_xlabel('Time (seconds)')
        self.axs[1].set_ylabel('Velocity (pixels/frame)')
        self.axs[1].legend()
        self.axs[1].grid(True)

        # Acceleration vs Time Plot
        self.axs[2].plot(df['time_seconds'], df['acceleration'], label='Acceleration', color='orange', linewidth=2)

        # Add vertical lines at clicked times for Acceleration vs Time
        for event_time in file_event_times:
            self.axs[2].axvline(x=event_time[0], color='red', linestyle=':', label="Event Marker" if event_time[0] == file_event_times[0][0] else "")

        self.axs[2].set_title('Acceleration vs Time')
        self.axs[2].set_xlabel('Time (seconds)')
        self.axs[2].set_ylabel('Acceleration (pixels/frame²)')
        self.axs[2].legend()
        self.axs[2].grid(True)

        # Create a new canvas to embed the figure in the Tkinter window
        for widget in self.plot_frame.winfo_children():
            widget.destroy()  # Destroy all existing widgets (canvas) in the frame

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)  # Create a new canvas for the plot
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)  # Embed the plot into the Tkinter frame

        # Redraw the canvas with the updated plot
        self.canvas.draw()

        # Force an update to ensure the window refreshes
        self.window.update_idletasks()  # Update idle tasks
        self.window.update()  # Force update the Tkinter window

    def refresh_sidebar(self, current_file):
        # Clear existing events in the sidebar
        for widget in self.sidebar_frame.winfo_children():
            widget.destroy()

        # Reload the events for the current file
        self.event_widgets = []  # Store references to event widgets for later management

        file_event_times = self.event_times.get(current_file, [])
        for event_time in file_event_times:
            event_frame = self.df[self.df['time_seconds'] >= event_time[0]].iloc[0]
            frame_number = event_frame['frame']
            event_row = tk.Frame(self.sidebar_frame)
            event_row.pack(fill=tk.X, pady=5)

            # Frame number label
            frame_label = tk.Label(event_row, text=f"Frame: {frame_number}")
            frame_label.pack(side=tk.LEFT, padx=5)

            # Event type dropdown
            event_type_var = tk.StringVar()
            event_type_var.set(self.selected_event_types.get(event_time[0], event_time[1]))  # Load the event type for this event

            event_type_dropdown = ttk.Combobox(event_row, textvariable=event_type_var, values=self.event_types)
            event_type_dropdown.pack(side=tk.LEFT, padx=5)

            # Add trace to save the event when the dropdown value changes
            event_type_var.trace("w", lambda name, index, mode, var=event_type_var, time=event_time[0]: self.update_label(time, var, event_time, self.event_times, current_file))

            # Remove button
            # Remove button inside refresh_sidebar function
            remove_button = tk.Button(event_row, text="Remove", command=lambda row=event_row, time=event_time[0]: self.remove_event(row, time))

            remove_button.pack(side=tk.RIGHT, padx=5)

            # Store the row to manage it later
            self.event_widgets.append({
                "row": event_row,
                "time": event_time[0],
                "frame_label": frame_label,
                "event_type_dropdown": event_type_dropdown,
                "remove_button": remove_button
            })

    def update_label(self, time, event, event_time, event_times, current_file):
        self.selected_event_types.update({time: event.get()})
        event_type = self.selected_event_types.get(time)

        event_list = self.event_times.get(current_file, [])

        # Iterate through the event list to find the event with the matching timestamp
        for event in event_list:
            if np.array_equal(event[0], time):
                # Update the event type to the new value
                event[1] = event_type
                break  # Exit the loop once the update is done

    def remove_event(self, event_row, event_time):
        # Remove the event from the selected_event_types dictionary
        if event_time in self.selected_event_types:
            del self.selected_event_types[event_time]

        # Remove the event from event_times dictionary (for the current file)
        current_file = self.file_list[self.current_file_index]
        if current_file in self.event_times:
            # Fix the comparison by iterating over event_times and checking time explicitly
            self.event_times[current_file] = [
                event for event in self.event_times[current_file] if event[0] != event_time
            ]

        # Remove the event row from the sidebar UI
        event_row.destroy()  # Remove the event row from the sidebar

        # After removing, refresh the sidebar to reflect the current state of event times
        self.refresh_sidebar(current_file)

        # Optional: You can refresh the plot as well to ensure the event is removed visually
        self.plot_trajectory(self.event_times.get(current_file, []))

    def save_event(self, event_time, event_type):
        # Save the event type for the event time
        self.selected_event_types[event_time] = event_type

    def on_click(self, event):
    # This will get the time of the click on the graph
        if event.inaxes:
            x_pos = event.xdata
            event_type = "Pass" #standard
            event_list = self.event_times.get(self.file_list[self.current_file_index], [])
            event_list.append([x_pos, event_type])
            event_list = list({event[0]: event for event in event_list}.values())
            event_list = sorted(event_list, key=lambda x: x[0])
            self.event_times[self.file_list[self.current_file_index]] = event_list
            self.load_file() 

    def prev_file(self):
        if self.current_file_index > 0:
            self.current_file_index -= 1
            self.load_file()

    def next_file(self):
        if self.current_file_index < len(self.file_list) - 1:
            self.current_file_index += 1
            self.load_file()


# Running the application
if __name__ == "__main__":
    annotation_folder = "/home/lars/thesis-soccertrack-lars/soccertrack/top_view/annotations"
    app = LabelingUI(annotation_folder)
    app.window.mainloop()
