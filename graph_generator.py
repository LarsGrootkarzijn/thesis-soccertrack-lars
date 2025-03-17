import matplotlib.pyplot as plt

class GraphGenerator:
    def __init__(self, df, pass_frames, deflection_frames, angle_changes_df, output_prefix):
        self.df = df
        self.pass_frames = pass_frames
        self.deflection_frames = deflection_frames
        self.angle_changes_df = angle_changes_df
        self.output_prefix = output_prefix

    def plot_velocity_acceleration(self):
        df = self.df
        plt.figure(figsize=(14, 8))

        plt.subplot(3, 1, 1)
        plt.plot(df['time_seconds'], df['velocity'], label="Velocity (pixels/sec)")
        plt.title('Velocity Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity')
        plt.grid(True)
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(df['time_seconds'], df['acceleration'], label="Acceleration (pixels/sec^2)", color='red')
        plt.title('Acceleration Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration')
        plt.grid(True)
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(df['time_seconds'], df['velocity'], label="Velocity (pixels/sec)")
        plt.scatter(df.loc[df['frame'].isin(self.pass_frames), 'time_seconds'],
                    df.loc[df['frame'].isin(self.pass_frames), 'velocity'],
                    color='green', label="Passes")
        plt.scatter(df.loc[df['frame'].isin(self.deflection_frames), 'time_seconds'],
                    df.loc[df['frame'].isin(self.deflection_frames), 'velocity'],
                    color='red', label="Deflections")
        plt.title('Detected Passes & Deflections')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{self.output_prefix}_velocity_acceleration.png")
        plt.close()

    def plot_trajectory(self):
        df = self.df
        plt.figure(figsize=(10, 6))
        plt.gca().invert_yaxis()

        plt.plot(df['bb_left'], df['bb_top'], label='Ball Trajectory', linewidth=2)

        plt.scatter(df.loc[df['frame'].isin(self.pass_frames), 'bb_left'],
                    df.loc[df['frame'].isin(self.pass_frames), 'bb_top'],
                    color='green', label='Pass Events', zorder=5, s=100)

        plt.scatter(df.loc[df['frame'].isin(self.deflection_frames), 'bb_left'],
                    df.loc[df['frame'].isin(self.deflection_frames), 'bb_top'],
                    color='red', label='Deflection Events', zorder=5, s=100)

        for idx, row in self.angle_changes_df.iterrows():
            plt.annotate(f"{row['angle_change']:.1f}°",
                         (row['bb_left'], row['bb_top']),
                         textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color='purple')

        plt.scatter(df['bb_left'].iloc[0], df['bb_top'].iloc[0], color='yellow', s=150, marker='*', label='Start')
        plt.scatter(df['bb_left'].iloc[-1], df['bb_top'].iloc[-1], color='black', s=150, marker='X', label='End')

        plt.title('Ball Trajectory with Pass and Deflection Events')
        plt.xlabel('X Position (pixels)')
        plt.ylabel('Y Position (pixels)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"{self.output_prefix}_trajectory.png")
        plt.close()
