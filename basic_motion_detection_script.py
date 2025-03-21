import cv2
import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import percentileofscore
from copy import deepcopy

# peaks = [160, 222, 270, 351, 403]

def interactive_frames_and_difference_plot(frames, subtracted_images, norms, post_process_data, digital_data, output_file, save_video= True):
    """
    Displays an interactive window that shows:
      - The *original* frame (left)
      - The *difference* image (right)
      - A line plot of norms (bottom)
      - A slider to scroll through frames manually
      - Play and Pause buttons to animate from the current slider position to the end (no looping).
    
    Parameters
    ----------
    frames : list of original frames (BGR) from get_frames_from_video
    subtracted_images : list of grayscale difference images from process_frames
    norms : list or array of norm values for each difference image
    """

    if frames == None or subtracted_images == None or norms == None:
        print("No frames, subtracted images, or norms found.")
        return

    # Convert original frames (for the relevant indices) to RGB
    # Because subtracted_images go from index=1..(len(frames)-1),
    # we'll map each difference image "i" to frames[i+1] or frames[i].
    # For consistency with how process_frames is enumerated, let's use frames[1:-1].
    original_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames[10:-1]]

    # Convert the grayscale difference images to RGB for Matplotlib display
    difference_rgb = [img for img in subtracted_images]
    
    # Convert norms to a numpy array (if not already)
    norm_array = np.array(norms)
    frame_indices = np.arange(len(norm_array))
    
    # Post-processed data 
    post_process_data_arr = np.array(post_process_data)
    
    #Digital Data
    digital_data_arr = np.array(digital_data)

    # --- Create the figure and axes layout ---
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(7, 2, height_ratios=[8,8,1,8,1,8,1])

    ax_orig = fig.add_subplot(gs[0, 0])      # top-left
    ax_diff = fig.add_subplot(gs[0, 1])      # top-right
    ax_plot_orignal = fig.add_subplot(gs[1, :])      # entire bottom row
    ax_plot_post_processsed = fig.add_subplot(gs[3, :])
    ax_digital_signal = fig.add_subplot(gs[5,:])
    

    # Display the first original frame and the first difference image
    current_orig_im = ax_orig.imshow(original_rgb[0])
    ax_orig.set_title("Original Frame")
    ax_orig.axis("off")

    current_diff_im = ax_diff.imshow(difference_rgb[0])
    ax_diff.set_title("Difference Image (Frame n vs. Frame n+1)")
    ax_diff.axis("off")

    # Plot the norms on the bottom subplot
    ax_plot_orignal.plot(frame_indices, norm_array, label="Frame Difference Norm", color='b')
    ax_plot_orignal.set_title("Frame-to-Frame Grayscale Difference")
    ax_plot_orignal.set_xlabel("Subtracted Image Index")
    ax_plot_orignal.set_ylabel("Norm of Difference")
    ax_plot_orignal.grid(True)
    ax_plot_orignal.legend()
    
    ax_plot_post_processsed.plot(frame_indices, post_process_data_arr, label="Post Processed Norm", color='r')
    ax_plot_post_processsed.set_title("Filters Applied to Norm data")
    ax_plot_post_processsed.set_xlabel("Subtracted Image Index")
    ax_plot_post_processsed.set_ylabel("Filtered Norm of Difference")
    ax_plot_post_processsed.grid(True)
    ax_plot_post_processsed.legend()
    
    ax_digital_signal.plot(frame_indices, digital_data_arr, label="Digital Signal", color='g')
    ax_digital_signal.set_title("Analog to Digital Converted Signal")
    ax_digital_signal.set_xlabel("Image Index")
    ax_digital_signal.set_ylabel("High or Low Signal")
    ax_digital_signal.grid(True)
    ax_digital_signal.legend()

    # A vertical line that we'll move to indicate the current index
    marker_line = ax_plot_orignal.axvline(x=0, color='r', linestyle='--', linewidth=2)
    marker_line_1 = ax_plot_post_processsed.axvline(x=0, color='r', linestyle='--', linewidth=2)
    marker_line_2 = ax_digital_signal.axvline(x=0, color='r', linestyle='--', linewidth=2)

    # --- Slider to navigate frames manually ---
    # Place the slider below the bottom row
    slider_ax = plt.axes([0.15, 0.07, 0.7, 0.03])
    slider = Slider(
        ax=slider_ax,
        label='Index',
        valmin=0,
        valmax=len(subtracted_images) - 1,
        valinit=0,
        valstep=1,
        color='lightblue'
    )
    
    # --- Saving the Video ---
    if save_video:
        frame_size = (fig.canvas.get_width_height()[0], fig.canvas.get_width_height()[1])
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'XVID' for .avi
        out = cv2.VideoWriter(output_file, fourcc, 20.0, frame_size)
    
    # --- Closing the plot usng ESC key --- 
    def on_keypress(event):
        """Closes the Matplotlib window when 'Esc' key is pressed."""
        if event.key == 'escape':
            plt.close(fig)

    # Connect the keypress event to the figure
    fig.canvas.mpl_connect('key_press_event', on_keypress)
    
    # --- Buttons for Play and Pause ---
    play_pause_ax = plt.axes([0.45, 0.01, 0.15, 0.05])
    play_pause_button = Button(play_pause_ax, 'Play', color='lightgreen', hovercolor='green')

    # Use lists so nested functions can modify these variables
    is_playing = [False]
    current_index = [0]

    def update_slider(idx):
        """Sets the slider to a new index and triggers the display update."""
        slider.set_val(idx)  # will call slider_update
        fig.canvas.draw_idle()

    def slider_update(val):
        """Callback for manual slider changes or forced updates via set_val."""
        idx = int(slider.val)
        # Update both images
        current_orig_im.set_data(original_rgb[idx])
        current_diff_im.set_data(difference_rgb[idx])
        # Update the vertical line in the plot
        marker_line.set_xdata([idx, idx])
        marker_line_1.set_xdata([idx, idx])
        marker_line_2.set_xdata([idx, idx])
        # Store current index
        current_index[0] = idx

    slider.on_changed(slider_update)

    def play_pause_event(event):
        """Toggles between playing and pausing."""
        if is_playing[0]:  # If currently playing, pause it
            is_playing[0] = False
            play_pause_button.label.set_text('Play')
        else:  # If paused, start playing
            is_playing[0] = True
            play_pause_button.label.set_text('Pause')
            start_idx = current_index[0]
            for idx in range(start_idx, len(subtracted_images)):
                if not is_playing[0]:  # Stop if user pauses
                    break
                update_slider(idx)
                plt.pause(0.03)  # Adjust speed as needed
                
                # --- Capture and Save Frame (if recording is enabled) ---
                if save_video:
                    fig.canvas.draw()
                    frame = np.array(fig.canvas.buffer_rgba())
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                    out.write(frame)

            is_playing[0] = False  # Reset state when done
            play_pause_button.label.set_text('Play')

    play_pause_button.on_clicked(play_pause_event)

    # Initialize at index=0
    slider_update(0)
    plt.show()
    
    # Release video writer if enabled
    if save_video:
        out.release()
        print(f"Video saved as {output_file}")
  
#------------------------------------------------------------------------------------------------------------------------# 


def median_of_sliding_window():
    
    #
    
    pass 


def filtered_norm(norm, window_size, kernel = [1,1,1,1,1]):
    
    clipped_norm = np.clip(norm, 0, 120)    
    convlouted_array =  np.convolve(clipped_norm.flatten(), kernel, 'same')
    
    #--- Setting Upper bound of the Data: Clipping data above 30k value of Norm to 120 ---#
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.clip = True
    scaled_norm = scaler.fit_transform(convlouted_array.reshape(-1,1)).reshape(-1,1)
    
    #--- Converting Analog Signal to Digital Signal ---#
    threshold = 0.25
    digital_signal = np.where(scaled_norm>threshold, 1, 0).reshape(-1,1)
    
    #--- Extracting Peaks for Clipping the video ---#
    frame_indices = set()
    digital_signal[0] = 0
    digital_signal[-1] = 0
    rising_edges = np.where((digital_signal[:-1] == 0) & (digital_signal[1:] == 1))[0] + 1
    falling_edges = np.where((digital_signal[:-1] == 1) & (digital_signal[1:] == 0))[0]
        
        
        
    return scaled_norm, digital_signal, frame_indices

def extracting_frames_around_peak(window_size, frames, peaks, frames_in_window = None): 
    if frames_in_window == None:
        frames_in_window = set()
    size = frames[0].shape
    print(f"Total Length of videos: {len(frames)}")
    
    video = cv2.VideoWriter("processed_videos/output.avi", cv2.VideoWriter_fourcc(*'MJPG'), 10, (size[1],size[0]))
    for i in peaks:
        pass
        
    
    print(f"Total length of processed video {len(frames_in_window)}")
    print(f"Diff = {len(frames) - len(frames_in_window)}")
    video.release()
    cv2.destroyAllWindows()
    
    return frames_in_window

def crop_frames(frames, cropping, cropped_frames = []):

    dimension = frames[0].shape
    for i in range(1,len(frames)-1):
        # cropped_frame = frames[i][0:135, 0:dimension[1]] # for Data_MD videos
        # cropped_frame = frames[i][0:150, 0:dimension[1]] # for Queens_bev videos
        cropped_frame = frames[i][36:dimension[0], 0:dimension[1]] # for Test_vid1 videos

        cropped_frames.append(cropped_frame)
    if cropping == True:
        return cropped_frames
    return frames

def get_frames_from_video(video_path, frames = None):
    
    if frames is None:
        frames = []
    video = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while frame_count<=3000:
        ret, frame = video.read()
        if not ret:
            break        
        frames.append(frame)
        frame_count+=1
        """path = f"frames/frames{frame_count}.jpg"
        cv2.imwrite(path, frame)"""
        
    video.release()
    cv2.destroyAllWindows()
    return frames

def process_frames(frames, norm = [], subtracted_images= []):
    
    for idx in range(10, len(frames)-10):
        
        current_frame  = frames[idx]
        next_frame = frames[idx+1]
        gs_current_frame = cv2.cvtColor(current_frame,cv2.COLOR_BGR2GRAY)
        # gs_current_frame = current_frame[:,:,2]
        gs_current_frame_normalized = gs_current_frame/255.0
        gs_next_frame = cv2.cvtColor(next_frame,cv2.COLOR_BGR2GRAY)
        # gs_next_frame = next_frame[:,:,2]
        gs_next_frame_normalized = gs_next_frame/255.0
        
        kernel = np.ones((5,5),np.float32)/25
        gaussian_blur_curr = cv2.filter2D(gs_current_frame_normalized,-1,kernel)
        gaussian_blur_next = cv2.filter2D(gs_next_frame_normalized, -1, kernel)
        subtracted_image = cv2.absdiff(gaussian_blur_curr, gaussian_blur_next)
        subtracted_image = cv2.resize(subtracted_image,(640,480))
        subtracted_images.append(subtracted_image*5)
        norm.append(np.linalg.norm(subtracted_image))
        
    return subtracted_images, norm

    
def main(video_name):
    window_size = 15
    video_path = f"data/{video_name}"
    original_frames = get_frames_from_video(video_path)
    cropped_frames = crop_frames(original_frames, cropping=True)
    subtracted_images, norms = process_frames(cropped_frames)
    norm_arr = np.array(norms, dtype = np.int32).reshape(-1,1)
    scaled_norm, digital_signal, peaks =  filtered_norm(norm_arr, window_size, kernel = np.array([1,2,3,4,5,4,3,2,1]).reshape(-1,1).flatten()) 
    interactive_frames_and_difference_plot(cropped_frames, subtracted_images, norms, scaled_norm, digital_signal, output_file=f"processed_videos/{video_name}", save_video= False)
    # extracting_frames_around_peak(window_size, original_frames, peaks)
   

if __name__ == '__main__':
    video_name = "test_vid1.mp4"
    main(video_name)
    
