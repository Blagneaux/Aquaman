import cv2
import numpy as np

def find_crop_start_by_subtraction(full_video_path, cropped_video_path, start_time_seconds=0):
    # Load the full video and the cropped video
    full_video = cv2.VideoCapture(full_video_path)
    cropped_video = cv2.VideoCapture(cropped_video_path)
    
    if not full_video.isOpened():
        raise FileNotFoundError(f"Unable to open full video: {full_video_path}")
    if not cropped_video.isOpened():
        raise FileNotFoundError(f"Unable to open cropped video: {cropped_video_path}")
    
    # Get the total number of frames in the cropped video
    total_cropped_frames = int(cropped_video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get the frame rate of the full video to calculate the start frame
    fps = full_video.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time_seconds * fps)
    
    # Set the full video to start from the given timestamp
    full_video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Read the first frame of the cropped video
    ret, crop_frame = cropped_video.read()
    if not ret:
        raise ValueError("Cropped video is empty or cannot be read.")
    
    # Convert the cropped frame to grayscale
    crop_frame_gray = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)
    
    frame_number = start_frame
    found_frame_number = None
    
    # Loop through each frame of the full video
    while True:
        ret, full_frame = full_video.read()
        if not ret:
            break
        
        # Convert the full frame to grayscale
        full_frame_gray = cv2.cvtColor(full_frame, cv2.COLOR_BGR2GRAY)
        
        # Subtract the cropped frame from the full frame
        if full_frame_gray.shape[0] >= crop_frame_gray.shape[0] and full_frame_gray.shape[1] >= crop_frame_gray.shape[1]:
            # Extract the region of interest (ROI) from the full frame
            roi = full_frame_gray[:crop_frame_gray.shape[0], :crop_frame_gray.shape[1]]
            
            # Compute the absolute difference
            diff = cv2.absdiff(roi, crop_frame_gray)
            
            # Check if the difference is close to zero
            if np.sum(diff) < 1e-6:  # Adjust tolerance as needed
                found_frame_number = frame_number
                break
        
        frame_number += 1
    
    # Release video resources
    full_video.release()
    cropped_video.release()
    
    if found_frame_number is not None:
        end_frame_number = found_frame_number + total_cropped_frames - 1
        return found_frame_number, end_frame_number
    else:
        raise ValueError("No matching frame found where the cropped video begins.")

# Example usage
full_video_path = "E:/crop_nadia/1/full_video.mp4"
cropped_video_path = "E:/crop_nadia/1/11/11.mp4"

try:
    # Provide a rough timestamp in seconds
    rough_start_time = 540  # Start looking from 30 seconds into the full video
    start_frame, end_frame = find_crop_start_by_subtraction(full_video_path, cropped_video_path, start_time_seconds=rough_start_time)
    print(f"The cropped video starts at frame {start_frame} and ends at frame {end_frame} of the full video.")
except Exception as e:
    print(f"Error: {e}")