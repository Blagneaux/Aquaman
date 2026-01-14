import cv2

def crop_video(input_path, output_path, start_frames, end_frames):
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Validate frame cropping parameters
    if start_frames + end_frames >= total_frames:
        print("Error: Too many frames to remove. The video would become empty.")
        cap.release()
        return

    # Define the codec and create the VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Start reading and writing frames
    current_frame = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Write frames that are within the cropped range
        if start_frames <= current_frame < (total_frames - end_frames):
            out.write(frame)

        current_frame += 1

    # Release resources
    cap.release()
    out.release()
    print(f"Video cropped successfully! Saved to {output_path}")

# Example usage
input_video = "E:/crop_nadia/37/8/8.mp4"  # Replace with your input video path
output_video = "E:/crop_nadia/37/8/8_crop.mp4"  # Replace with your desired output video path
frames_to_remove_start = 0  # Number of frames to remove from the start
frames_to_remove_end = 100  # Number of frames to remove from the end

crop_video(input_video, output_video, frames_to_remove_start, frames_to_remove_end)
