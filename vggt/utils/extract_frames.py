import os
import cv2

def extract_frames(video_path, output_folder, frame_interval=10):
    """
    Extract frames from a video at specified intervals.
    
    Args:
        video_path (str): Path to the input video file
        output_folder (str): Directory to save extracted frames
        frame_interval (int): Extract every nth frame
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not video.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    frame_count = 0
    saved_count = 0
    
    print(f"Extracting frames from {video_path}...")
    
    # Read until video is completed
    while True:
        # Read the next frame
        ret, frame = video.read()
        
        # Break the loop if we have reached the end of the video
        if not ret:
            break
        
        # Save frame if it matches our interval
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"{saved_count:02d}.png")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
            
        frame_count += 1
    
    # Release the video capture object
    video.release()
    
    print(f"Extraction complete. Saved {saved_count} frames to {output_folder}")

if __name__ == "__main__":
    video_path = "DJI_Short.mp4"
    output_folder = "examples/dji"
    
    extract_frames(video_path, output_folder, frame_interval=10)
