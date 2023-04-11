# import os
# import cv2 as cv

# def mkvid(target_dir = '../frames', endswith = "jpg"):
#     videodims = (640,480)
#     fps = 24  # frames per second

#     # Create a list of the frames filenames.
#     images = [img for img in os.listdir(target_dir) if img.endswith(".jpg")] 
#     images.sort(key=lambda x: (int(x.split(None, 1)[0]) if x[-4:].isdigit() else 999, x)) #  Sort the names in the list for joining the frames consecutively.  
#     frame = cv.imread(os.path.join(target_dir, images[0])) # Initialize first frame. 
#     height, width, layers = frame.shape 
#     video = cv.VideoWriter('demo.avi',cv.VideoWriter_fourcc(*'XVID'), fps, (width,height))


#     for image in images:
#         video.write(cv.imread(os.path.join(target_dir, image)))


#     cv.destroyAllWindows()
#     video.release()

import os
import cv2 as cv
import argparse

def mkvid(args):
    
    # Set output variables
    videodims = (640, 480)
    fps = args.fps
    target_dir = args.target_dir

    # Create a list of the frames filenames
    images = [img for img in os.listdir(target_dir) if img.endswith(".jpg")]
    images.sort(key=lambda x: (int(x.split(None, 1)[0]) if x[-4:].isdigit() else 999, x))

    # Initialize first frame
    frame = cv.imread(os.path.join(target_dir, images[0]))
    height, width, layers = frame.shape
    video = cv.VideoWriter(args.output_file, cv.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    for image in images:
        video.write(cv.imread(os.path.join(target_dir, image)))

    cv.destroyAllWindows()
    video.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a video from a sequence of frames')
    parser.add_argument('--fps', type=int, default=24, help='Frames per second for the output video')
    parser.add_argument('--target_dir', type=str, default='frames', help='Directory containing the frames')
    parser.add_argument('--output_file', type=str, default='dome_with_tesseract.avi', help='Output video file name')
    args = parser.parse_args()

    mkvid(args)

