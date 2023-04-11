import cv2 as cv
import numpy as np
from craft_model.craft.craft import CRAFT
import tensorflow as tf
from craft_model.craft.test import copyStateDict, test_net
import torch
import torch.backends.cudnn as cudnn
import miscutils.file_utils as file_utils
import pytesseract
import argparse
import os

"""
This file can be called in the terminal using the following format: 

    python main.py --vid_path target_data/rgb_vid.mp4 --trained_model_path craft_model/craft/weights/craft_mlt_25k.pth --result_folder ./result/ --frame_filename rgbvid_frame --mask_folder mask --init_frame_number 1200 --final_frame_number 2500 --text_threshold 0.75 --link_threshold 0.4 --low_text 0.4 --poly --cuda

* Not all arguments are needed: 

'init_frame_number', 
'final_frame_number', 
'text_threshold', 
'link_threshold', 
'low_text', 
'cuda', 
'poly', 
'visualize' 

are all optional.

A reduced version of the command is: 
    python main.py --vid_path target_data/video.mp4 --trained_model_path craft_model/craft/weights/craft_mlt_25k.pth --result_folder ./result/ --frame_filename rgbvid_frame --mask_folder mask 

"""

def main(args):
    """
    Create path variables and parameters. 
    """

    vid_path = args.vid_path
    trained_model_path = args.trained_model_path
    result_folder = args.result_folder
    frame_filename = args.frame_filename
    mask_folder = args.mask_folder
    

    # Initialize NN.
    net = CRAFT()
    # Check if CUDA is available. If yes, compute with CUDA.  
    if torch.cuda.is_available():
        net.load_state_dict(copyStateDict(torch.load(trained_model_path)))  # Load pre-trained model
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False
    else:
        net.load_state_dict(copyStateDict(torch.load(trained_model_path, map_location='cpu'))) # Load pre-trained model with CPU

    # Create working variables. 
    vid = cv.VideoCapture(vid_path)
    vid.set(1, args.init_frame_number)                # The resulting video begins at frame 1200 and ends at frame 2500. It was set like this for demo purposes. 
    ret, frame = vid.read()
    refine_net = None

    # Create folder containing images to be joined into a video with mkvid.py script. 
    init_frame_number = args.init_frame_number
    i = args.init_frame_number
    final_frame_number = args.final_frame_number

    # Recognising text from bounding boxes using pytesseract. 
    def recognize_text_from_bboxes(image, bboxes):
        recognized_texts = []
        for bbox in bboxes:
            coordinates = [int(coordinate) for coordinate in bbox.split(",")]        # Parsing the boundary box numbers from .txt files.
            x_coordinates = coordinates[0::2]
            y_coordinates = coordinates[1::2]

            # Exctracting the boundaries of the boxes. 
            x_min, x_max = min(x_coordinates), max(x_coordinates)
            y_min, y_max = min(y_coordinates), max(y_coordinates)

            # Apply pytesseract to extract text from the region of interest defined by the boundary boxes. 
            text_roi = image[y_min:y_max, x_min:x_max]
            recognized_text = pytesseract.image_to_string(text_roi).replace('/n', '').strip()
            recognized_texts.append(recognized_text)
        return recognized_texts
    
    while i <= final_frame_number:
        vid.set(1, i)
        _, frame = vid.read()
        net.eval()
        _, polys, score_text = test_net(net, frame, params['text_threshold'], params['link_threshold'], params['low_text'], params['poly'], refine_net)
        mask_file = result_folder + mask_folder + "/frame" + frame_filename + f"{i-init_frame_number}" + "_mask.jpeg"
        cv.imwrite(mask_file, score_text)
        bboxes_txt_file = result_folder + f"frame{i}.txt"

        if os.path.exists(bboxes_txt_file):

            with open(bboxes_txt_file, "r") as file:
                bboxes_lines = file.readlines()

            recognized_texts = recognize_text_from_bboxes(frame, bboxes_lines)

            # Set parameters for text-box
            font = cv.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            padding = 4
            vertical_spacing = 3
            y_offset = int(frame.shape[0] * 0.95)

            # Create the dark-green box at the bottom
            # box_height = int(frame.shape[0] * 0.05)
            
            cv.rectangle(frame, (0, y_offset), (frame.shape[1], frame.shape[0]), (36, 255, 12), 1)
            y_pos = y_offset + padding
            
            # for idx, poly in enumerate(polys):
            for idx, text in enumerate(recognized_texts):
                if args.visualize:
                    text_size = cv.getTextSize(text, font, font_scale, font_thickness)[0]

                    # Calculate the x-coordinate to center the text
                    x_center = (frame.shape[1] - text_size[0]) // 2

                    # Add text to the text-box
                    text_coords = (x_center, y_pos)
                    cv.putText(frame, text, text_coords, font, font_scale, (255, 255, 255), font_thickness, cv.LINE_AA)

                    # Update y_pos for the next text
                    y_pos += text_size[1] + vertical_spacing

                    
                    # This doesn't work as intended, needs rework in future iterations.
                    # Check if y_pos exceeds the frame boundaries, if so break the loop 
                    if y_pos > frame.shape[0] - padding:
                        break

        file_utils.saveResult(f"{i}" + ".jpg", frame[:,:,::-1], polys, dirname=result_folder)
        print (f"File number {i-init_frame_number} with index {i} is done. {final_frame_number-i} left!")
        i += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video and apply CRAFT plus pytesseract.")
    parser.add_argument("--vid_path", type=str, default="target_data/rgb_vid.mp4", help="Path to the video file")
    parser.add_argument("--trained_model_path", type=str, default="craft_model/craft/weights/craft_mlt_25k.pth", help="Path to the trained CRAFT model")
    parser.add_argument("--result_folder", type=str, default="./frames/", help="Folder to save the resulting frames")
    parser.add_argument("--frame_filename", type=str, default="rgbvid_frame", help="Filename prefix for frames")
    parser.add_argument("--mask_folder", type=str, default="mask", help="Folder to save mask files")
    parser.add_argument("--init_frame_number", type=int, default=1301, help="Initial frame number to start processing")
    parser.add_argument("--final_frame_number", type=int, default=2500, help="Final frame number to stop processing")
    parser.add_argument("--text_threshold", type=float, default=0.75, help="Text confidence threshold")
    parser.add_argument("--link_threshold", type=float, default=0.4, help="Link confidence threshold")
    parser.add_argument("--low_text", type=float, default=0.4, help="Low text confidence threshold")
    parser.add_argument("--visualize", type=bool, default=True, help="Add text-box")
    parser.add_argument("--poly", action="store_true", help="Enable polygon output")
    parser.add_argument("--cuda", action="store_true", help="Enable CUDA")

    args = parser.parse_args()

    # Create params dictionary from argparse results
    params = {
        'init_frame_number': args.init_frame_number,
        'final_frame_number': args.final_frame_number,
        'text_threshold': args.text_threshold,
        'link_threshold': args.link_threshold,
        'low_text': args.low_text,
        'cuda': args.cuda,
        'poly': args.poly,
        'visualize': args.visualize

    }
    main(args)






