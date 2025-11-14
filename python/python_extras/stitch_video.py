import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# Read first frames (to determine sizes and fps)
def read_first_frame(path):
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS) if ok else 25.0
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read video: {path}")
    return frame, fps

def FindMatches(BaseImage, SecImage):
    # Using SIFT to find the keypoints and decriptors in the images
    Sift = cv2.SIFT_create()
    BaseImage_kp, BaseImage_des = Sift.detectAndCompute(cv2.cvtColor(BaseImage, cv2.COLOR_BGR2GRAY), None)
    SecImage_kp, SecImage_des = Sift.detectAndCompute(cv2.cvtColor(SecImage, cv2.COLOR_BGR2GRAY), None)

    # Using Brute Force matcher to find matches.
    BF_Matcher = cv2.BFMatcher()
    InitialMatches = BF_Matcher.knnMatch(BaseImage_des, SecImage_des, k=2)

    # Applying ratio test and filtering out the good matches.
    GoodMatches = []
    base_pts = []
    sec_pts = []

    for m, n in InitialMatches:
        if m.distance < 0.65 * n.distance:
            GoodMatches.append([m])
            base_pts.append(BaseImage_kp[m.queryIdx].pt)
            sec_pts.append(SecImage_kp[m.trainIdx].pt)

    # Make copies so we don’t draw directly on the original images
    keypointBaseImage = BaseImage.copy()
    keypointSecImage = SecImage.copy()

    # Draw circles on the base image
    for (x, y) in base_pts:
        cv2.circle(keypointBaseImage, (int(x), int(y)), radius=3, color=(0, 0, 255), thickness=2)

    # Draw circles on the secondary image
    for (x, y) in sec_pts:
        cv2.circle(keypointSecImage, (int(x), int(y)), radius=3, color=(0, 255, 0), thickness=2)

    # Plot with matplotlib (convert BGR -> RGB for correct colors)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Base image keypoints")
    plt.imshow(cv2.cvtColor(keypointBaseImage, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Secondary image keypoints")
    plt.imshow(cv2.cvtColor(keypointSecImage, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.show()

    return GoodMatches, BaseImage_kp, SecImage_kp



def FindHomography(Matches, BaseImage_kp, SecImage_kp):
    # If less than 4 matches found, exit the code.
    if len(Matches) < 4:
        print("\nNot enough matches found between the images.\n")
        exit(0)

    # Storing coordinates of points corresponding to the matches found in both the images
    BaseImage_pts = []
    SecImage_pts = []
    for Match in Matches:
        BaseImage_pts.append(BaseImage_kp[Match[0].queryIdx].pt)
        SecImage_pts.append(SecImage_kp[Match[0].trainIdx].pt)

    # Changing the datatype to "float32" for finding homography
    BaseImage_pts = np.float32(BaseImage_pts)
    SecImage_pts = np.float32(SecImage_pts)

    # Finding the homography matrix(transformation matrix).
    (HomographyMatrix, Status) = cv2.findHomography(SecImage_pts, BaseImage_pts, cv2.RANSAC, 4.0)

    return HomographyMatrix, Status

    
def GetNewFrameSizeAndMatrix(HomographyMatrix, Sec_ImageShape, Base_ImageShape):
    # Reading the size of the image
    (Height, Width) = Sec_ImageShape
    
    # Taking the matrix of initial coordinates of the corners of the secondary image
    # Stored in the following format: [[x1, x2, x3, x4], [y1, y2, y3, y4], [1, 1, 1, 1]]
    # Where (xi, yi) is the coordinate of the i th corner of the image. 
    InitialMatrix = np.array([[0, Width - 1, Width - 1, 0],
                              [0, 0, Height - 1, Height - 1],
                              [1, 1, 1, 1]])
    
    # Finding the final coordinates of the corners of the image after transformation.
    # NOTE: Here, the coordinates of the corners of the frame may go out of the 
    # frame(negative values). We will correct this afterwards by updating the 
    # homography matrix accordingly.
    FinalMatrix = np.dot(HomographyMatrix, InitialMatrix)

    [x, y, c] = FinalMatrix
    x = np.divide(x, c)
    y = np.divide(y, c)

    # Finding the dimentions of the stitched image frame and the "Correction" factor
    min_x, max_x = int(round(min(x))), int(round(max(x)))
    min_y, max_y = int(round(min(y))), int(round(max(y)))

    New_Width = max_x
    New_Height = max_y
    Correction = [0, 0]
    if min_x < 0:
        New_Width -= min_x
        Correction[0] = abs(min_x)
    if min_y < 0:
        New_Height -= min_y
        Correction[1] = abs(min_y)
    
    # Again correcting New_Width and New_Height
    # Helpful when secondary image is overlaped on the left hand side of the Base image.
    if New_Width < Base_ImageShape[1] + Correction[0]:
        New_Width = Base_ImageShape[1] + Correction[0]
    if New_Height < Base_ImageShape[0] + Correction[1]:
        New_Height = Base_ImageShape[0] + Correction[1]

    # Finding the coordinates of the corners of the image if they all were within the frame.
    x = np.add(x, Correction[0])
    y = np.add(y, Correction[1])
    OldInitialPoints = np.float32([[0, 0],
                                   [Width - 1, 0],
                                   [Width - 1, Height - 1],
                                   [0, Height - 1]])
    NewFinalPonts = np.float32(np.array([x, y]).transpose())

    # Updating the homography matrix. Done so that now the secondary image completely 
    # lies inside the frame
    HomographyMatrix = cv2.getPerspectiveTransform(OldInitialPoints, NewFinalPonts)
    
    return [New_Height, New_Width], Correction, HomographyMatrix



def StitchImages(TopVideo, BotVideo, output_path, auto):
    BaseImage, fps_top = read_first_frame(TopVideo)
    SecImage, fps_bot = read_first_frame(BotVideo)
    fps = fps_top if fps_top > 0 else (fps_bot if fps_bot > 0 else 25.0)

    # Your fixed correspondence points:
    # (TOP image points first)
    top_pts = np.float32([
        (304, 942), (1704, 938), (308, 1142), (1704, 1139)
    ])

    # (BOTTOM image points second)
    bottom_pts = np.float32([
        (308, 130), (1692, 117), (310, 325), (1694, 314)
    ])
    
    # Finding homography matrix.
    if auto:
        # Finding matches between the 2 images and their keypoints AUTOMATICALLY
        Matches, BaseImage_kp, SecImage_kp = FindMatches(BaseImage, SecImage)
        HomographyMatrix, Status = FindHomography(Matches, BaseImage_kp, SecImage_kp)
    else:
        HomographyMatrix, Status = cv2.findHomography(bottom_pts, top_pts, cv2.RANSAC, 4.0)
    
    # Finding size of new frame of stitched images and updating the homography matrix 
    NewFrameSize, Correction, HomographyMatrix = GetNewFrameSizeAndMatrix(HomographyMatrix, SecImage.shape[:2], BaseImage.shape[:2])

    # Open videos for frame-by-frame stitching
    capA = cv2.VideoCapture(TopVideo)
    capB = cv2.VideoCapture(BotVideo)

    # Output writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (NewFrameSize[1], NewFrameSize[0]))

    print("Stitching, please wait...")

    while True:
        okA, frameA = capA.read()
        okB, frameB = capB.read()
        if not (okA and okB):
            break

        # Finally placing the images upon one another.
        StitchedImage = cv2.warpPerspective(frameB, HomographyMatrix, (NewFrameSize[1], NewFrameSize[0]))
        StitchedImage[Correction[1]:Correction[1]+frameA.shape[0], Correction[0]:Correction[0]+frameA.shape[1]] = frameA

        out.write(StitchedImage)

    capA.release()
    capB.release()
    out.release()

    print(f"\n✅ Done! Output saved to:\n{output_path}")


if __name__ == "__main__":
    auto = True # Auto detection seems to give better results...
    full_dataset = True

    if not full_dataset:
        # Reading the 2 videos.
        top_video_path = "C:/Users/blagn771/Desktop/000000-top - Trim.mp4"          # Top view video
        bottom_video_path = "C:/Users/blagn771/Desktop/000000-bot - Trim.mp4"   # Bottom view video
        output_path = "C:/Users/blagn771/Desktop//stitched_output.mp4"           # Output file

        # Calling function for stitching images.
        StitchedImage = StitchImages(top_video_path, bottom_video_path, output_path, auto)

    else:
        main_folder = "E:/XP-CAMBRIDGE"
        video1_folder = "VIDEO_1"
        video2_folder = "VIDEO_2"
        exp_list = os.listdir(os.path.join(main_folder, video1_folder))
        exp_list = [s.split('.')[0] for s in exp_list]
        
        for exp in exp_list:
            print(exp)
            top_video_path = os.path.join(main_folder, video1_folder, exp) + ".224902806/000000.mp4"
            bottom_video_path = os.path.join(main_folder, video2_folder, exp) + ".224902807/000000.mp4"
            output_path = os.path.join(main_folder, "stitched_mp4", exp) + ".mp4"

            # Calling function for stitching images.
            StitchedImage = StitchImages(top_video_path, bottom_video_path, output_path, auto)

    