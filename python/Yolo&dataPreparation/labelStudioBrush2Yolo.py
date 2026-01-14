import cv2
import os

folder = 'C:/Users/blagn771/Desktop/labelStudioLabel'
files = os.listdir(folder)
count = 1

for file in files:
    # Check if the file is a picture
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Constructe the full file path
        file_path = os.path.join(folder, file)

        img = cv2.imread(file_path)
        w,h,_ = img.shape

        blur = cv2.GaussianBlur(img, (5,5), 0)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

        trigger = 75
        mask = cv2.inRange(gray, 0, trigger)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        if h == w:
            contoursYolo = contours[1]/w
        else:
            contoursYolo = []
            for elmt in contours:
                contoursYolo.append([elmt[0][0]/h, elmt[0][1]/w])

        contoursText = ['0']
        for elmt in contoursYolo:
            contoursText.append(str(elmt[0][0]))
            contoursText.append(str(elmt[0][1]))

        with open(folder+'/frame-'+str(10000+count)[1:]+'.txt', 'w') as writingFile:
            # Join elements from the list into a single string separeted by a blank space
            contoursString = ' '.join(contoursText)

            # Write the string to the file
            writingFile.write(contoursString)

        count += 1