from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from numpy.polynomial import polynomial as pl

model = YOLO("yolov8n-seg-custom.pt")
cap = cv2.VideoCapture("testYolomov.mp4")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
   
size = (frame_width, frame_height)

# Init list of all the coordinates
XY = []
XY_interpolated = []

# loop through the video frames
while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        # run inference on a frame
        results = model(frame)

        # view results
        for r in results:
            if r.masks == None:
                break
            mask = r.masks.xy
            xys = mask[0]
            XY.append(np.int32(xys))
            cv2.polylines(frame, np.int32([xys]), True, (0, 255, 255), 2)

        cv2.imshow("img", frame)

        #break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    else:
        break

cap.release()
cv2.destroyAllWindows()

XY = XY[2:]

# Set the number of points the segmentation needs to be
desired_points_count = 100
screenX, screenY = 1400, 1400
resX, resY = 2**7, 2**7

# Solution derived from https://meshlogic.github.io/posts/jupyter/curve-fitting/parametric-curve-fitting/

def uniform_param(P):
    u = np.linspace(0, 1, len(P))
    return u
    
def chordlength_param(P):
    u = generate_param(P, alpha=1.0)
    return u
    
def centripetal_param(P):
    u = generate_param(P, alpha=0.5)
    return u
    
def generate_param(P, alpha):
    n = len(P)
    u = np.zeros(n)
    u_sum = 0
    for i in range(1,n):
        u_sum += np.linalg.norm(P[i,:]-P[i-1,:])**alpha
        u[i] = u_sum
    
    return u/max(u)

#-------------------------------------------------------------------------------
# Find Minimum by Golden Section Search Method
# - Return x minimizing function f(x) on interval a,b
#-------------------------------------------------------------------------------
def find_min_gss(f, a, b, eps=1e-4):
    
    # Golden section: 1/phi = 2/(1+sqrt(5))
    R = 0.61803399
    
    # Num of needed iterations to get precision eps: log(eps/|b-a|)/log(R)
    n_iter = int(np.ceil(-2.0780869 * np.log(eps/abs(b-a))))
    c = b - (b-a)*R
    d = a + (b-a)*R

    for i in range(n_iter):
        if f(c) < f(d):
            b = d
        else:
            a = c
        c = b - (b-a)*R
        d = a + (b-a)*R

    return (b+a)/2

def iterative_param(P, u, fxcoeff, fycoeff):
    
    global iter_i
    u_new = u.copy()
    f_u = np.zeros(2)

    #--- Calculate approx. error s(u) related to point P_i
    def calc_s(u):
        f_u[0] = pl.polyval(u, fxcoeff)
        f_u[1] = pl.polyval(u, fycoeff)

        s_u = np.linalg.norm(P[i]-f_u)
        return s_u
    
    #--- Find new values u that locally minimising the approximation error (excl. fixed end-points)
    for i in range(1, len(u)-1):
        
        #--- Find new u_i minimising s(u_i) by Golden search method
        u_new[i] = find_min_gss(calc_s, u[i-1], u[i+1])
        
    return u_new

def bestFit(P):
    #-------------------------------------------------------------------------------
    # Options for the approximation method
    #-------------------------------------------------------------------------------
    polydeg = 3           # Degree of polygons of parametric curve
    n = len(P)
    w = np.ones(n)           # Set weights for knot points
    w[0] = w[-1] = 1e6
    max_iter = 20         # Max. number of iterations
    eps = 1e-3

    #-------------------------------------------------------------------------------
    # Init variables
    #-------------------------------------------------------------------------------
    f_u = np.zeros([n,2])
    uu = np.linspace(0,1,desired_points_count)
    f_uu = np.zeros([len(uu),2])
    S_hist = []

    #-------------------------------------------------------------------------------
    # Compute the iterative approximation
    #-------------------------------------------------------------------------------
    for iter_i in range(max_iter):

        #--- Initial or iterative parametrization
        if iter_i == 0:
            # u = uniform_param(P)
            # u = chordlength_param(P)
            u = centripetal_param(P)
        else:
            u = iterative_param(P, u, fxcoeff, fycoeff)
        
        #--- Compute polynomial approximations and get their coefficients
        fxcoeff = pl.polyfit(u, P[:,0], polydeg, w=w)
        fycoeff = pl.polyfit(u, P[:,1], polydeg, w=w)
        
        #--- Calculate function values f(u)=(fx(u),fy(u),fz(u))
        f_u[:,0] = pl.polyval(u, fxcoeff)
        f_u[:,1] = pl.polyval(u, fycoeff)
        
        #--- Calculate fine values for ploting
        f_uu[:,0] = pl.polyval(uu, fxcoeff)
        f_uu[:,1] = pl.polyval(uu, fycoeff)
        
        #--- Total error of approximation S for iteration i
        S = 0
        for j in range(len(u)):
            S += w[j] * np.linalg.norm(P[j] - f_u[j])
        
        #--- Add bar of approx. error
        S_hist.append(S)
        
        #--- Stop iterating if change in error is lower than desired condition
        if iter_i > 0:
            S_change = S_hist[iter_i-1] / S_hist[iter_i] - 1
            #print('iteration:%3i, approx.error: %.4f (%f)' % (iter_i, S_hist[iter_i], S_change))
            if S_change < eps:
                break
    
    return f_uu

count = 1
for xy in XY:
    step_size = len(xy) / desired_points_count
    interpolated_frame = []
    interpolated_f = bestFit(xy)

    interpolated_f[:,0] = interpolated_f[:,0]*resX/screenX
    interpolated_f[:,1] = interpolated_f[:,1]*resY/screenY

    # # Linear interpolation for the frame
    # for i in range(desired_points_count):
    #     index = int(i * step_size)
    #     fraction = (i * step_size) - index

    #     if index < len(xy) - 1:
    #         x1, y1 = xy[index][0]*resX/screenX, xy[index][1]*resY/screenY
    #         x2, y2 = xy[index + 1][0]*resX/screenX, xy[index + 1][1]*resY/screenY

    #         interpolated_x = x1 + (x2 - x1) * fraction
    #         interpolated_y = y1 + (y2 - y1) * fraction
    #         interpolated_frame.append((interpolated_x, interpolated_y))
    #     else:
    #         # Handle the case where index is out of bounds
    #         interpolated_frame.append((xy[-1][0]*resX/screenX, xy[-1][1]*resY/screenY))

    # Add the interpolated frame to the list
    XY_interpolated.append(interpolated_f)

# Define the names for the output CSV files
x_file = 'x.csv'
y_file = 'y.csv'

# Open the CSV files for writing
with open(x_file, 'w', newline='') as file1, open(y_file, 'w', newline='') as file2:
    writer1 = csv.writer(file1)
    writer2 = csv.writer(file2)

    # Iterate through the main list
    for i in range(desired_points_count):
        # Extract the x and the y from each tuple
        columnX = [round(item[i][0],2) for item in XY_interpolated]
        columnY = [round(item[i][1],2) for item in XY_interpolated]

        # Write the columns to the respective CSV files
        writer1.writerow(columnX)
        writer2.writerow(columnY)

print(f"CSV files {x_file} and {y_file} have been created.")

X_data = pd.read_csv("x.csv", header=None)
Y_data = pd.read_csv("y.csv", header=None)

fig, ax = plt.subplots()
for i in range(len(X_data.columns)):
    ax.clear()
    ax.invert_yaxis()
    x = X_data[i]
    y = Y_data[i]
    ax.plot(x,y,"-o")
    plt.pause(0.1)