import tkinter as tk
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import messagebox
from PIL import ImageTk, Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


root = tk.Tk()									
root.title("Image Processsing")
my_font = ('times', 18, 'bold')


#------------------------------------------------------------------------------------------------------
# Helper Functions
#Function 1  : uploading & Displaying input image
def upload ():
    f_types = [('JPG files','*.jpg'),('PNG files', '*.png'),('JPEG files', '*.jpeg')]
    filename = tk.filedialog.askopenfilename(filetypes = f_types)
    img = Image.open(filename) 
    img = img.resize((360,360))
    img1 = ImageTk.PhotoImage(img) #display it
    label = tk.Label(frame1)
    label.grid(row =1,column =1)
    label.image = img1
    label['image'] = img1
    return img


#Function 2  : Displaying output image
def display(img):
    img2 = Image.fromarray(img)
    img2 = ImageTk.PhotoImage(img2) #display it
    label = tk.Label(frame2)
    label.grid(row =1,column =1)
    label.image = img2
    label['image'] = img2

#Function 3 : Convert to gray scale
def RGB_gs(img):
   
    img_array = np.array(img)
    nimg_array=np.zeros(shape=(360,360))
    for i in range (len(img_array)):
        for j in range(len(img_array[i])):
            blue = img_array[i,j,0]
            green = img_array[i,j,1]
            red = img_array[i,j,2]
            grayscale_value = blue*0.114 + green*0.587 + red*0.299
            nimg_array[i,j] = grayscale_value
       
    return nimg_array;


#Function 4 : Image Convolution
def Convolve(img , mask):

    
    row,col = img.shape
    m,n = mask.shape
    arr = np.zeros((row+m-1, col+n-1))
    n = n//2
    m = m//2
    filtered_img = np.zeros(img.shape)
    arr[m:arr.shape[0]-m,n:arr.shape[1]-n] = img
    for i in range(m, arr.shape[0]-m):
        for j in range(n, arr.shape[1]-n):
            temp = arr[i-m:i+m+1 , j-m:j+m+1]
            res = temp * mask
            filtered_img[i-m, j-n] = res.sum()

    return filtered_img

#Helper functions of Laplacian of Gaussian

#Function 5 : Compute the second image derevative and smooth it
#Generate laplacian of gaussian filter
def log(img ,size, sigma):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]

    temp1 = -1 / (np.pi * sigma**4)
    temp2 = ((x**2) + (y**2))/( 2 * sigma**2)
    g = temp1 * -temp2 * np.exp(-temp2)

    ni = Convolve(img,g)

    return ni


#Function 6  :zero crossing function for keeping or removing the edge
#x=-1,y=1 with x=1,y=-1
#x=-1,y=0 with x=1,y=0
#x=-1,y=-1 with x=1,y=1
#x=0,y=1,with x=0,y=-1
def zeros(mat):
    if((mat[0,0]*mat[2,2]) >0):
        if((mat[1,0]*mat[1,2]) >0):
            if((mat[2,0]*mat[0,2]) >0):
                if((mat[0,1]*mat[2,1]) >0):
                    mat[1,1]=0
                else:
                    mat[1,1] = mat[1,1]
            else:
               mat[1,1]=mat[1,1]
        else:
            mat[1,1]=mat[1,1]
    else:
        mat[1,1]=mat[1,1]

#Function 7: 
def zero_cross(img):
    img2=img.copy()
    for i in range (1,img.shape[0]-1):
        for j in range (1,img.shape[0]-1):
            array = np.array(img[i-1:i+2,j-1:j+2])
            zeros(array)
            img2[i-1:i+2,j-1:j+2]=array

    #we change the values in new image not in the given one
    return img2;


#Canny helper functions 
#Function 8  : Gaussian Smoothing
def gaussian_smothing(img,size, sigma):

    # generate gaussian kernal
    size = int(size) // 2
    #mgrid function for create dense multidimensional array
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    gauss_kernel =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal

    # image convolution using gaussian kernal
    filtered_img = Convolve(img,gauss_kernel)

    return filtered_img

#Function 9 : Gradient calculation  (Gradient Magnitude and Gradient Direction) 
def Gradient_cal(img):

    #sobel filters
    sobel_x = np.array([[-1, 0, 1], 
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1,-2,-1]])
    
    #convolve using vertical filter
    Ix = Convolve(img,sobel_x)

    #convolve using horizontal filter
    Iy = Convolve(img,sobel_y)
    
    #calculate gradient magnitude  using manhaten distance
    Gmag = Ix.copy()
    for i in range(Ix.shape[0]):
        for j in range(Ix.shape[1]):
            Gmag[i,j]= np.abs(Ix[i,j] - Iy[i,j])

    #calculate gradient orintation*
    Gdir = np.degrees(np.arctan2(Iy, Ix))
    return (Gmag,Gdir)


# Function 10 : Non Maximum Suppression
def non_max_suppression(gradient_magnitude, gradient_direction):

    #Check if the pixels on the same direction are more or less intence than the ones being processed.
    #If one those pixels are more intence than the one being processed, then only the more intense one is kept
    image_row, image_col = gradient_magnitude.shape
 
    arr_output = np.zeros(gradient_magnitude.shape)
 
    PI = 180
 
    for row in range(1, image_row - 1):
        for col in range(1, image_col - 1):
            direction = gradient_direction[row, col]

            #angle 0 
            if (0 <= direction < PI / 8) or (15 * PI / 8 <= direction <= 2 * PI):
                before_pixel = gradient_magnitude[row, col - 1]
                after_pixel = gradient_magnitude[row, col + 1]
 
            #angle 45
            elif (PI / 8 <= direction < 3 * PI / 8) or (9 * PI / 8 <= direction < 11 * PI / 8):
                before_pixel = gradient_magnitude[row + 1, col - 1]
                after_pixel = gradient_magnitude[row - 1, col + 1]
 
            #angle 90
            elif (3 * PI / 8 <= direction < 5 * PI / 8) or (11 * PI / 8 <= direction < 13 * PI / 8):
                before_pixel = gradient_magnitude[row - 1, col]
                after_pixel = gradient_magnitude[row + 1, col]
 
            #angle 135
            else:
                before_pixel = gradient_magnitude[row - 1, col - 1]
                after_pixel = gradient_magnitude[row + 1, col + 1]
 
            if gradient_magnitude[row, col] >= before_pixel and gradient_magnitude[row, col] >= after_pixel:
                arr_output[row, col] = gradient_magnitude[row, col]
 
    
 
    return arr_output

#Function 11  : Hysteresis Thresholding 
def Do_Thresh_Hyst(img,TH,TL):
    
    GSup = np.copy(img)
    n,m = GSup.shape
    
    highThreshold = TH

    lowThreshold =TL  
    
    for i in range(1,n-1):
        for j in range(1,m-1):

            #If intensity of pixel is bigger than highthreshold then it is an edge
            if(GSup[i,j] > highThreshold):
                GSup[i,j] = 255

            #If intensity of pixel is lower than lowThreshold then it isn't an edge
            elif(GSup[i,j] < lowThreshold):
                GSup[i,j] = 0
            else:
                #If one of neighbors of processed pixel is an edge then the processed pixel is an edge
                if((GSup[i-1,j-1] > highThreshold) or 
                    (GSup[i-1,j] > highThreshold) or
                    (GSup[i-1,j+1] > highThreshold) or
                    (GSup[i,j-1] > highThreshold) or
                    (GSup[i,j+1] > highThreshold) or
                    (GSup[i+1,j-1] > highThreshold) or
                    (GSup[i+1,j] > highThreshold) or
                    (GSup[i+1,j+1] > highThreshold)):
                    GSup[i,j] = 255
        
  
    
    return GSup





#----------------------------------------------------------------------------------------------------------------
#Main functions

#Function 1 : Laplacian of Gausse
##Detect edge by implement Laplacian of Gausse
def LOG():
    img = upload()
    img_array = RGB_gs(img)
    T = int(retrieve_input(textBoxTH))
    #calculate the second derevative and smooth it
    sig = int(retrieve_input(textBoxsig))
    size = int(retrieve_input(textBoxsize))
    sec_der_img = log(img_array,size,sig)

    #detecting the zero crossing
    sec_der_img = zero_cross(sec_der_img)

    #getting the magnitude 
    x= Gradient_cal(sec_der_img)[0]

    #keeping the the edge >T
    for i in range (sec_der_img.shape[0]):
        for j in range(sec_der_img.shape[0]):
            if x[i,j]<T :
                x[i,j] = 0
            else:
                x[i,j] = 255

    display(x)

#-------------------------------------------------------------------------------------------

#Function 2 : Canny Edge Detector
def Canny_fct():

    img = upload()
    #convert images into grayscale, cause this algorithm based on grayscal images
    img_array = RGB_gs(img)

    
    #calculate the second derevative and smooth it
    sig = int(retrieve_input(textBoxsig))
    size = int(retrieve_input(textBoxsize))
    #Step 1: Reduse the noise using Gaussian smothing
    smooth_img = gaussian_smothing( img_array , size , sig )

    #Step 2: Calculate Gradient
    gradient_magnitude, gradient_direction = Gradient_cal(smooth_img)

    #Step 3: Non-Maximum Suppression
    thin_edge = non_max_suppression(gradient_magnitude, gradient_direction)

    #Step 4: Hysteresis Thresholding
    TH = int(retrieve_input(textBoxTH))
    TL = int(retrieve_input(textBoxTL))
    final_img = Do_Thresh_Hyst(thin_edge,TH,TL)
    display(final_img)


#---------------------------------------------------------------------------------------------------------------

canvas = tk.Canvas(root,height = 600, width = 1200, bg = '#D8BFD8')
canvas.pack()

frame1 = tk.Frame(root, bg ="white")
frame1.place(relwidth =0.3, relheight = 0.6,relx = 0.1, rely = 0.18  )


frame2 = tk.Frame(root, bg ="white")
frame2.place(relwidth =0.3, relheight = 0.6,relx = 0.6, rely = 0.18 )

Upl = Label (root,text = 'uploaded image',bg = '#D8BFD8' )
Upl.config(font=('Helvatical',13))
Upl.place(x=230,y=475) 

Out = Label (root,text = 'Output image',bg = '#D8BFD8' )
Out.config(font=('Helvatical',13))
Out.place(x=850,y=475) 

Title = Label(root,text='TP Digital Image ',bg = '#D8BFD8')
Title.config(font=('Helvatical',30))
Title.place(x =470, y= 30)

Note = Label(root,text ='Note:',bg = '#D8BFD8')
Note.place(x =90, y= 550)

Ins = Label(root,text = 'click on button so you can play the function.',bg = '#D8BFD8')
Ins.place(x =75, y= 570)


########################################################################################################################


radio = IntVar()


def retrieve_input(textBox):
    inputValue=textBox.get("1.0",END)
    return inputValue


T_shold = Label(root,text = 'Enter the threshold.',bg = '#D8BFD8')
T_shold.place(x =520, y= 240)
textBoxTH=Text(root, height=1, width=5)
textBoxTH.place(x =580, y= 260)


TL_shold = Label(root,text = 'Enter the low threshold for canny.',bg = '#D8BFD8')
TL_shold.place(x =520, y= 280)
textBoxTL=Text(root, height=1, width=5)
textBoxTL.place(x =580, y= 300)

K_size = Label(root,text = 'Enter size of karnel.',bg = '#D8BFD8')
K_size.place(x =520, y= 320)
textBoxsize=Text(root, height=1, width=5)
textBoxsize.place(x =580, y= 340)

S_sig = Label(root,text = 'Enter sigma.',bg = '#D8BFD8')
S_sig.place(x =520, y= 360)
textBoxsig=Text(root, height=1, width=5)
textBoxsig.place(x =580, y= 380)

LGauss = tk.Button(root, text = "Laplacian of Gaussian" , padx=25, pady=5, command=lambda : LOG())
LGauss.place(x =520, y= 420)




Canny = tk.Button(root, text = "Canny algorithm" , padx=38, pady=5, command=lambda: Canny_fct())
Canny.place(x =520, y= 470)

root.mainloop()




