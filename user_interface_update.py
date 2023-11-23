import os
import cv2 
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras.preprocessing import image
from PIL import Image
import webcolors
import csv 
import random

cwd = os.getcwd()
users_path = cwd + '/user_list.txt'
base_matrix_path = cwd + '/compatibility_matrix.csv'
base_matrix_data=''
with open(base_matrix_path) as f:
    base_matrix_data=f.read()

# ----- LOAD SAVED MODEL -----
json_file = open('model.json', 'r')     
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load pre trained weights weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk.")

users=[]
with open(users_path) as f:
    for i  in f.readlines():
        users.append(i.strip())
print(users)

top={}
bottom={}
top_mer_path = cwd + '/top_merged.csv'
bottom_mer_path = cwd + '/bottom_merged.csv'

with open(top_mer_path) as f:
    reader=csv.reader(f)
    for row in reader:
        top[row[0]] = (row[2] , row[1]) #top id: pattern, color

with open(bottom_mer_path) as f:
    reader = csv.reader(f)
    for row in reader:
        bottom[row[0]] = (row[2], row[1])

def start():
    
    print("WELCOME TO KAKA OUTFIT RECOMMENDER:")
    print("1.Existing User\n2.New User\n")
    u = int(input())
    if(u==1):
        if(len(users)==0):  #based on users list file
            print("no existing users")
            return ""
        else:
            name = input("Enter your username:")
            if name not in users:
                print("User not found. Creating new account...")
                c=input("Embracing fashion trends over personal choice??(y or n)")
                dfactor=0
                if(c=='y'):
                    dfactor = 0.2
                else :
                    dfactor = 1
                dfactor_path = name + '_dfactor.txt'
                with open(dfactor_path, 'w') as f:  #creating d factor file for user
                    f.write(dfactor)
                with open(users_path, 'a') as f:    #updating users list file
                    f.write(name+'\n')
                cur_comp_matrix_path = name+'_matrix.csv'
                with open(cur_comp_matrix_path,'w') as f:
                    f.write(base_matrix_data)
            return name
    elif(u==2):
        name=input("Enter your username:")
        with open(users_path, 'a') as f:
            f.write(name+'\n')
        cur_comp_matrix_path = name+'_matrix.csv'
        with open(cur_comp_matrix_path,'w') as f:
            f.write(base_matrix_data)
        c=input("Embracing fashion trends over personal choice??(y or n)")
        dfactor=0
        if(c=='y'):
            dfactor = 0.2
        else :
            dfactor = 1
        dfactor_path = name + '_dfactor.txt'
        with open(dfactor_path,'w') as f:
            f.write(str(dfactor))
        return name
    else:
        print("Invalid Choice")
        return ""

def find_color(img):    #finding rgb
    white_threshold = 220
    img1 = Image.open(img)
    image_rgb = img1.convert('RGB')
    image_array = list(image_rgb.getdata())

    # Create a dictionary to count color occurrences, excluding white
    color_count = {}
    for pixel in image_array:
        r, g, b = pixel
        if r >= white_threshold and g >= white_threshold and b >= white_threshold:
            continue  # Ignore white pixels
        if pixel in color_count:
            color_count[pixel] += 1
        else:
            color_count[pixel] = 1

    # Find the color with the highest count
    if(len(color_count)==0):
        return (255, 255, 255)
    most_prominent_color = max(color_count, key=color_count.get)
    print(f'Most Prominent Color (RGB): {most_prominent_color}')
    return most_prominent_color




from scipy.spatial import KDTree
from webcolors import CSS3_HEX_TO_NAMES

def get_closest_color_name(rgb_value):  #kd tree for unknown colors 
    rgb_list = [value for value in rgb_value]
    css3_names = list(CSS3_HEX_TO_NAMES.values())
    css3_rgb_values = [webcolors.hex_to_rgb(webcolors.name_to_hex(name)) for name in css3_names]

    # Build a KD tree for efficient nearest-neighbor search
    tree = KDTree(css3_rgb_values)

    # Query the KD tree to find the closest color
    distance, index = tree.query(rgb_list)

    # Return the closest color name
    return css3_names[index]




def get_colour_name(requested_colour):
    try:
        closest_name =webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = get_closest_color_name(requested_colour)
    return closest_name


labels={0:'OTHER', 1:'animal', 2:'cartoon', 3:'chevron', 4:'floral',5:'geometry', 6:'houndstooth',7:'ikat',8:'letter_numb',9:'plain',10:'polka dot',11:'scales',12:'skull', 13:'squares', 14:'stars',15:'stripes',16:'tribal'}
def find_pattern(img):
    test_image = image.load_img(img, target_size=(64,64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)   #(1, 64, 64, 3) 1-> number of samples in a batch
    result = loaded_model.predict(test_image)
    index = np.where(result[0]==1)  #indices where predicted result isF
    if(len(index[0])==0):
        return labels[0]
    return labels[index[0][0]]

def rec(matrix, dfactor):
    print("1.Upload top\n2.Upload bottom\n3.Recommend from closet")
    ch = int(input())
    if(ch==1 or ch==2):
        path = input("Enter path:")
        rgb = find_color(path)
        color = get_colour_name(rgb)
        pattern = find_pattern(path)
        k=''
        res1=[]
        res2=[]
        res3=[]
        pair1=()
        pair2=()
        pair3=()
        if(ch==1):
            row_values = matrix.loc[(color, pattern)].values
            sorted_row_values = sorted(row_values, reverse=True)
            max1=sorted_row_values[0]
            pair1= matrix.columns[matrix.loc[(color, pattern)]==max1][0]
            max2=sorted_row_values[1]
            pair2 =matrix.columns[matrix.loc[(color, pattern)]==max2][0]
            max3=sorted_row_values[2]
            pair3 =matrix.columns[matrix.loc[(color, pattern)]==max3][0]
            k='bottom'
            for key,value in bottom.items():
                if(value == pair1):
                    res1.append(key)
                if(value == pair2):
                    res2.append(key)
                if(value == pair3):
                    res3.append(key)
            
        elif(ch==2):
        #max_value = matrix[(color, pattern)].max()
        #corresponding_top = matrix[(color, pattern)].idxmax()
            column_values = matrix.loc[(color, pattern)].values
            sorted_column_values = sorted(column_values, reverse=True)
            max1=matrix[(color, pattern)].nlargest(1).iloc[-1]
            pair1= matrix[matrix[(color, pattern)]==max1].index[0]
            max2=matrix[(color, pattern)].nlargest(2).iloc[-1]
            pair2= matrix[matrix[(color, pattern)]==max2].index[0]
            max3=matrix[(color, pattern)].nlargest(3).iloc[-1]
            pair3= matrix[matrix[(color, pattern)]==max3].index[0]
            k='top'
            for key,value in top.items():
                if(value == pair1):
                    res1.append(key)
                if(value == pair2):
                    res2.append(key)
                if(value == pair3):
                    res3.append(key)
        l1 = len(res1)
        l2 = len(res2)
        l3 = len(res3)

        #print(res)
        index1 = random.randint(0, l1-1)
        index2 = random.randint(0, l2-1)
        index3= random.randint(0, l3-1)
        res_path1 = cwd + '/FashionVCdata/'+k+'/'+res1[index1]+'.jpg'
        res_path2 = cwd + '/FashionVCdata/'+k+'/'+res2[index2]+'.jpg'
        res_path3 = cwd + '/FashionVCdata/'+k+'/'+res3[index3]+'.jpg'
    
        img1= Image.open(path)
        img2= Image.open(res_path1)
        img3= Image.open(res_path2)
        img4= Image.open(res_path3)
        fig, ax = plt.subplots(2,2, figsize=(10,10))
        ax[0, 0].imshow(img1)
        ax[0, 0].set_title('Input')
        ax[0, 1].imshow(img2)
        ax[0, 1].set_title('Suggestion 1')
        ax[1, 0].imshow(img3)
        ax[1, 0].set_title('Suggestion 2')
        ax[1, 1].imshow(img4)
        ax[1, 1].set_title('Suggestion 3')
        plt.show()
        print("Give rating on a scale of 1 to 10:")
        r1=int(input("Rating 1:"))
        r2=int(input("Rating 2:"))
        r3=int(input("Rating 3:"))
        if(ch==1):
            matrix.loc[(color, pattern),pair1] +=(r1-5)*dfactor
            matrix.loc[(color, pattern),pair2] +=(r2-5)*dfactor
            matrix.loc[(color, pattern),pair3] +=(r3-5)*dfactor
        else:
            matrix.loc[pair1,(color, pattern)] +=(r1-5)*dfactor
            matrix.loc[pair2,(color, pattern)] +=(r2-5)*dfactor
            matrix.loc[pair3,(color, pattern)] +=(r3-5)*dfactor

    else:
        t=int(input("No of tops:"))
        b=int(input("No of bottoms:"))
        tops=[]
        tops_feature=[]
        bottoms=[]
        bottoms_feature=[]
        print("Path of tops:")
        for i in range(t):
            s=input()
            t_rgb = find_color(s)
            t_colour = get_colour_name(t_rgb)
            t_pattern = find_pattern(s)
            tops.append(s)
            tops_feature.append((t_colour, t_pattern))
        print("Path of bottoms:")
        for i in range(b):
            s=input()
            b_rgb= find_color(s)
            b_colour = get_colour_name(b_rgb)
            b_pattern = find_pattern(s)
            bottoms.append(s)
            bottoms_feature.append((b_colour,b_pattern))
        max_val = -100000
        top_res=0
        bottom_res=0
        for i in range(t):
            for j in range(b):
                val=matrix.loc[tops_feature[i],bottoms_feature[j]]
                if(val>max_val):
                    max_val=val
                    top_res=i
                    bottom_res=j
        #print(top)
        #print(bottom)
        #print(top_res)
        #print(bottom_res)
        img1=Image.open(tops[top_res])
        img2=Image.open(bottoms[bottom_res])

        fig,ax=plt.subplots(1,2,figsize=(10,5))
        ax[0].imshow(img1)
        ax[0].set_title('Top')
        ax[1].imshow(img2)
        ax[1].set_title('Bottom')

        plt.show()
        rating=int(input("Outfit rating:"))
        matrix.loc[tops_feature[top_res],bottoms_feature[bottom_res]]+=dfactor*rating

    
    
        

    return matrix
    

        

i = start()
if(i==""):
    print("Bye!!")
else:
    matrix = pd.read_csv(cwd+'/'+i+'_matrix.csv', index_col=[0, 1], header=[0, 1])
    dfactor = float(open(i+'_dfactor.txt').read())

    print(matrix)
    while(True):
        print("1.Reccomend\n2.Exit")
        ch = int(input())
        if(ch!=1):
            matrix.to_csv(cwd+'/'+i+'_matrix.csv')

            break
        else:
            matrix=rec(matrix, dfactor)
