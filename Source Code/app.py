import tkinter as tk
from tkinter import filedialog, ttk
import tkinter.font as tkFont
from PIL import Image, ImageTk
import os
from utils import resize

import features
import svm


TARGET_SIZE = 200
img_path = None
init_method = 0

def open_image():
    global img_path
    initial_dir = os.getcwd()
    initial_dir = os.path.join(initial_dir,'datasets','demo')
    img_path = filedialog.askopenfilename(initialdir=initial_dir, title="Select Image", filetypes=(("Image files", "*.jpg *.jpeg *.png *.gif"), ("all files", "*.*")))
    if img_path:
        #check if image size is 200x200
        img = Image.open(img_path)
        
        if img.width != 200 or img.height != 200:
            img_path=resize.resize_img(img_path, False)     #resize actual size
       
        img = Image.open(img_path)   
        img_copy = img.copy()
        resized_image = img_copy.resize((400, 300))        # resize for the window
        photo = ImageTk.PhotoImage(resized_image)
        canvas.create_image(200, 200, image=photo, anchor=tk.CENTER)
        canvas.image = photo
        prediction_label.config(text="")        #reset label

def predict():
    global img_path, init_method
    print("Init method", init_method)
    selected_method = method_selector.current()
    print("Selected method", selected_method)
   
    if img_path and selected_method>=0:
        features.get_features(img_path,selected_method)
        pred = svm.make_predictions(selected_method)
        prediction = "Real" if pred == 0 else "Fake"
        prediction_label.config(text=prediction)
        
def change_method(event):
    global init_method
    selected_method = method_selector.current()
    
    if selected_method != init_method:
        prediction_label.config(text="")
        init_method = selected_method

root = tk.Tk()
root.title("GAN Image Classifier")
root.geometry("700x600")

root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

btn_font = tkFont.Font(family="Helvetica", size = 14, weight="bold")
lbl_font = tkFont.Font(family="Helvetica", size = 20, weight="bold")

canvas_frame = tk.Frame(root)
canvas_frame.grid(row=1, column=0, columnspan=2, sticky="ew")

canvas = tk.Canvas(canvas_frame, width=400, height=400)
canvas.pack(side="top", fill=None, expand=True)  


btn_frame = tk.Frame(root)
btn_frame.grid(row=0, column=0, columnspan=2, pady=(20,0), sticky="ew")  

btn_open = tk.Button(btn_frame, text="Open Image", command=open_image, font=btn_font, height =2, width=50)
btn_open.pack() 



method_selector = ttk.Combobox(root, values=["DCT", "DWT", "BOTH"], font=btn_font,width=1)
method_selector.grid(row=2, column=0, padx=10, pady=5, sticky="ew") 
method_selector.current(0)  
#check if method changed
method_selector.bind("<<ComboboxSelected>>",change_method)


btn_predict = tk.Button(root, text="Predict", command=predict, font=btn_font,height=2)
btn_predict.grid(row=2, column=1, padx=10, pady=10, sticky="ew")  


predict_label = tk.Label(root, text="Image Prediction: ",font=btn_font)
predict_label.grid(row=3, column=0, padx=10, pady=5, sticky="ew")  

prediction_label = tk.Label(root, text="",font=lbl_font)
prediction_label.grid(row=3, column=1, padx=10, pady=10, sticky="ew")  

root.mainloop()
