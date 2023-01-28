###############################################################################################################################
#                                                   ===== Import Library =====                                                #
###############################################################################################################################
# keras & tensorflow
from keras.models import load_model

# File
import json
import pickle

# DIP
from PIL import Image
from PIL.Image import fromarray
import cv2

# numpy
from numpy import asarray
from numpy import expand_dims
from numpy import round
from numpy import linalg

# os
from os import listdir, mkdir
from os.path import exists
from os import system

# date & time
from datetime import datetime
import time

# GUI
from tkinter import *
from tkinter import messagebox
from tkinter import ttk
###############################################################################################################################



###############################################################################################################################
#                                                    ===== FUNCTIONS =====                                                    #
###############################################################################################################################
# Code Here
###############################################################################################################################



###############################################################################################################################
# Load Data and Basic Configuration
###############################################################################################################################
system("cls")
x = datetime.now()
date = str(x.day) + "/" + str(x.month) + "/" + str(x.year)

# Load FaceNet model & HaarCascade file
print("Load FaceNet model & HaarCascade file\n")
# MyFaceNet = load_model("Model\\facenet_keras.h5")
# HaarCascade = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))

# Training Face Path
savePath = "Data\\Dataset\\train"
if not exists(savePath):
    mkdir(savePath)

###############################################################################################################################



###############################################################################################################################
#                                                      ====== GUI ======                                                      #
###############################################################################################################################

#=============================================================================================================================#
#                                                      ----- Menu -----                                                       #
#=============================================================================================================================#
# Sub Menu 1 - Training Wajah
def trainingData() :
    print("Open submenu 1 - Training Wajah")        # debug status button
    return 0

# Sub Menu 2 - Ambil Abensisi
def ambilAbsensi() :
    print("Open submenu 2 - Ambil Abensisi")        # debug status button
    return 0

# Sub Menu 3 / Rekap Absensi
def rekapAbsensi() :
    print("Open submenu 3 - Rekap Absensi")         # debug status button
    rekap = Toplevel(window)
    rekap.title("Rekap Absensi")
    rekap.iconbitmap("D:/Dev/Python/tkinter/img/ok.ico")
    w,h = 700,800
    x,y = int((screenWidth/2) - (w/2)), int((screenHeight/2) - (h/2))
    rekap.minsize(700,800)
    rekap.geometry(f"{w}x{h}+{x+500}+{y}")
    rekap.resizable(0,0)
    
    style = ttk.Style()
    
    style.configure("Treeview",
        # background="silver",
        foreground="black",
        rowheight=30
    )
    
    style.map("Treeview", background=[("selected", "black")])
    
    # Grid Formate
    rekap.columnconfigure(0, weight=10)
    rekap.columnconfigure(1, weight=10)
    rekap.columnconfigure(2, weight=1)
    rekap.columnconfigure(3, weight=1)
    rekap.columnconfigure(4, weight=1)
    
    # Judul
    maintitle = Label(rekap, text="REKAP ABSENSI", font=('Times New Roman',30,'bold'))
    maintitle.grid(row=0, column=0, columnspan=5, pady=25)
    # Label Tanggal
    dateLabel = Label(rekap, text="DATE: ")
    dateLabel.grid(row=1, column=2, pady=10, sticky="E")
    # Entry Tanggal
    dateEntry = Entry(rekap, width=10)
    dateEntry.grid(row=1, column=3, pady=10, sticky="WE")
    dateEntry.insert(0, date)
    # Tombol Pencarian
    search = Button(rekap, text="show", command=lambda:insertTreeview(dateEntry.get()))
    search.grid(row=1, column=4, pady=10, sticky="W", padx=5)
    
    tableFrame = LabelFrame(rekap)
    tableFrame.grid(padx=20, row=2, column=0, columnspan=5, sticky="WENS")
    
    # Define table
    global table
    table = ttk.Treeview(tableFrame, height=20)
    
    # Define columns
    table['columns'] = ("no", "nobp", "nama", "date", "hour", "ket")
    
    table.tag_configure("odd", background="white")
    table.tag_configure("even", background="silver")
    
    # formate columns
    table.column("#0", width=0, stretch=NO)
    table.column("no", anchor=CENTER, width=30, minwidth=30)
    table.column("nobp", anchor=W, width=100, minwidth=100)
    table.column("nama", anchor=W, width=190, minwidth=190)
    table.column("date", anchor=CENTER, width=100, minwidth=100)
    table.column("hour", anchor=CENTER, width=100, minwidth=100)
    table.column("ket", width=120, minwidth=120)
    
    # create headings
    table.heading("#0", anchor=W)
    table.heading("no", text="NO", anchor=CENTER)
    table.heading("nobp", text="NOBP", anchor=W)
    table.heading("nama", text="Nama", anchor=W)
    table.heading("date", text="Tanggal", anchor=CENTER)
    table.heading("hour", text="Jam", anchor=CENTER)
    table.heading("ket", text="Keterangan", anchor=W)
    
    # Add Data
    insertTreeview(date)
    
    # Pack to screen
    # table.grid(padx=20, row=2, column=0, columnspan=5)
    table.pack(side=LEFT, expand=1)
    
    scrollbar = ttk.Scrollbar(tableFrame, orient=VERTICAL, command=table.yview)
    table.configure(yscrollcommand=scrollbar.set)
    # scrollbar.grid(row=0, column=1, sticky="NS")
    # scrollbar.pack(side=RIGHT, fill=Y)
    scrollbar.pack(side=RIGHT, fill=Y)
    return 0
#=============================================================================================================================#



#=============================================================================================================================#
#                                                    ----- Main Menu -----                                                    #
#=============================================================================================================================#
window = Tk()
window.title("Smart Attendance")
window.iconbitmap("D:/Dev/Python/tkinter/img/ok.ico")

width, height = 800, 500
screenWidth, screenHeight = window.winfo_screenwidth(), window.winfo_screenheight()
x,y = int((screenWidth/2) - (width/2)), int((screenHeight/2) - (height/2))
window.geometry(f"{width}x{height}+{x}+{y-50}")
window.resizable(0,0)           # Disable resize Main Window  

# Setup for Main Grid / Grid on Main Window(window)
window.columnconfigure(0, weight=1)
window.rowconfigure(0, weight=1)
# window.rowconfigure(0, weight=9)
# window.rowconfigure(1, weight=1)

# Main Frame
mainFrame = LabelFrame(window)
mainFrame.grid(row=0, column=0, sticky='WENS')

# Grid Setup for Main Frame
mainFrame.columnconfigure(0, weight=1)
mainFrame.rowconfigure(0, weight=6)
mainFrame.rowconfigure(1, weight=4)

# Define title and button frame in Main frame
titleFrame = Frame(mainFrame)
buttonFrame = Frame(mainFrame)
titleFrame.grid(row=0,column=0,sticky='WENS')
buttonFrame.grid(row=1,column=0,sticky='WENS')

# Title Frame Grid Configuration
titleFrame.rowconfigure(0, weight=1)
titleFrame.rowconfigure(1, weight=1)
titleFrame.rowconfigure(2, weight=1)
titleFrame.columnconfigure(0, weight=1)
titleFrame.columnconfigure(1, weight=1)
titleFrame.columnconfigure(2, weight=1)
titleFrame.columnconfigure(3, weight=1)
titleFrame.columnconfigure(4, weight=1)

# Button Frame Grid Configuration
buttonFrame.columnconfigure(0, weight=1)
buttonFrame.columnconfigure(1, weight=1)
buttonFrame.columnconfigure(2, weight=1)
buttonFrame.columnconfigure(3, weight=1)
buttonFrame.columnconfigure(4, weight=1)
buttonFrame.columnconfigure(5, weight=1)
buttonFrame.columnconfigure(6, weight=1)

# Title Frame
mainTitle = Label(titleFrame, text="APLIKASI SMART ATTENDANCE", font=('Times New Roman',30,'bold'))
mainTitle.grid(row=2, column=2)

# Button Frame
button1 = Button(buttonFrame, text="TRAINING WAJAH", padx=50, pady=25, command=trainingData)
button2 = Button(buttonFrame, text="AMBIL ABSENSI", padx=50, pady=25, command=ambilAbsensi)
button3 = Button(buttonFrame, text="REKAP ABSENSI", padx=50, pady=25, command=rekapAbsensi)
button1.grid(row=0, column=2)
button2.grid(row=0, column=3)
button3.grid(row=0, column=4)

# Footer / Copyright
copyright = u"\u00A9"
copyright = Label(window, text="Muhammad Yasir " + copyright + " 2022")
# copyright = Label(window, text="Muhammad Yasir " + copyright + " 2022\n Contact : yasir112358@gmail.com")
copyright.grid()

window.mainloop()
#=============================================================================================================================#
###############################################################################################################################