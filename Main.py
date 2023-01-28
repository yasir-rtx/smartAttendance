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
# Generate file absensi.json
def generateAbsensiFile(data):
    print("Generate file : absensi.json")
    with open(absensi_path, "w") as absensi:
        json.dump(data, absensi, indent=4)
    print("File absensi is created\n")
    #########################################
    
# Generate object key(date) into absensi.json
def generateKeyDate(date):
    print(f"Generate key({date})")
    with open(absensi_path, "r") as absensi:
        data = json.load(absensi)
        data.update({date:[]})
        json.dump(data, open(absensi_path, "w"), indent=4)
    print(f"Key({date}) for today is created\n")
###########################################################

# Generate objects key(list mahasiswa) into object key(date)
def generateListMahasiswa(date):
    print("Generate Mahasiswa Key")
    with open(mahasiswa_path, "r") as file:
        mahasiswa = json.load(file)
        mahasiswaValue = mahasiswa.values()
        # Generate mahasiswa key into date object in absensi.json
        for value in mahasiswaValue:
            # store nobp value into temporary variable
            nobp = value[0]["nobp"]
            print(f"Generate Mahasiswa Key for {nobp}")
            with open(absensi_path, "r") as absensi:
                # Generate mahasiswa value into date object in absensi.json
                y = {nobp:{
                    "nobp": nobp,
                    "nama": value[0]["nama"],
                    "date": date,
                    "time": "-",
                    "ket": "Tidak Hadir"
                }}
                data = json.load(absensi)
                # metode dump
                temp = data[date]   # akses object hari ini
                temp.append(y)
                json.dump(data, open(absensi_path, "w"), indent=4)
                print(f"Key({nobp}) is generated into object({date})\n")
    print("Mahasiswa Key is generated\n")
########################################################################

# Nobp check
def cekNobp(nobp):
    y = datetime.now()
    dates = str(y.day) + "/" + str(y.month) + "/" + str(y.year)
    with open(absensi_path, "r") as absensi:
        data = json.load(absensi)
        temp = data[dates]
        status = 0
        for mahasiswa in temp:
            for key, value in mahasiswa.items():
                if key == nobp:
                    status = 1
        # print(status)   # Jika status = 0, artinya nobp belum ada di dalam absensi.json
        return status
    
# Generate new nobp in absensi.json
def generateNewNobp(nobp):
    y = datetime.now()
    dates = str(y.day) + "/" + str(y.month) + "/" + str(y.year)
    with open(absensi_path, "r") as absensi:
        data = json.load(absensi)
        temp = data[dates]
        
            
        newData = []
        for y in temp:
            newData.append(y)
        
        # print(newData)
        
        with open(mahasiswa_path, "r") as mahasiswa:
            entry = json.load(mahasiswa)
            value = entry[nobp][0]
        
        newData.append(
            {
                nobp:{
                    "nobp": value["nobp"],
                    "nama": value["nama"],
                    "date": dates,
                    "time": "-",
                    "ket": "Tidak Hadir"
        }})
        
    # print(newData)
    data.update({dates:newData})
    # print(data)
    json.dump(data, open(absensi_path, "w"), indent=4)

# Take face images and generate signature
def pose(nobp, nama, password):
    # print(f"Nobp : {nobp} | Nama : {nama} | Password : {password}")
    cap = cv2.VideoCapture(1)
    label = nobp + " - " + nama + "\\"
    labelPath = savePath + label
    print(labelPath)
    if not exists(labelPath):
        training.destroy()
        mkdir(labelPath)
        for i in range(3):
            desc = "front" if i==0 else "side" if i==1 else "tambahan"
            messagebox.showinfo(f"Tahap {i}", f"Deskripsi Pose {desc}")
            sum = 0
            while 1:
                key = cv2.waitKey(5) & 0xFF
                _, imgVideo = cap.read()
                FaceDetect = HaarCascade.detectMultiScale(imgVideo, 1.3, 5)
                if len(FaceDetect) > 0:
                    x1, y1, w, h = FaceDetect[0]
                else :
                    x1, y1, w, h = 1,1,10,10
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1+w, y1+h
                img = cv2.cvtColor(imgVideo, cv2.COLOR_BGR2RGB)
                img = fromarray(img)
                img_array = asarray(img)
                face = img_array[y1:y2, x1:x2]
                face = fromarray(face)
                face_save = face.resize((160,160))
                face = asarray(face_save)
                if key == 13:
                    fileName = labelPath + nama + "_" + desc + "_" + str(sum) + ".jpg"
                    print(f"saving : {fileName}")
                    face_save.save(fileName)
                    sum+=1
                cv2.rectangle(imgVideo, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.imshow('Rekam Wajah', imgVideo)
                if sum == 10:
                    break
                if key == 27:
                    break
            cv2.destroyAllWindows()
        
        for files in listdir(labelPath):
            file = labelPath + files
            face_raw = cv2.imread(file)
            face = cv2.cvtColor(face_raw, cv2.COLOR_BGR2RGB)
            face = fromarray(face)
            face = face.resize((160,160))
            face = asarray(face)
            face = face.astype('float32')
            mean, std = face.mean(), face.std()
            face = (face - mean) / std

            face = expand_dims(face, axis=0)
            print(f"Training file : {files}")
            signature = MyFaceNet.predict(face)
            faceDatabase[nobp] = signature
            
        mySignature = open("Data\\signature.pkl", "wb")
        pickle.dump(faceDatabase, mySignature)
        mySignature.close()
        print("Face signature is created\n")
        
        with open(mahasiswa_path, "r") as file:
            data = json.load(file)
            
            data.update({nobp:[{"nobp": nobp, "nama": nama, "password": password}]})
            json.dump(data, open(mahasiswa_path, "w"), indent=4)
            
        # Insert new mahasiswa into absensi.json
        if not cekNobp(nobp):
            generateNewNobp(nobp)
        
        messagebox.showinfo("Smart Attendance", "Data Wajah Berhasil Direkam")
    else:
        print("Label sudah ada")
        messagebox.showerror("Smart Attendance", "Label sudah ada.")
    cap.release()
############################################################################################

# Take Attendance
def takeAttendance(nobp): 
    # dates="14/12/2022"    # debug fungsi untuk memastikan perubahan hanya terjadi di hari itu
    y = datetime.now()
    dates = str(y.day) + "/" + str(y.month) + "/" + str(y.year)
    times = str(y.hour) + ":" + str(y.minute) + ":" + str(y.second)
    with open(absensi_path, "r") as absensi:
        data = json.load(absensi)
        temp = data[dates]
        
        newData = []
        for mahasiswa in temp:
            # print(mahasiswa)        # dictionary mahasiswa dalam objek date
            for key, value in mahasiswa.items():
                if key == nobp:
                    # print("mahasiswa ditemukan")
                    newData.append(
                        {
                            nobp:{
                                "nobp": value["nobp"],
                                "nama": value["nama"],
                                "date": dates,
                                "time": times,
                                "ket": "Hadir"
                    }})
                else:
                    # print("mahasiswa tidak ditemukan")
                    newData.append(mahasiswa)

    # print(newData)  # debug nilai newdata sebelum di dump
    data.update({dates:newData})
    # print(data)
    json.dump(data, open(absensi_path, "w"), indent=4)
    print(f"{nobp} telah hadir pada {dates}")
    messagebox.showinfo("Rekam Absensi", "Absensi anda untuk hari ini berhasil direkam")
#####################################################################################

# Pre Ambil Absensi
def preAbsen():
    with open(mahasiswa_path, "r") as mahasiswa:
        data = json.load(mahasiswa)
        # Check mahasiswa.json is empty
        if data == {}:
            messagebox.showwarning("Ambil Absensi", "Absensi Tidak dapat dilakukan.\n\nCause    : Tidak data wajah yang tersimpan.\nSolution: Lakukan Training Wajah terlebih dahulu.")
        else:
            # messagebox.showinfo("Ambil Absensi", "Ambil absensi dapat dilakukan")
            ambilAbsensi()

def askPassword():
    global passForm
    passForm = Toplevel()
    passForm.title("Ambil Absensi")
    passForm.iconbitmap("D:/Dev/Python/tkinter/img/ok.ico")
    w,h = 400,100
    x,y = int((screenWidth/2) - (w/2)), int((screenHeight/2) - (h/2))
    passForm.geometry(f"{w}x{h}+{x}+{y-50}")
    
    labelpassword = Label(passForm, text="Password: ")
    # global passwordForm
    passwordForm = Entry(passForm, width=40)
    btnOk = Button(passForm, text="OK", width=10, command=lambda: getPassword(passwordForm.get()))
    btnCancel = Button(passForm, text="CANCEL", width=10, command=passForm.destroy)
    
    # Grid Configuration for Ambil Absensi Form
    passForm.columnconfigure(0, weight=1)
    passForm.columnconfigure(1, weight=1)
    passForm.columnconfigure(2, weight=1)
    passForm.columnconfigure(3, weight=1)
    passForm.rowconfigure(0, weight=1)
    passForm.rowconfigure(1, weight=1)
    passForm.rowconfigure(2, weight=1)
    passForm.rowconfigure(3, weight=1)
    
    # render form
    labelpassword.grid(row=1, column=1, sticky='E')
    passwordForm.grid(row=1, column=2, columnspan=3)
    btnOk.grid(row=2, column=1, columnspan=2)
    btnCancel.grid(row=2, column=2, columnspan=2)
    
def getPassword():
    # passForm.destroy()
    # print(x)
    askPassword()
    return x

# Ambil Absensi
def absen(nobp):
    absensiForm.destroy()
    signatureBase = faceDatabase
    status = 0
    identity = 0
    for key, value in signatureBase.items():
        if key == nobp:
            status = 1
            print(f"Signature {nobp} found")
            
            cap = cv2.VideoCapture(1)
            t=1
            while t:
                # time.sleep(0)
                _, imgVideo = cap.read()
                FaceDetect = HaarCascade.detectMultiScale(imgVideo, 1.3, 10)
                
                for (x1, y1, width, height) in FaceDetect:
                    x1, y1 = abs(x1), abs(y1)
                    x2, y2 = x1 + width, y1 + height
                    
                    img = cv2.cvtColor(imgVideo, cv2.COLOR_BGR2RGB)
                    img = fromarray(img)
                    img_array = asarray(img)
                    
                    face = img_array[y1:y2, x1:x2]
                    
                    face = fromarray(face)
                    face = face.resize((160,160))
                    face = asarray(face)
                    
                    face = face.astype('float32')
                    mean, std = face.mean(), face.std()
                    face = (face - mean) / std
                    
                    face = expand_dims(face, axis=0)
                    signature = MyFaceNet.predict(face)
                    
                    distance = linalg.norm(value - signature)
                    if (distance > 7):
                        cv2.rectangle(imgVideo, (x1,y1), (x2,y2), (0,0,255), 2)
                        cv2.rectangle(imgVideo, (x1,y1-40), (x2,y1), (0,0,255), -2)
                        cv2.putText(imgVideo, "Unknown", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
                    else:
                        identity = key
                        cv2.rectangle(imgVideo, (x1,y1), (x2,y2), (0,255,0), 2)
                        cv2.rectangle(imgVideo, (x1,y1-40), (x2,y1), (0,255,0), -2)
                        cv2.putText(imgVideo, identity + ", " + str(round(distance, 2)), (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
                        
                cv2.imshow('Face Recognition', imgVideo)   
                
                # # Jika ingin proses verifikasi berjalan otomatis
                # time.sleep(5)
                # break
                # Menampilkan proses verifiaksi di kamera
                k = cv2.waitKey(5) & 0xFF
                if k == 27:
                    t-=1
            cv2.destroyAllWindows()
            cap.release()
        
    if status != 1:
        messagebox.showwarning("Rekam Absensi", "Data anda tidak ditemukan.\n Periksa nobp!")        
    else:
        if identity == 0:
            messagebox.showwarning("Rekam Absensi", "Wajah tidak cocok. Silahkan isi password jika nobp identitas anda benar")
            def askPassword():
                global formPassword
                formPassword = Toplevel()
                formPassword.title("Ambil Password")
                formPassword.iconbitmap("D:/Dev/Python/tkinter/img/ok.ico")
                w,h = 400,100
                x,y = int((screenWidth/2) - (w/2)), int((screenHeight/2) - (h/2))
                formPassword.geometry(f"{w}x{h}+{x}+{y-50}")
                label = Label(formPassword, text="Password: ")
                label.pack()
                passw = Entry(formPassword, width=40)
                passw.pack()
                btnOk = Button(formPassword, text="OK", width=10, command=lambda: getPassword(passw.get()))
                btnCancel = Button(formPassword, text="CANCEL", width=10, command=formPassword.destroy)
                btnOk.pack()
                btnCancel.pack()
            
            def getPassword(passwordAsk):
                formPassword.destroy()
                # print(f"Password Input : {passwordAsk}")
            
                # Ambil password dari mahasiswa.json
                with open(mahasiswa_path, "r") as mahasiswa:
                    data = json.load(mahasiswa)
                    temp = data[nobp]
                    passwordMhs =  temp[0]["password"]
                    print(f"Password Asli : {passwordMhs}")
                
                if passwordAsk != "":
                    print(f"Password Input : {passwordAsk}")
                    if passwordAsk == passwordMhs:
                        takeAttendance(nobp)
                        # print("password cocok")
                    else:
                        messagebox.showerror("Ambil Absensi", "Password Salah")
                        askPassword()
                else:
                    # print("Password is empty")
                    messagebox.showerror("Ambil Absensi", "Password Kosong")
                    askPassword()
                
            
            askPassword()
            
            
        else:
            # check if absensi.json and mahasiswa.json is ready
            if absensi_path:
                print("absensi.json is ready")
            if mahasiswa_path:
                print("mahasiswa.json is ready")
            takeAttendance(nobp)
            
# Show record of "rekap absensi"
def insertTreeview(newDate):
    date = newDate
    # Delete treeviw/table
    for record in table.get_children():
        # print(record)
        table.delete(record)

    # update treeview/table
    with open(absensi_path, "r") as file:
        absensi = json.load(file)
        data = absensi[date]
        for i, mahasiswa in enumerate(data):
            for key, value in mahasiswa.items():
                if i % 2 == 0:
                    table.insert(parent="", index="end", iid=i, text="", values=(i+1, value["nobp"], value["nama"], value["date"], value["time"], value["ket"]), tags="even")
                else:
                    table.insert(parent="", index="end", iid=i, text="", values=(i+1, value["nobp"], value["nama"], value["date"], value["time"], value["ket"]), tags="odd")
###############################################################################################################################



###############################################################################################################################
# Load Data and Basic Configuration
###############################################################################################################################
system("cls")
x = datetime.now()
date = str(x.day) + "/" + str(x.month) + "/" + str(x.year)

# Load FaceNet model & HaarCascade file
print("Load FaceNet model & HaarCascade file\n")
MyFaceNet = load_model("Model\\facenet_keras.h5")
HaarCascade = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))

# Directory Data
dataPath = "Data\\"
if not exists(dataPath):
    mkdir(dataPath)
    print("Directory Data is created")

# Dataset Path
datasetPath = dataPath + "Dataset\\"
if not exists(datasetPath):
    mkdir(datasetPath)
    print("Directory Dataset is created")

# Training Face Path
savePath = datasetPath + "train\\"
if not exists(savePath):
    mkdir(savePath)
    print("Directory Data Training is created")
    
# Testing Face Path
trainPath = datasetPath + "test\\"
if not exists(trainPath):
    mkdir(trainPath)
    print("Directory Data Testing is created")
    
# Face Signature
signature_path = "Data\\signature.pkl"
if exists(signature_path):
    mySignature = open("Data\\signature.pkl", "rb")
    faceDatabase = pickle.load(mySignature)
    mySignature.close()
else:
    faceDatabase = {}
    
# Mahasiswa
mahasiswa_path = "Data\\mahasiswa.json"
if not exists(mahasiswa_path):
    print("Generate File : mahasiswa.json")
    data = {}
    with open(mahasiswa_path, "w") as mahasiswa:
        json.dump(data, mahasiswa, indent=4)
        print("File mahasiswa.json is creasted\n")
        
# Absensi
absensi_path = "Data\\absensi.json"
# Prepare absensi.json
if not exists(absensi_path):                                    # create absensi.json
    generateAbsensiFile({})
    generateKeyDate(date)
    generateListMahasiswa(date)
else:                                                           # if absensi.json has been created
    # check if date object is already made or not
    finding = 0
    with open(absensi_path, "r") as absensi:
        data = json.load(absensi)
        dates = data.keys()
        for dateFound in dates:
            if dateFound == date:
                finding = 1
    if finding == 1:                                            # prevent object(date) from being reset
        print("File absensi.json is ready")
    else:                                                       # generate object(date)
        generateKeyDate(date)
        generateListMahasiswa(date)
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
    global training
    training = Toplevel()
    training.title("Training Data")
    training.iconbitmap("D:/Dev/Python/tkinter/img/ok.ico")
    w,h = 400,200
    x,y = int((screenWidth/2) - (w/2)), int((screenHeight/2) - (h/2))
    training.geometry(f"{w}x{h}+{x}+{y-50}")
    
    labelnobp = Label(training, text="NOBP: ")
    nobp = Entry(training, width=40)
    labelnama = Label(training, text="NAMA: ")
    nama = Entry(training, width=40)
    labelpassword = Label(training, text="PASSWORD: ")
    password = Entry(training, width=40)
    btnOk = Button(training, text="OK", width=10, command=lambda: pose(nobp.get(), nama.get(), password.get()))
    btnCancel = Button(training, text="CANCEL", width=10, command=training.destroy)
    
    # Grid Configuration for Training Form
    training.columnconfigure(0, weight=1)
    training.columnconfigure(1, weight=1)
    training.columnconfigure(2, weight=1)
    training.columnconfigure(3, weight=1)
    training.rowconfigure(0, weight=1)
    training.rowconfigure(1, weight=1)
    training.rowconfigure(2, weight=1)
    training.rowconfigure(3, weight=1)
    training.rowconfigure(4, weight=1)
    training.rowconfigure(5, weight=1)
    
    # render form
    labelnobp.grid(row=1, column=1, sticky='E')
    nobp.grid(row=1, column=2, columnspan=3)
    labelnama.grid(row=2, column=1, sticky='E')
    nama.grid(row=2, column=2, columnspan=3)
    labelpassword.grid(row=3, column=1, sticky='E')
    password.grid(row=3, column=2, columnspan=3)
    btnOk.grid(row=4, column=1, columnspan=2)
    btnCancel.grid(row=4, column=2, columnspan=2)

# Sub Menu 2 - Ambil Abensisi
def ambilAbsensi() :
    print("Open submenu 2 - Ambil Abensisi")        # debug status button
    global absensiForm
    absensiForm = Toplevel()
    absensiForm.title("Ambil Absensi")
    absensiForm.iconbitmap("D:/Dev/Python/tkinter/img/ok.ico")
    w,h = 400,100
    x,y = int((screenWidth/2) - (w/2)), int((screenHeight/2) - (h/2))
    absensiForm.geometry(f"{w}x{h}+{x}+{y-50}")
    
    labelnobp = Label(absensiForm, text="NOBP: ")
    nobp = Entry(absensiForm, width=40)
    btnOk = Button(absensiForm, text="OK", width=10, command=lambda: absen(nobp.get()))
    btnCancel = Button(absensiForm, text="CANCEL", width=10, command=absensiForm.destroy)
    
    # Grid Configuration for Ambil Absensi Form
    absensiForm.columnconfigure(0, weight=1)
    absensiForm.columnconfigure(1, weight=1)
    absensiForm.columnconfigure(2, weight=1)
    absensiForm.columnconfigure(3, weight=1)
    absensiForm.rowconfigure(0, weight=1)
    absensiForm.rowconfigure(1, weight=1)
    absensiForm.rowconfigure(2, weight=1)
    absensiForm.rowconfigure(3, weight=1)
    
    # render form
    labelnobp.grid(row=1, column=1, sticky='E')
    nobp.grid(row=1, column=2, columnspan=3)
    btnOk.grid(row=2, column=1, columnspan=2)
    btnCancel.grid(row=2, column=2, columnspan=2)

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
button2 = Button(buttonFrame, text="AMBIL ABSENSI", padx=50, pady=25, command=preAbsen)
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