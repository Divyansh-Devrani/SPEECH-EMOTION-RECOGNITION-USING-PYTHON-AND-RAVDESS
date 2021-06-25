import librosa
import soundfile
import os, glob, pickle , sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import time
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfile 
import tkinter.messagebox
from playsound import playsound
import matplotlib.pyplot as plt
import wave, sys

filename = None
bgcolor = 'lightgreen'
model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)

root = tk.Tk()
root.title("Emotion Recognition from Speech")
canvas = tk.Canvas(root, width = 800, height = 500,  relief = 'raised' , bg=bgcolor)
canvas.pack()

labelHeading = tk.Label(root, text="Emotion Recognition from Speech",  fg="blue" , bg=bgcolor)
labelHeading.config(font=('helvetica', 26))
canvas.create_window(400, 50, window=labelHeading) 

labelSelectInput = tk.Label(root, text="Select Input:",  fg="black" , bg=bgcolor)
labelHeading.config(font=('helvetica', 16 , 'bold'))
canvas.create_window(100, 150, window=labelSelectInput)

txtInputFile = tk.Text(root, height = 1, width = 30)
txtInputFile.config(font=('helvetica', 16))
txtInputFile.pack()
canvas.create_window(350, 150, window=txtInputFile)

labelNotification = tk.Label(root, text="Notification:",  fg="black" , bg=bgcolor)
labelNotification.config(font=('helvetica', 16))
canvas.create_window(100, 250, window=labelNotification)

txtNotification = tk.Text(root, height = 3, width = 40)
txtNotification.config(font=('helvetica', 16))
txtNotification.pack()
canvas.create_window(450, 250, window=txtNotification)
   
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result

emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}

observed_emotions=['calm', 'happy', 'fearful', 'disgust']

def load_data(test_size=0.2):
    x,y=[],[]
    for file in glob.glob("E:\\Speech Emotion Recognition\\ravdess_data\\Actor_*\\*.wav"):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

def visualize(path: str):
    raw = wave.open(path)
    signal = raw.readframes(-1)
    signal = np.frombuffer(signal, dtype ="int16")
    f_rate = raw.getframerate()
    time = np.linspace(
        0, # start
        len(signal) / f_rate,
        num = len(signal)
    )
    plt.figure(1)
    plt.title("Sound Wave")
    plt.xlabel("Time")
    plt.plot(time, signal)
    plt.show()
    
def SelectInputFile():
    global filename
    file = askopenfile(mode ='r', filetypes =[('All Files', '*.wav')]) 
    if file is not None: 
        txtInputFile.delete("1.0",tk.END)
        filepath = file.name
        dirname, basename = os.path.split(filepath)
        txtInputFile.insert(tk.END,basename)
        filename = file
 
        
def PlayAudio():
    global filename
    if filename is not None: 
        playsound(filename.name)
        visualize(filename.name)
    
def TrainModel():
    global model
    x_train,x_test,y_train,y_test=load_data(test_size=0.25)
    #print((x_train.shape[0], x_test.shape[0]))
    #print(f'Features extracted: {x_train.shape[1]}')
    model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
    model.fit(x_train,y_train)
    txtNotification.delete("1.0",tk.END)
    txtNotification.insert(tk.END,'Training Done Successfully' )

def Predict():
    global model
    if filename is not None:
        playsound(filename.name)
        feature=extract_feature(filename.name, mfcc=True, chroma=True, mel=True)
        feature = feature.reshape(1, -1)
        y_pred=model.predict(feature)
        #print("y_pred:",y_pred)
        txtNotification.delete("1.0",tk.END)
        txtNotification.insert(tk.END,y_pred[0] )
    
def Quit():
    root.destroy()
    sys.exit()

btnBrowse = tk.Button(text='Browse', command=SelectInputFile, bg='green', fg='white', font=('helvetica', 12, 'bold'))
canvas.create_window(600, 150, window=btnBrowse)

btnX = 200
btnY = 400

btnPlayAudio = tk.Button(text='Play Audio', command=PlayAudio, bg='green', fg='white', font=('helvetica', 12, 'bold'))
canvas.create_window(btnX, btnY, window=btnPlayAudio)

btnX = btnX + 150
btnTrainModel = tk.Button(text='Train Model', command=TrainModel, bg='green', fg='white', font=('helvetica', 12, 'bold'))
canvas.create_window(btnX, btnY, window=btnTrainModel)

btnX = btnX + 150
btnPredict = tk.Button(text='Predict', command=Predict, bg='green', fg='white', font=('helvetica', 12, 'bold'))
canvas.create_window(btnX, btnY, window=btnPredict)

btnX = btnX + 100
btnQuit = tk.Button(text='Quit', command=Quit, bg='green', fg='white', font=('helvetica', 12, 'bold'))
canvas.create_window(btnX, btnY, window=btnQuit)

root.mainloop()
