import tkinter as tk
from tkinter import messagebox
from tkinter import*
from recognition_methods import RecognitionMethods
recognizer = RecognitionMethods()
class Aplicacion:
    def __init__(self):
        self.ventana1=tk.Tk()
        self.ventana1.geometry('150x80')
        self.ventana1.title("Creando usuario")
        self.frame= Frame()
        self.frame.pack()
        self.label1=tk.Label(self.frame,text="Ingrese nombre de usuario:")
        self.label1.grid(column=0, row=2)
        self.dato=tk.StringVar()
        self.entry1=tk.Entry(self.frame, width=20, textvariable=self.dato)
        self.entry1.grid(column=0, row=4)
        self.boton1=tk.Button(self.frame, text="Ingresar", command=self.ingresar)
        self.boton1.grid(column=0, row=6)
        self.ventana1.mainloop()
    def ingresar(self):
        #recognizer.nuevoUsuario(self.dato.get())
        print("ya guardo fotos")
        recognizer.EntrenarUsuario()
        print("ya entreno")
        self.ventana1.destroy()   