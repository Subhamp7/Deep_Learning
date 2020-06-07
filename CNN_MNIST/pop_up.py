import tkinter as tk
from tkinter import simpledialog

ROOT = tk.Tk()
ROOT.withdraw()
# the input dialog
def index_name():
    USER_INP = simpledialog.askstring(title="CNN",
                                      prompt="Enter any random index(0-59999):")
    if(int(USER_INP)<0 or int(USER_INP)>59999):
        index_name()
    print(("The image and value with index {} be displayed now..... ").format(USER_INP))
    return int(USER_INP)

