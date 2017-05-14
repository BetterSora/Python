# -*- coding: utf-8 -*-
"""
Created on Sat May 13 20:37:05 2017

@author: Qin
"""

import numpy as np
import tkinter as tk
import regTrees
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

def reDraw(tolS, tolN):
    '绘制树'
    reDraw.f.clf()
    reDraw.a = reDraw.f.add_subplot(111)
    if chkBtnVar.get():
        if tolN < 2: 
            tolN = 2
        myTree=regTrees.createTree(reDraw.rawDat, regTrees.modelLeaf, regTrees.modelErr, (tolS, tolN))
        yHat = regTrees.createForeCast(myTree, reDraw.testDat, regTrees.modelTreeEval)
    else:
        myTree=regTrees.createTree(reDraw.rawDat, ops = (tolS, tolN))
        yHat = regTrees.createForeCast(myTree, reDraw.testDat)
    reDraw.a.scatter(reDraw.rawDat[:, 0], reDraw.rawDat[:, 1], s = 5, color = 'red') 
    reDraw.a.plot(reDraw.testDat, yHat, linewidth = 2.0)
    reDraw.canvas.show()

def getInputs():
    '得到输入框的值'
    try:
        tolN = int(tolNentry.get())
    except ValueError:
        tolN = 10
        print('enter Integer for tolN')
        tolNentry.delete(0,'end')
        tolNentry.insert(0, 10)
    try:
        tolS = float(tolSentry.get())
    except ValueError:
        tolS = 1.0
        print('enter Float for tolS')
        tolNentry.delete(0, 'end')
        tolNentry.insert(0, 1.0)
    
    return tolN, tolS

def drawNewTree():
    tolN, tolS = getInputs()
    reDraw(tolS, tolN)

root = tk.Tk()

reDraw.f = Figure(figsize = (5, 4), dpi = 100) #create canvas
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master = root)
reDraw.canvas.show()
reDraw.canvas.get_tk_widget().grid(row = 0, columnspan = 3)

tk.Label(root, text = 'tolN').grid(row = 1, column = 0)
tolNentry = tk.Entry(root)
tolNentry.grid(row = 1, column = 1)
#insert(index, string)
tolNentry.insert(0, '10')
tk.Label(root, text = 'tolS').grid(row = 2, column = 0)
tolSentry = tk.Entry(root)
tolSentry.grid(row = 2, column = 1)
tolSentry.insert(0, '1.0')
tk.Button(root, text = 'Redraw', command = drawNewTree).grid(row = 1, column = 2, rowspan = 3)

chkBtnVar = tk.IntVar()
chkBtn = tk.Checkbutton(root, text = 'Model Tree', variable = chkBtnVar)
chkBtn.grid(row = 3, column = 0, columnspan = 2)

reDraw.rawDat = np.mat(regTrees.loadDataSet('sine.txt'))
reDraw.testDat = np.arange(min(reDraw.rawDat[:, 0]), max(reDraw.rawDat[:, 0]), 0.01)

root.mainloop()