import tkinter as  tk
import numpy as np
#import kernels
import shutil
import tkinter.font as tkFont
import os
from tkinter import ttk
from PIL import Image, ImageTk 
from tkinter.filedialog import askopenfilename,askdirectory
from tkinter.messagebox import showerror
import pdb
import pickle
'''
def resize( w_box, h_box, pil_image): 
	w, h = pil_image.size  
	f1 = 1.0*w_box/w 
	f2 = 1.0*h_box/h    
	factor = min([f1, f2])   
	width = int(w*factor)    
	height = int(h*factor)    
	return pil_image.resize((width, height), Image.ANTIALIAS) 
'''
#['requests', 'size0', 'resolutionx', 'resolutiony', 'fps', 'spatial_complexity', 'temporal_complexity', 'color_complexity', 'chunk_complexity_variation', 'Animation', 'CoverSong', 'Gaming', 'HDR', 'HowTo', 'Lecture', 'LiveMusic', 'LyricVideo', 'Lyrics', 'MusicVideo', 'NewsClip', 'Sports', 'TelevisionClip', 'VR', 'VerticalVideo', 'Vlog']

encoder_dict = {'Animation': 9,\
				'CoverSong': 10,\
				'Gaming': 11,\
				'HDR': 12,\
				'HowTo': 13,\
				'Lecture': 14,\
				'LiveMusic': 15,\
				'LyricVideo': 16,\
				'Lyrics': 17,\
				'MusicVideo': 18,\
				'NewsClip': 19,
				'Sports': 20,\
				'TelevisionClip': 21,\
				'VR': 22,\
				'VerticalVideo': 23,\
				'Vlog': 24}
def gui():
	window = tk.Tk()
	window.title('Video Encoding Formating Recommendation')
	ft1 = tkFont.Font(family='Fixdsys', size=22, weight=tkFont.BOLD)
	#window.resizable(False, False)
	W = window.winfo_screenwidth()/1.5
	H = window.winfo_screenheight()/1.5
	W, H = int(W), int(H)
	window.geometry('{}x{}'.format(W, H))
	path = tk.StringVar()
	s1 = tk.StringVar()

	def selectPath():
		path.set(askopenfilename())

	def button1_func(infile, size0, width, height, fps, scomp, ccomp, tcomp, chunk_var, category, requests):

		features = np.zeros((1, 25))
		features[0][0] = np.log10(float(requests))
		features[0][1] = np.log10(float(size0))
		features[0][2] = float(width)
		features[0][3] = float(height)
		features[0][4] = float(fps)
		features[0][5] = float(scomp)
		features[0][6] = float(tcomp)
		features[0][7] = float(ccomp)
		features[0][8] = float(chunk_var)
		features[0][encoder_dict[category]] = 1.0
		model, scalar, feat_name = pickle.load(open('model.sav', 'rb'))
		x = scalar.transform(features)
		y = model.predict(x)
		print(features)
		s1.set('{}'.format(y[0]))


	frame12 = tk.Frame(window, width=100, height=100, bg='green', bd=1)
	#tk.Label(frame12, text="Input file:", font=ft1).grid(row=0, sticky=tk.W)
	tk.Label(frame12, text="Video size (B)", font=ft1).grid(row=1, sticky=tk.W)
	tk.Label(frame12, text="Video width", font=ft1).grid(row=2, sticky=tk.W)
	tk.Label(frame12, text="Video height", font=ft1).grid(row=3,sticky=tk.W)
	tk.Label(frame12, text="Frame per second", font=ft1).grid(row=4,sticky=tk.W)
	tk.Label(frame12, text="Spatical complexity", font=ft1).grid(row=5,sticky=tk.W)
	tk.Label(frame12, text="Color complexity", font=ft1).grid(row=6,sticky=tk.W)
	tk.Label(frame12, text="Temporal complexity", font=ft1).grid(row=7,sticky=tk.W)
	tk.Label(frame12, text="Chunk variation", font=ft1).grid(row=8,sticky=tk.W)
	tk.Label(frame12, text="Category", font=ft1).grid(row=9,sticky=tk.W)
	tk.Label(frame12, text="Requests", font=ft1).grid(row=10,sticky=tk.W)
	tk.Button(frame12, text = "Input file", font=ft1, fg = 'blue', command = selectPath).grid(row = 0, sticky=tk.W)
	e1 = tk.Entry(frame12,textvariable = path,font=ft1)
	e2 = tk.Entry(frame12, font=ft1)
	e3 = tk.Entry(frame12, font=ft1)
	e4 = tk.Entry(frame12, font=ft1)
	e5 = tk.Entry(frame12, font=ft1)
	e6 = tk.Entry(frame12, font=ft1)
	e7 = tk.Entry(frame12, font=ft1)
	e8 = tk.Entry(frame12, font=ft1)
	e9 = tk.Entry(frame12, font=ft1)
	e10 = ttk.Combobox(frame12, font=ft1, values=['Animation', 'CoverSong', 'Gaming', 'HDR', 'HowTo',\
												 'Lecture',  'LiveMusic', 'Lyrics', 'LyricVideo', 'MusicVideo',\
												 'NewsClip', 'Sports', 'TelevisionClip', 'VerticalVideo', 'Vlog', 'VR'])
	e11 = tk.Entry(frame12, font=ft1)

	e1.grid(row=0, column=1)
	e2.grid(row=1, column=1)
	e3.grid(row=2, column=1)
	e4.grid(row=3, column=1)
	e5.grid(row=4, column=1)
	e6.grid(row=5, column=1)
	e7.grid(row=6, column=1)
	e8.grid(row=7, column=1)
	e9.grid(row=8, column=1)
	e10.grid(row=9, column=1)
	e11.grid(row=10, column=1)
	#e6.current(0)
	button1 = tk.Button(frame12, text='Prediction', font=ft1, fg = 'red', command=lambda: button1_func(e1.get(), \
			e2.get(), e3.get(), e4.get(), e5.get(), e6.get(), e7.get(), e8.get(), e9.get(), e10.get(), e11.get())).grid(row=15, column=0, sticky=tk.W)
	
	#tk.Label(frame13, font=ft1, text="-----------------------------------------------------------------------").grid(row=0, sticky=tk.W)
	#tk.Label(frame12, font=ft1, text="The most energy/memory efficient encoding format : ").grid(row=16, sticky=tk.W)
	e1r = tk.Entry(frame12,font=ft1,textvariable = s1)
	e1r.grid(row=15, column=1)


	frame12.pack(side='left', anchor = 'n', padx=10, pady=10)
	#pdb.set_trace()
	window.mainloop()
if __name__ == "__main__":
	gui()