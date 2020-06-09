from PIL import Image
import matplotlib.pyplot as plt
from scipy import cluster
import pandas as pd
import cv2 as cv
import numpy as np

def color_palette(file_name):
	im = Image.open(file_name)
	im = im.resize((1920,1080), Image.ANTIALIAS)
	im.save("cp.png",quality=95,optimize=True)

	#Converting Image from BGR TO RGB
	ima = cv.imread("cp.png")
	ima = cv.cvtColor(ima, cv.COLOR_BGR2RGB)

	#Accesing each pixel and taking the RGB values
	red, green, blue = [], [], []
	for line in ima:
	    for pixel in line:
	        r, g, b = pixel
	        red.append(r)
	        green.append(g)
	        blue.append(b)

	#Putting those RGB values in a dataset
	df = pd.DataFrame({
	    'red' : red,
	    'green' : green,
	    'blue' : blue
	})

	#The Whiten feature scales the data for K-means Algo
	df['standardized_red'] = cluster.vq.whiten(df['red'])
	df['standardized_green'] = cluster.vq.whiten(df['green'])
	df['standardized_blue'] = cluster.vq.whiten(df['blue'])

	color_pallete, distortion = cluster.vq.kmeans(df[['standardized_red', 'standardized_green', 'standardized_blue']], 8)

	colors = []
	#Getting standard deviation of RGB values to convert values from scaled to normal
	red_std, green_std, blue_std = df[['red', 'green', 'blue']].std()

	#Final converted RGB values will be stores in colors list
	for color in color_pallete:
	    sc_red, sc_green, sc_blue = color
	    colors.append([
	            int(sc_red * red_std) ,
	            int(sc_green * green_std),
	            int(sc_blue * blue_std)])

	width, height = im.size
	pl_width, pl_height = width, height
	splw = (pl_width-84)//8
	splh = pl_height//4
	gap = int(0.01*pl_height)
	pl_height = height+gap+splh
	canvas = Image.new('RGB', (pl_width,pl_height), (255,255,255))
	canvas.paste(im)
	j = 0
	for i in range(8):
	    palette_color = np.zeros((splh,splw,3), np.uint8)
	    palette_color[:,:] = colors[i]
	    plc = Image.fromarray(palette_color,'RGB')
	    canvas.paste(plc, (j,height+gap))
	    j += splw+14
	plt.imshow(canvas)
	plt.show()
	final = canvas.save("ColorPalette.png")

#Enter the file_name in your working directory
file_name = input()
color_palette(file_name)


