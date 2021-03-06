#..............................................................ASSIGNENT1............................................................................................
#........................................................IMPORTING HEADER FILES......................................................................................
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys 

def conv(img):
	fit=np.array([[0,0,]])

#.......................................HISTOGRAM ANALYSIS OF AN IMGAGE..........................................................................
def histogram(img):
	T=img.shape[0]*img.shape[1] 
	bint=np.zeros(256) #collecting intensities for blue frame
	gint=np.zeros(256) #collecting intensities for green frame
	rint=np.zeros(256) #collecting intensities for red frame
	
	h=img.shape[0]
	w=img.shape[1]
	b,g,r=cv.split(img)             #spliting colors
	for i in range(h):
		for j in range(w):
			bint[b[i][j]]+=1
	bint/=T
			
	for i in range(h):
		for j in range(w):
			gint[g[i][j]]+=1
	gint/=T
			
	for i in range(h):
		for j in range(w):
			rint[r[i][j]]+=1
			
	rint/=T
	
	plt.plot(np.arange(0,256,1),bint,'-b.')
	plt.plot(np.arange(0,256,1),gint,'-g.')
	plt.plot(np.arange(0,256,1),rint, '-r.')
	plt.title("Histogram")
	plt.xlabel("Intensity of Pixel")
	plt.ylabel("Probabilty mass function")
	plt.show()
	
	                     
#.............................................SOME USEFUL FUNCTIONS.............................................................
def max_val(a,b,c):     #used in median cut
	if a>=b and a>=c:
		return 0
	elif b>=a and b>=c:
		return 1
	elif c>=b and c>=a:
		return 2
		
def toArr(img):                #use to find second parameter in median cut and 
	arr=[]
	for r_index,r in enumerate(img):
		for c_index,color in enumerate(r):
			arr.append([color[0],color[1],color[2],r_index,c_index])	
	return np.array(arr)
		
		
def distance(d1,d2):        #used in floydsteinberg
	d=((d1[0]-d2[0])**2+(d1[1]-d2[1])**2+(d1[2]-d2[2])**2)
	return d
	
def minmax(val):     #used in floydsteinberg

	if val[0]<0:
		val[0]=0
	if val[1]<0:
		val[1]=0
	if val[2]<0:
		val[2]=0
	if val[0]>255:
		val[0]=255
	if val[1]>255:
		val[1]=255
	if val[2]>255:
		val[2]=255
		
	return val
	
	
def findIndex(arr):
	b=np.mean(arr[:,0])       #finding means of bgr colors
	g=np.mean(arr[:,1])
	r=np.mean(arr[:,2])
	
	d2=np.array([b,g,r])
	index=0
	m=sys.maxsize
	for data in arr:
		d1=np.array([data[0],data[1],data[2]])
		dis=distance(d1,d2)
		if dis<m:
			m=dis
			index=data[3]
	return index
	

#...........................................UNIFORM QUANTIZATION...................................................................

def uniform(img,k):                           
	M=int(255/k)                      #diving into k boxes
	h=img.shape[0]
	w=img.shape[1]
	s=np.zeros(k+1,dtype=int)
	
	for i in range(1,k+1):
		s[i]=s[i-1]+M
	b,g,r=cv.split(img)
	
	for i in range(h):
		for j in range(w):
			for m in range(1,k+1):
				if(b[i][j]>=s[m-1] and b[i][j]<s[m]):
					b[i][j]=s[m-1]
					
	for i in range(h):
		for j in range(w):
			for m in range(1,k+1):
				if(g[i][j]>=s[m-1] and g[i][j]<s[m]):
					g[i][j]=s[m-1]
	for i in range(h):
		for j in range(w):
			for m in range(1,k+1):
				if(r[i][j]>=s[m-1] and r[i][j]<s[m]):
					r[i][j]=s[m-1]
					
	unif_img=cv.merge([b,g,r])
	
	return unif_img			
				
#...........................................................................................................................	
	
#.......................................................THE POPULARITY ALGORITHM...............................................
def popularAlgo(img,k):
	T=img.shape[0]*img.shape[1]
	color_count=np.zeros(256**3, dtype=int)  #bgr is represented by 256^2*r+256*g+b
	
	indexToColor=[]                      #give color for given index
	for r in range(256):
		for g in range(256):
			for b in range(256):
				indexToColor.append([b,g,r])
	indexToColor=np.array(indexToColor)
	
	
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			color_count[(256**2)*img[i][j][2]+256*img[i][j][1]+img[i][j][0]]+=1
			
	M=T
	arr=[]
	

	for i in range(k):
		index=0
		m=0
		for ix,count in enumerate(color_count):
			if count>m and count<M:
				m=count
				index=ix
				
		arr.append(index)
		M=m
	arr.append(0)
		
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			color=img[i][j][0]+256*img[i][j][1]+(256**2)*img[i][j][2]
			for l in range(1,k+1):
				if color<arr[l-1] and color>=arr[l]:
					img[i][j]=indexToColor[arr[l-1]]
					
	return img
	
#.......................................................................................................................

#...............................................MEDIAN CUT.............................................................

def medianCut(img,arr,k):
	if len(arr)==0:
		return 
		
	if k==0:
		m_b=np.mean(arr[:,0])
		m_g=np.mean(arr[:,1])
		m_r=np.mean(arr[:,2])
		for d in arr:
			img[d[3]][d[4]]=[m_b,m_g,m_r]
		return 
		
	else:
		b_range=np.max(arr[:,0])-np.min(arr[:,0])
		g_range=np.max(arr[:,1])-np.min(arr[:,1])
		r_range=np.max(arr[:,2])-np.min(arr[:,2])
		
		sel=max_val(b_range,g_range,r_range) #return 0,1,2 depending of which color have wider range
		
		arr=arr[arr[:, sel].argsort()]
		
		mid_val=(len(arr)+1)//2   #median of arr
	
		
		medianCut(img,arr[0:mid_val],k-1)
		medianCut(img,arr[mid_val:],k-1)
			
#......................................................................................................................

#..........................................FLOYD STEMBURG BY EXHAUSTIVE SEARCH.............................................................
	
def floydSteinberg(img,arr,k):
	if len(arr)==0:
		return
	elif k==0:
		for data in arr:
			old_pix=np.array([data[0],data[1],data[2]])
			h=data[3]
			w=data[4]
			
			new_pix=np.round((7*old_pix/255))*255/7
			
			e=old_pix-new_pix
			if h<img.shape[0]-1:
				img[h+1][w]=minmax(img[h+1][w]+e*5/16)
			if h<img.shape[0]-1 and w<img.shape[1]-1:
				img[h+1][w+1]=minmax(img[h+1][w+1]+e*1/16)
			if w<img.shape[1]-1:
				img[h][w+1]=minmax(img[h][w+1]+e*7/16)
			if h<img.shape[0]-1 and w>1:
				img[h+1][w-1]=minmax(img[h+1][w-1]+e*3/16)
			
		return 
		
	else:
		index=findIndex(arr)
		
		floydSteinberg(img,arr[0:index],k-1)
		floydSteinberg(img,arr[index:],k-1)
#......................................................................................................................
#..................................................Main.................................................................

if __name__=="__main__":
	src=input("Enter Image Location: ")         #taking image location
	k=int(input("Enter Value k: ") )  #no of boxes for quantization
	
	img=cv.imread(src)
	img1=img.copy()
	img2=img.copy()
	img3=img.copy()
	img4=img.copy()
	
	arr1=toArr(img3)
	arr2=toArr(img4)
	
	img1=uniform(img1,k)
	img2=popularAlgo(img2,k)
	medianCut(img3,arr1,k)
	floydSteinberg(img4,arr2,k)
	
	
	plt.subplot(2,2,1)
	plt.title("Uniform Quantization")
	plt.xticks([])
	plt.yticks([])
	plt.imshow(cv.cvtColor(img1,cv.COLOR_BGR2RGB))
	
	plt.subplot(2,2,2)
	plt.title("Popular Algorithm Quantization")
	plt.xticks([])
	plt.yticks([])
	plt.imshow(cv.cvtColor(img2,cv.COLOR_BGR2RGB))
	
	
	plt.subplot(2,2,3)
	plt.title("medianCut (no dither)")
	plt.xticks([])
	plt.yticks([])
	plt.imshow(cv.cvtColor(img3,cv.COLOR_BGR2RGB))
	
	plt.subplot(2,2,4)
	plt.title("FloydSteinberg (dither)")
	plt.xticks([])
	plt.yticks([])
	plt.imshow(cv.cvtColor(img4,cv.COLOR_BGR2RGB))
	
	plt.show()
	
	histogram(img1)
	histogram(img2)
	histogram(img3)
	histogram(img4)
	
#............................................THE END.......................................................................




