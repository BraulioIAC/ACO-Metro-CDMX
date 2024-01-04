import numpy as np
import matplotlib.pyplot as plt
import skimage as sk
import cv2
import pandas as pd
from math import sqrt

def DistEuclidiana(p1,p2):
    suma = 0
    for i in range(len(p1)):
        suma += (p1[i] - p2[i])**2
    d =sqrt(suma)
    return d



# img=cv2.imread('Metro.jpg')
# nodos = []
# def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         xy = "%d,%d" % (x, y)
#         cv2.circle(img, (x, y), 1, (255, 0, 0), thickness = -1)
#         cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
#                     1.0, (0,0,0), thickness = 1)
#         cv2.imshow("image", img)
#         nodos.append([x,y])
# cv2.namedWindow("image")
# cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
# while(1):
#     cv2.imshow("image", img)
#     if cv2.waitKey(0)&0xFF==27:
#         break
# cv2.destroyAllWindows()

# df = pd.DataFrame(nodos)
# df.to_csv('Nodos.csv')  

nodos = np.array(pd.read_csv('Nodos.csv'))
nodos = nodos[:,1:]

conexiones = np.array([[0,1],[0,6],
              [1,0],[1,2],[1,4],
              [2,1],[2,4],[2,3],
              [3,2],[3,5],
              [4,1],[4,2],[4,5],[4,7],
              [5,3],[5,4],[5,9],[5,12],
              [6,0],[6,10],[6,18],
              [7,4],[7,8],[7,10],
              [8,7],[8,11],[8,12],
              [9,5],[9,14],[9,22],
              [10,6],[10,7],[10,11],[10,15],
              [11,8],[11,10],[11,16],[11,17],
              [12,5],[12,8],[12,13],[12,14],
              [13,12],[13,14],[13,17],[13,21],
              [14,9],[14,12],[14,13],[14,22],
              [15,10],[15,16],[15,18],[15,19],
              [16,11],[16,15],[16,17],[16,20],
              [17,11],[17,13],[17,16],[17,20],
              [18,6],[18,15],[18,19],
              [19,15],[19,18],[19,20],
              [20,16],[20,17],[20,19],[20,21],
              [21,13],[21,20],[21,22],
              [22,9],[22,14],[22,21]])

distancias = np.zeros((len(conexiones),1))
visibilidad = np.zeros((len(conexiones),1))
feromona = np.ones((len(conexiones),1))*0.01

for i in range(len(conexiones)):
    distancias[i] = DistEuclidiana(nodos[conexiones[i][0]],nodos[conexiones[i][1]])
    visibilidad[i] = 1/distancias[i]



alpha = 1
beta = 0.01

nAnts = 50
allAnts = []

for i in range(nAnts):
    inicio = 22
    fin = 0
    anterior = -1
    currentAnt = []
    while(inicio != fin):
        rutasP = np.where(conexiones[:,0] == inicio)
        [indx,indy] = np.where(conexiones[rutasP,1] == anterior)
        if (indy.size > 0):
            rutasP = np.delete(rutasP, int(indy))
        
        denominador = ((feromona[rutasP[:]]**alpha)*(visibilidad[rutasP[:]]**beta))
        denominador = sum(denominador)
        
        probabilidad = (((feromona[rutasP[:]]**alpha)*(visibilidad[rutasP[:]]**beta)) / denominador)
        rectaSelect = []
        suma1 = 0
        for i in range(len(probabilidad)):
            suma1 = suma1 + probabilidad[i]
            rectaSelect.append(suma1)
        rectaSelect = np.array(rectaSelect)
        
        select = np.random.rand()
        [camx,camy] = np.where(select < rectaSelect)
        
        [antTox,antToy] = np.where(probabilidad  == probabilidad[camx[0]])
        anterior = inicio
        inicio = int(conexiones[np.array(rutasP).T[antTox[0]],1])
        currentAnt.append([anterior,inicio])
    allAnts.append(currentAnt)

# ------------------ Calcular distancias por cada ruta de hormiga -------------
allDists = []
for i in range(len(allAnts)):
    indDists = []
    for j in range(len(allAnts[i])):
        for k in range(len(conexiones)):
            if(allAnts[i][j] == conexiones[k]).all():
                indDists.append(k)
    allDists.append(sum(distancias[indDists]))            


# ----------------------------- Actualizar feromona ---------------------------
rho = 0.5 # Factor de evaporación
Q = 1 # Factor de olvido

for i in range(len(feromona)):
    dTau = 0
    for j in range(nAnts):
        paso = False
        hormiga = np.array(allAnts[j])
        pasoPor = np.zeros(np.shape(hormiga))
        pasoPor = np.where(hormiga == conexiones[i][0], True, pasoPor)
        pasoPor = np.where(hormiga == conexiones[i][1], True, pasoPor)
        for k in range(len(pasoPor)):
            if(pasoPor[k][0] == 1 and pasoPor[k][1] == 1): paso = True
        
        if (paso): dTau = dTau + (1/allDists[j])
    feromona[i] = (1 - rho) * feromona[i] + dTau
        
ElBraus = feromona/max(feromona)
    
    
inicio = 22
fin = 0
anterior = -1
rutaFinal = []
while(inicio != fin):
    rutasP = np.where(conexiones[:,0] == inicio)
    [indx,indy] = np.where(conexiones[rutasP,1] == anterior)
    if (indy.size > 0):
        rutasP = np.delete(rutasP, int(indy))
    
    denominador = ((feromona[rutasP[:]]**alpha)*(visibilidad[rutasP[:]]**beta))
    denominador = sum(denominador)
    probabilidad = (((feromona[rutasP[:]]**alpha)*(visibilidad[rutasP[:]]**beta)) / denominador)

    [antTox,antToy] = np.where(probabilidad  == max(probabilidad))
    anterior = inicio
    inicio = int(conexiones[np.array(rutasP).T[antTox[0]],1])
    rutaFinal.append([anterior,inicio])





























