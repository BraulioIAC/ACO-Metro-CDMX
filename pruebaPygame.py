import pygame
import math
import sys

pygame.init()
clock = pygame.time.Clock()

black = (0,0,0)
white = (255,255,255)

windowsWidth = 1400
windowHeight = 900
window = pygame.display.set_mode((windowsWidth,windowHeight))
pygame.display.set_caption('ACO Metro')

def CrearNodos(x1,y1,x2,y2):
    pygame.draw.line(window, black, (x1,y1), (x2,y2), 5)
    
    return

state = True
while state:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            state = False        
    window.fill(white)
    startX = 0
    startY = 0
    endX = 500
    endY = 500
    CrearNodos(startX,startY,endX,endY)
    startX = 500
    startY = 300
    endX = 1000
    endY = 1000
    CrearNodos(startX,startY,endX,endY)

    
    pygame.display.update()
    clock.tick(30)
    
pygame.quit()