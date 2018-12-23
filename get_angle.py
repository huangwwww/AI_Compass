import numpy as np
import pandas as pd
import math

data=pd.read_csv('Testresult2.csv')

x1=data["x1"]
y1=data["y1"]
x2=data["x3"]
y2=data["y3"]
Name=data["Name"]
ang=list()
distance=list()
count=0
count2=0
for i in range(135):
    angle=(x1[i]*x2[i]+y1[i]*y2[i])/(math.sqrt(math.pow(x1[i], 2)+math.pow(y1[i], 2))*math.sqrt(math.pow(x2[i], 2)+math.pow(y2[i], 2)))
    angle=math.acos(angle)
    angle=math.degrees(angle)
    ang.append(angle)
    disx=x2[i]-x1[i]
    disy=y2[i]-y1[i]
    dis=math.sqrt(math.pow(disx, 2)+math.pow(disy, 2))
    distance.append(dis)
    if angle<=45:
        count+=1
    if dis<=500:
        count2+=1

ang = pd.DataFrame(
    {'img': Name,
     'angle': ang,
     'distance':distance
    })

ang.to_csv('angle_error.csv')
print('angle :',count/135)
print('distance :',count2/135)
