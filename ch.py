session_path={1:'C:\ProyectoInicial\Datos\Kilosort\Thy7\\2020-11-11_16-05-00',  # Val
        0:'C:\ProyectoInicial\Datos\Kilosort\Dlx1\\2021-02-12_12-46-54',  # Val
        2:'C:\ProyectoInicial\Datos\Kilosort\PV6\\2021-04-19_14-02-31',      #Val
        3:'C:\ProyectoInicial\Datos\Kilosort\PV7xChR2\\2021-05-18_13-24-33', #Val
        4:'C:\ProyectoInicial\Datos\Kilosort\Thy9\\2021-03-16_12-10-32',     #Val
        5:'C:\ProyectoInicial\Datos\Kilosort\Thy1GCam1\\2020-12-18_14-40-16',     #Val
        }
f=open('prueba.txt','w')
f.write(session_path[5][0:-20])
aux=[19.025,19.5]

f.write('\n')
print(str(aux)[1:-2])
f.close()
