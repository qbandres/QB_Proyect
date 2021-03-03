from tkinter import *
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkcalendar import *
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import fsolve
from plotly.subplots import make_subplots
from datetime import date, timedelta
import xlwings as xw
import math

# Definir Colores
d_color = {'fondo': '#BDBDBD', 'boton': 'gray', 'framew': 'gray60', 'letra': '#BDBDBD'}
# DEFINIR PONDERACIONES
d_pon = {'TR': 0.05, 'PA': 0.1, 'MO': 0.45, 'NI': 0.2, 'PI': 0.1, 'PU': 0.1}


class Widget:
    def __init__(self, fram, back, ancho, altura, pox, poy):
        self.fram = fram
        self.back = back
        self.pox = pox
        self.poy = poy
        self.altura = altura
        self.ancho = ancho

    def boton(self, name, action):
        Button(self.fram, text=name, bg=self.back, width=self.ancho, height=self.altura, command=action).place(
            x=self.pox, y=self.poy)

    def marco(self):
        Frame(self.fram, bg=self.back, width=self.ancho, height=self.altura, relief='sunken', bd=2).place(
            x=self.pox, y=self.poy)

    def letra(self, name):
        Label(self.fram, text=name, bg=self.back, padx=self.ancho, pady=self.altura).place(x=self.pox,y=self.poy)

class Equation:
    def __init__(self, Wt, t):
        self.Wt = Wt
        self.t = t

    def calc1(self):
        def equ(pa):
            a, b, c = pa
            return (a*self.t-b, -2 * a * self.t ** 3 + 3 * b * self.t ** 2 + 6 * c * self.t - 6 * self.Wt,
                    -a * self.t ** 2 + 2 * b * self.t - c)

        a, b, c = fsolve(equ, (1, 1, 1))
        x = np.arange(0, self.t)
        y = -a * x ** 2 + b * x + c
        return y

    def calc2(self):
        n = np.ones(self.t)
        y=n*self.Wt/self.t
        return y

class splitdate:
    def __init__(self, fi, ff, Wt, pwa, pwb, pwc, pta, ptb, ptc):
        self.Wt = Wt
        self.pwa = pwa
        self.pwb = pwb
        self.pwc = pwc
        self.pta = pta
        self.ptb = ptb
        self.ptc = ptc
        self.ft = [fi + timedelta(days=d) for d in range((ff - fi).days + 1)] #CREAMOS LA LISTA DE FECHAS
        self.dft = pd.DataFrame({'Fecha': self.ft})
        self.dft.set_index('Fecha', inplace=True)
        self.t=(ff - fi).days + 1
    def calc1(self):
        #particionar las fechas de ejecución de los quiebres
        fta = self.ft[:int(self.t * self.pta)]                                          #Fecha de Montaje
        ftb = self.ft[int(self.t * (1-self.ptc-self.ptb)):int(self.t * (1-self.ptc))]   #Fecha de Torque
        ftc = self.ft[int(self.t * (1-self.ptc)):]                   #se extrae la parte de la lista de pucnh
        ta = len(fta)
        tb = len(ftb)
        tc = len(ftc)

        # Distribucion de pesos
        Wa = self.Wt * self.pwa
        Wb = self.Wt * self.pwb
        Wc = self.Wt * self.pwc

        return fta, ftb, ftc, Wa, Wb, Wc, ta, tb, tc


class convdf:                               #convertir en dataframe
    def __init__(self, ft, y, name):
        self.ft = ft
        self.y = y
        self.name = name

    def act(self):
        df = pd.DataFrame({'Fecha': self.ft, self.name: self.y})
        df.set_index('Fecha', inplace=True)
        return df

class Calular:
    def __init__(self, W, fi, ff):
        self.W = W
        self.fi = fi
        self.ff = ff
        self.t=(ff - fi).days + 1

    def met1(self):                     # Calculo de las sub-Equation
        fta1, ftb1, ftc1, Wa1, Wb1, Wc1, ta1, tb1, tc1 = splitdate(self.fi, self.ff, self.W, 0.6, 0.3, 0.1, 0.58, 0.27, 0.15).calc1()
        ya1 = Equation(Wa1, ta1).calc1()
        yb1 = Equation(Wb1, tb1).calc1()
        yc1 = Equation(Wc1, tc1).calc2()


        #Se calcula la variación de error por cada quiebre
        var_a1 = (Wa1 - ya1.sum()) / len(ya1)
        self.ya1 = ya1 + var_a1
        var_b1 = (Wb1 - yb1.sum()) / len(yb1)
        self.yb1 = yb1 + var_b1
        var_c1 = (Wc1 - yc1.sum()) / len(yc1)
        self.yc1 = yc1 + var_c1


        dfa1 = convdf(fta1, self.ya1, 'Montaje').act()
        dfb1 = convdf(ftb1, self.yb1, 'Torque').act()
        dfc1 = convdf(ftc1, self.yc1, 'Punch').act()

        datos_f1 = pd.concat([dfa1, dfb1, dfc1], axis=1)
        datos_f1 = datos_f1.fillna(0)

        return datos_f1

class Semana:                                   #CREAR DATA FRAME CON LA SEMANA
    def __init__(self,fi,T):
        self.fi=fi
        self.T=T

    def split(self):
        s = pd.date_range(start=self.fi, periods=self.T, freq='D')
        Nsemana = pd.DataFrame(s, columns=['Fecha'])
        Nsemana['SEMANA'] = Nsemana.index
        Nsemana["Fecha"] = pd.to_datetime(Nsemana.Fecha).dt.date
        Nsemana.set_index('Fecha', inplace=True)
        Nsemana['Semana'] = Nsemana.SEMANA // 7 + 1
        del Nsemana['SEMANA']
        return Nsemana

def importar():

    global df, dfv, df_dtr, df_dpa, df_dmo, df_dni, df_dpi, df_dpu, fusion

    import_file_path = filedialog.askopenfilename()
    df = pd.read_excel(import_file_path, sheet_name='Data')

    df = df[['ID', 'ESP', 'Barcode', 'WEIGHT', 'Ratio', 'TRASLADO', 'PREARMADO', 'MONTAJE',
             'NIVELACION', 'TOUCHUP', 'PUNCHLIST', 'PROTOCOLO','FASE']]

    df.rename(columns={'TRASLADO': 'DTR', 'PREARMADO': 'DPA',
                       'MONTAJE': 'DMO', 'NIVELACION': 'DNI', 'TOUCHUP': 'DPI',
                       'PUNCHLIST': 'DPU'},
              inplace=True)

    df = df[df.Ratio.notnull()]  # LIMPIAMOS DATOS QUE ESTEN NULOS EN EL RATIO
    df = df[df.WEIGHT.notnull()]  # LIMPIAMOS LOS DATOS QUE ESTEN NULOS EN EL PESO

    df['DTR'] = pd.to_datetime(df['DTR'])
    df['DPA'] = pd.to_datetime(df['DPA'])
    df['DMO'] = pd.to_datetime(df['DMO'])
    df['DNI'] = pd.to_datetime(df['DNI'])
    df['DPI'] = pd.to_datetime(df['DPI'])
    df['DPU'] = pd.to_datetime(df['DPU'])

#CALCULO DE PESOS TOTALES DE SEGUN PONDERACION

    df['TOTAL_WTR'] = df.WEIGHT * d_pon['TR']
    df['TOTAL_WPA'] = df.WEIGHT * d_pon['PA']
    df['TOTAL_WMO'] = df.WEIGHT * d_pon['MO']
    df['TOTAL_WNI'] = df.WEIGHT * d_pon['NI']
    df['TOTAL_WPI'] = df.WEIGHT * d_pon['PI']
    df['TOTAL_WPU'] = df.WEIGHT * d_pon['PU']

# CALCULO DE HH EARNED TOTALES SEGUN MODERATION


    df['TOTAL_ETR'] = df.WEIGHT * d_pon['TR'] * df.Ratio / 1000
    df['TOTAL_EPA'] = df.WEIGHT * d_pon['PA'] * df.Ratio / 1000
    df['TOTAL_EMO'] = df.WEIGHT * d_pon['MO'] * df.Ratio / 1000
    df['TOTAL_ENI'] = df.WEIGHT * d_pon['NI'] * df.Ratio / 1000
    df['TOTAL_EPI'] = df.WEIGHT * d_pon['PI'] * df.Ratio / 1000
    df['TOTAL_EPU'] = df.WEIGHT * d_pon['PU'] * df.Ratio / 1000

    # CALCULO DE PESO SEGUN AVANCE
    df['WTR'] = np.where(df['DTR'].isnull(), 0, df.WEIGHT * d_pon['TR'])
    df['WPA'] = np.where(df['DPA'].isnull(), 0, df.WEIGHT * d_pon['PA'])
    df['WMO'] = np.where(df['DMO'].isnull(), 0, df.WEIGHT * d_pon['MO'])
    df['WNI'] = np.where(df['DNI'].isnull(), 0, df.WEIGHT * d_pon['NI'])
    df['WPI'] = np.where(df['DPI'].isnull(), 0, df.WEIGHT * d_pon['PI'])
    df['WPU'] = np.where(df['DPU'].isnull(), 0, df.WEIGHT * d_pon['PU'])

    # CALCULO DE HH EARNED SEGUN AVANCE
    df['ETR'] = np.where(df['DTR'].isnull(), 0, df.WEIGHT * d_pon['TR'] * df.Ratio / 1000)
    df['EPA'] = np.where(df['DPA'].isnull(), 0, df.WEIGHT * d_pon['PA'] * df.Ratio / 1000)
    df['EMO'] = np.where(df['DMO'].isnull(), 0, df.WEIGHT * d_pon['MO'] * df.Ratio / 1000)
    df['ENI'] = np.where(df['DNI'].isnull(), 0, df.WEIGHT * d_pon['NI'] * df.Ratio / 1000)
    df['EPI'] = np.where(df['DPI'].isnull(), 0, df.WEIGHT * d_pon['PI'] * df.Ratio / 1000)
    df['EPU'] = np.where(df['DPU'].isnull(), 0, df.WEIGHT * d_pon['PU'] * df.Ratio / 1000)

    df['WBRUTO'] = np.where(df['DMO'].isnull(), 0, df.WEIGHT)
    df['WPOND'] = df.WTR + df.WPA + df.WMO + df.WNI + df.WPI + df.WPU

##########################SEPARAMOS LOS PESOS POR AVANCE DE CADA ETAPA
    df_dtr = df[["ESP", "ID", 'Barcode', "WEIGHT", "Ratio", "DTR", "WTR", "ETR"]]
    df_dtr = df_dtr.dropna(subset=['DTR'])  # Elimina llas filas vacias de DTR
    df_dtr["Etapa"] = "1-Traslado"
    df_dtr = df_dtr.rename(columns={'WTR': 'WPOND', "DTR": 'Fecha', 'ETR': 'HGan'})

    df_dpa = df[["ESP", "ID", 'Barcode', "WEIGHT", "Ratio", "DPA", "WPA", "EPA"]]
    df_dpa = df_dpa.dropna(subset=['DPA'])
    df_dpa["Etapa"] = "2-Ensamble"
    df_dpa = df_dpa.rename(columns={'WPA': 'WPOND', "DPA": 'Fecha', 'EPA': 'HGan'})

    df_dmo = df[["ESP", "ID", 'Barcode', "WEIGHT", "Ratio", "DMO", "WMO", "EMO"]]
    df_dmo = df_dmo.dropna(subset=['DMO'])
    df_dmo["Etapa"] = "3-Montaje"
    df_dmo = df_dmo.rename(columns={'WMO': 'WPOND', "DMO": 'Fecha', 'EMO': 'HGan'})

    df_dni = df[["ESP", "ID", 'Barcode', "WEIGHT", "Ratio", "DNI", "WNI", "ENI"]]
    df_dni = df_dni.dropna(subset=['DNI'])
    df_dni["Etapa"] = "4-Alineamiento"
    df_dni = df_dni.rename(columns={'WNI': 'WPOND', "DNI": 'Fecha', 'ENI': 'HGan'})

    df_dpi = df[["ESP", "ID", 'Barcode', "WEIGHT", "Ratio", "DPI", "WPI", "EPI"]]
    df_dpi = df_dpi.dropna(subset=['DPI'])
    df_dpi["Etapa"] = "5-Touch Up"
    df_dpi = df_dpi.rename(columns={'WPI': 'WPOND', "DPI": 'Fecha', 'EPI': 'HGan'})

    df_dpu = df[["ESP", "ID", 'Barcode', "WEIGHT", "Ratio", "DPU", "WPU", "EPU"]]
    df_dpu = df_dpu.dropna(subset=['DPU'])
    df_dpu["Etapa"] = "6-Punch List"
    df_dpu = df_dpu.rename(columns={'WPU': 'WPOND', "DPU": 'Fecha', 'EPU': 'HGan'})

    ##CONCATENAR VERTICAL DE LAS COLUMNAS DE RESUMEN

    dfv = pd.concat(
        [df_dtr.round(1), df_dpa.round(1), df_dmo.round(1), df_dni.round(1), df_dpi.round(1), df_dpu.round(1)], axis=0)

    dfv['WBRUTO'] = np.where(dfv.Etapa != '3-Montaje', 0, dfv.WEIGHT)

    np_array = dfv.to_numpy()

    dfv = pd.DataFrame(data=np_array,
                       columns=['ESP', 'ID', 'Barcode', 'WEIGHT', 'Ratio', 'Fecha', 'WPOND', 'HGan', 'Etapa',
                                   'WBRUTO'])
    dfv["Fecha"] = pd.to_datetime(dfv.Fecha).dt.date


    #########################infromacion general########################                                                INFORMACION GENERAL
    sum_proy = round(df["WEIGHT"].sum() / 1000, 0)
    wpond_proy = round((df["WTR"].sum() + df["WPA"].sum() + df["WMO"].sum() + df["WNI"].sum() + df["WPI"].sum() + df[
        "WPU"].sum()) / 1000, 0)
    wbrut_proy = round((df["WBRUTO"].sum()) / 1000, 0)
    porcwpon_proy = round(wpond_proy / sum_proy * 100, 2)
    porcbbrut_proy = round(wbrut_proy / sum_proy * 100, 2)

    Label(General, text=sum_proy, bg=d_color['fondo']).place(x=55, y=42)
    Label(General, text=wpond_proy, bg=d_color['fondo']).place(x=55, y=64)
    Label(General, text=wbrut_proy, bg=d_color['fondo']).place(x=55, y=86)
    Label(General, text=porcwpon_proy, bg=d_color['fondo']).place(x=190, y=64)
    Label(General, text=porcbbrut_proy, bg=d_color['fondo']).place(x=190, y=86)

    #####################################################################################

    fusion = pd.read_excel(import_file_path, sheet_name='Input')
    fusion=fusion[['Code','Weight_OT','Fi','Ff','Weight_Tk']]

    fusion["Fi"] = pd.to_datetime(fusion.Fi).dt.date                    #Se eliminan las horas a la fecha
    fusion["Ff"] = pd.to_datetime(fusion.Ff).dt.date                    #Se eliminan las horas a la fecha

    tree = ttk.Treeview(Regre)
    tree['column'] = list(fusion.columns)
    tree['show'] = 'headings'
    # loop trhu column
    for column in tree['column']:
        tree.heading(column, text=column)

    df_rows = fusion.to_numpy().tolist()
    for row in df_rows:
        tree.insert("", "end", values=row)
    tree.place(x=5, y=30)

    tree.column("#1", width=89, minwidth=89, stretch=tk.NO)
    tree.column("#2", width=65, minwidth=65, stretch=tk.NO)
    tree.column("#3", width=68, minwidth=68, stretch=tk.NO)
    tree.column("#4", width=79, minwidth=79, stretch=tk.NO)
    tree.column("#5", width=79, minwidth=79, stretch=tk.NO)





################################################################DETALLE##############################################


def pbi1():  # FUNCION EXPORTAR PARA PBI
    global df, dfv

    dfpbi = df[
        ['ID', 'ESP', 'Barcode', 'WEIGHT', 'DTR', 'DPA', 'DMO', 'DNI', 'DPI', 'DPU', 'WTR', 'WPA', 'WMO', 'WNI', 'WPI',
         'WPU']]

    export_file = filedialog.askdirectory()
    dfpbi.to_csv(export_file + '/Matriz.csv', index=False)
    dfv.to_csv(export_file + '/QB_PBI.csv', header=True, index=False)


#VENTANA DETALLE

def filtrar():
    global animatrix2, info1, info2, info3, Nsemana, df

    efechai = pd.to_datetime(fechai.get())
    efechaf = pd.to_datetime(fechaf.get())

    #FILTRAMOS LA INFORMACION GENERAL DFV

    animat = dfv

    animat['filtro'] = np.where((qui1.get() == 1) & (animat['Etapa'] == "1-Traslado"), "positivo",'')  # APLICA FILTRO DE QUIEBRE
    filtro1 = animat[animat['filtro'] == 'positivo']

    animat['filtro'] = np.where((qui2.get() == 1) & (animat['Etapa'] == "2-Ensamble"), "positivo", '')
    filtro2 = animat[animat['filtro'] == 'positivo']

    animat['filtro'] = np.where((qui3.get() == 1) & (animat['Etapa'] == "3-Montaje"), "positivo", '')
    filtro3 = animat[animat['filtro'] == 'positivo']

    animat['filtro'] = np.where((qui4.get() == 1) & (animat['Etapa'] == "4-Alineamiento"), "positivo", '')
    filtro4 = animat[animat['filtro'] == 'positivo']

    animat['filtro'] = np.where((qui5.get() == 1) & (animat['Etapa'] == "5-Touch Up"), "positivo", '')
    filtro5 = animat[animat['filtro'] == 'positivo']

    animat['filtro'] = np.where((qui6.get() == 1) & (animat['Etapa'] == "6-Punch List"), "positivo", '')
    filtro6 = animat[animat['filtro'] == 'positivo']

    animatrix2 = pd.concat([filtro1, filtro2, filtro3, filtro4, filtro5, filtro6], axis=0)

    animatrix2 = animatrix2[
        (animatrix2['Fecha'] >= efechai) & (animatrix2['Fecha'] <= efechaf)]  # SELECCION MONTAJE DIARIO

    temp_info1 = animatrix2[
        ['Fecha', 'WPOND', 'WBRUTO']]  # ANIMATRIX EXTRAE DE ANIMATRIX2 SOLO FECHA, WPOND Y WBRUTO FILTRADO
    info1 = temp_info1.groupby(['Fecha']).sum() / 1000

    info1['WPACUM'] = info1['WPOND'].cumsum()
    info1['WBACUM'] = info1['WBRUTO'].cumsum()
    info1.reset_index(inplace=True)

    info1 = info1.round(2)
    info1.fillna(0, inplace=True)

    tree = ttk.Treeview(General)
    tree['column'] = list(info1.columns)
    tree['show'] = 'headings'
    # loop trhu column
    for column in tree['column']:
        tree.heading(column, text=column)

    df_rows = info1.to_numpy().tolist()
    for row in df_rows:
        tree.insert("", "end", values=row)
    tree.place(x=7, y=162)

    tree.column("#1", width=89, minwidth=89, stretch=tk.NO)
    tree.column("#2", width=65, minwidth=65, stretch=tk.NO)
    tree.column("#3", width=68, minwidth=68, stretch=tk.NO)
    tree.column("#4", width=79, minwidth=79, stretch=tk.NO)
    tree.column("#5", width=79, minwidth=79, stretch=tk.NO)
    ######################################frame 3#################                                                      #SELECCION POR ESP

    temp_info2 = df[['ESP', 'WEIGHT', 'WPOND', 'WBRUTO']]

    info2 = temp_info2.groupby(['ESP']).sum() / 1000

    info2.reset_index(inplace=True)

    info2.rename(columns={'WEIGHT': 'Total'}, inplace=True)

    info2['Pond%'] = info2.WPOND / info2.Total * 100
    info2['Bruto%'] = info2.WBRUTO / info2.Total * 100

    info2 = info2.round(2)
    info2.fillna(0, inplace=True)

    tree2 = ttk.Treeview(General)
    tree2['column'] = list(info2.columns)
    tree2['show'] = 'headings'
    # loop trhu column
    for column in tree2['column']:
        tree2.heading(column, text=column)

    df_rows = info2.to_numpy().tolist()
    for row in df_rows:
        tree2.insert("", "end", values=row)
    tree2.place(x=7, y=440)

    tree2.column("#1", width=75, minwidth=75, stretch=tk.NO)
    tree2.column("#2", width=55, minwidth=55, stretch=tk.NO)
    tree2.column("#3", width=55, minwidth=55, stretch=tk.NO)
    tree2.column("#4", width=55, minwidth=55, stretch=tk.NO)
    tree2.column("#5", width=70, minwidth=70, stretch=tk.NO)
    tree2.column("#6", width=70, minwidth=70, stretch=tk.NO)

    ##label resultados
    Label(General, text=round(info1['WPOND'].sum(), 2), bg=d_color['fondo']).place(x=87, y=392)
    Label(General, text=round(info1['WBRUTO'].sum(), 2), bg=d_color['fondo']).place(x=155, y=392)

    Label(General, text=round(info2['Total'].sum(), 2), bg=d_color['fondo']).place(x=80, y=670)
    Label(General, text=round(info2['WPOND'].sum(), 2), bg=d_color['fondo']).place(x=138, y=670)
    Label(General, text=round(info2['WBRUTO'].sum(), 2), bg=d_color['fondo']).place(x=192, y=670)
    #####################SEMANA

    #####CREAR DF NRO SEMANA#######

    Nsemana=Semana('2019-04-12',1200).split()               #invocamos a la clase semana

    temp_info3 = info1[['Fecha', 'WPOND', 'WBRUTO']]
    temp_info3.set_index('Fecha', inplace=True)

    filtro_sem = pd.concat([Nsemana, temp_info3], axis=1)  # Juntamos la nueva matriz con la del montaje diario


    info3 = filtro_sem.groupby(['Semana']).sum()  # Agrupamod por Semana
    info3 = info3[(info3['WPOND'] != 0) | (info3['WBRUTO'] != 0)]  # Limpiamos la matriz de los ceros


    info3['WPACUM'] = info3['WPOND'].cumsum()
    info3['WBACUM'] = info3['WBRUTO'].cumsum()
    info3.reset_index(inplace=True)


    info3 = info3.round(2)
    info3.fillna(0, inplace=True)


    #CREAMOS EL LISTADO FILTRADO POR SEMANA

    tree3 = ttk.Treeview(General)
    tree3['column'] = list(info3.columns)
    tree3['show'] = 'headings'
    # loop trhu column
    for column in tree3['column']:
        tree3.heading(column, text=column)

    df_rows3 = info3.to_numpy().tolist()
    for row in df_rows3:
        tree3.insert("", "end", values=row)
    tree3.place(x=424, y=162)

    tree3.column("#1", width=75, minwidth=75, stretch=tk.NO)
    tree3.column("#2", width=70, minwidth=70, stretch=tk.NO)
    tree3.column("#3", width=70, minwidth=70, stretch=tk.NO)
    tree3.column("#4", width=79, minwidth=79, stretch=tk.NO)
    tree3.column("#5", width=79, minwidth=79, stretch=tk.NO)


    temp_info4 = df[['FASE', 'WEIGHT', 'WPOND', 'WBRUTO']]

    info4 = temp_info4.groupby(['FASE']).sum() / 1000

    info4.reset_index(inplace=True)

    info4.rename(columns={'WEIGHT': 'Total'}, inplace=True)

    info4['Pond%'] = info2.WPOND / info2.Total * 100
    info4['Bruto%'] = info2.WBRUTO / info2.Total * 100

    info4 = info4.round(2)
    info4.fillna(0, inplace=True)

    tree4 = ttk.Treeview(General)
    tree4['column'] = list(info4.columns)
    tree4['show'] = 'headings'
    # loop trhu column
    for column in tree4['column']:
        tree4.heading(column, text=column)

    df_rows = info4.to_numpy().tolist()
    for row in df_rows:
        tree4.insert("", "end", values=row)
    tree4.place(x=423, y=440)

    tree4.column("#1", width=75, minwidth=75, stretch=tk.NO)
    tree4.column("#2", width=55, minwidth=55, stretch=tk.NO)
    tree4.column("#3", width=55, minwidth=55, stretch=tk.NO)
    tree4.column("#4", width=55, minwidth=55, stretch=tk.NO)
    tree4.column("#5", width=70, minwidth=70, stretch=tk.NO)
    tree4.column("#6", width=70, minwidth=70, stretch=tk.NO)



    #CREANDO LOS SCROLL DESLZADORES

    desl1 = ttk.Scrollbar(General, orient="vertical", command=tree.yview)
    desl1.place(x=389, y=163, height=225)
    tree.configure(yscrollcommand=desl1.set)

    desl2 = ttk.Scrollbar(General, orient="vertical", command=tree2.yview)
    desl2.place(x=381, y=441, height=225)
    tree2.configure(yscrollcommand=desl2.set)

    desl3 = ttk.Scrollbar(General, orient="vertical", command=tree3.yview)
    desl3.place(x=799, y=163, height=225)
    tree3.configure(yscrollcommand=desl3.set)

    desl4 = ttk.Scrollbar(General, orient="vertical", command=tree4.yview)
    desl4.place(x=800, y=441, height=225)
    tree4.configure(yscrollcommand=desl4.set)



def reporte():  # FUNCION EXPORTAR PARA PBI
    global animatrix2, info1, info2, info3, Nsemana

    yminfo1 = info1[['Fecha', 'WPOND', 'WBRUTO']]
    yminfo1['Year_month'] = pd.to_datetime(yminfo1['Fecha']).dt.to_period('M')  # get month an year

    filtro_mes = yminfo1.groupby(['Year_month']).sum()
    filtro_mes.reset_index(inplace=True)

    export_file = filedialog.askdirectory()  # Buscamos el directorio para gruafar

    # Creamos una excel y le indicamos la ruta
    writer = pd.ExcelWriter(export_file + '/' + 'Reporte.xlsx')

    # Write each dataframe to a different worksheet.
    animatrix2.to_excel(writer, sheet_name='General', index=True)
    info1.to_excel(writer, sheet_name='Dia', index=False)
    filtro_mes.to_excel(writer, sheet_name='Mes', index=False)
    info3.to_excel(writer, sheet_name='Semana')
    info2.to_excel(writer, sheet_name='ESP')

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()


def pintadog2():  # FUNCION EXPORTAR PARA PINTAR
    global animatrix2

    pintado = animatrix2[['ID', 'Etapa']]

    pint_dtr = pintado[pintado.Etapa == '1-Traslado']
    pint_dtr['USER_FIELD_1'] = 'dtr_' + euser.get()
    del pint_dtr['Etapa']

    pint_dpa = pintado[pintado.Etapa == '2-Ensamble']
    pint_dpa['USER_FIELD_1'] = 'dpa_' + euser.get()
    del pint_dpa['Etapa']

    pint_dmo = pintado[pintado.Etapa == '3-Montaje']
    pint_dmo['USER_FIELD_1'] = 'dmo_' + euser.get()
    del pint_dmo['Etapa']

    pint_dni = pintado[pintado.Etapa == '4-Alineamiento']
    pint_dni['USER_FIELD_1'] = 'dtr_' + euser.get()
    del pint_dni['Etapa']

    pint_dpi = pintado[pintado.Etapa == '5-Touch Up']
    pint_dpi['USER_FIELD_1'] = 'dpi_' + euser.get()
    del pint_dpi['Etapa']

    pint_dpu = pintado[pintado.Etapa == '6-Punch List']
    pint_dpu['USER_FIELD_1'] = 'dpu_' + euser.get()
    del pint_dpu['Etapa']

    export_file = filedialog.askdirectory()

    # Exportar para tekla
    pint_dtr.to_csv(export_file + '/dtr_tekla.csv', index=False)
    pint_dpa.to_csv(export_file + '/dpa_tekla.csv', index=False)
    pint_dmo.to_csv(export_file + '/dmo_tekla.csv', index=False)
    pint_dni.to_csv(export_file + '/dni_tekla.csv', index=False)
    pint_dpi.to_csv(export_file + '/dpi_tekla.csv', index=False)
    pint_dpu.to_csv(export_file + '/dpu_tekla.csv', index=False)


def graficos():
    global animatrix2, info1, info2, info3

    if combo.get()=="Montaje Diario":

        x = info1.Fecha
        fig = go.Figure(go.Bar(x=x, y=info1.WPOND, name='WPonderad'))
        fig.add_trace(go.Bar(x=x, y=info1.WBRUTO, name='WBruto'))

        fig.update_layout(barmode='stack',
                          font_color="black",  title="MONTAJE DIARIO",
                         xaxis_title="Fecha",
                         yaxis_title="Peso",
                         legend_title="Caracteristica",
                          font=dict(
                              family="Courier New, monospace",
                              size=18,
                              color="RebeccaPurple"
                          ))
        fig.update_xaxes(categoryorder='total ascending'
                         )
        fig.show()


    elif combo.get()=="Montaje Semanal":

        x = info3.Semana
        fig = go.Figure(go.Bar(x=x, y=info3.WPOND, name='WPonderado'))
        fig.add_trace(go.Bar(x=x, y=info3.WBRUTO, name='WBruto'))

        fig.update_layout(barmode='stack',
                          font_color="black",  title="MONTAJE SEMANAL",
                         xaxis_title="Fecha",
                         yaxis_title="Peso",
                         legend_title="Caracteristica",
                          font=dict(
                              family="Courier New, monospace",
                              size=18,
                              color="RebeccaPurple"
                          ))
        fig.update_xaxes(categoryorder='total ascending'
                         )
        fig.show()


    elif combo.get()=='Montaje Diario Acumulado':
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=info1.Fecha, y=info1.WPACUM,
                                 mode='lines+markers',
                                 name='lines+markers'))
        fig.add_trace(go.Scatter(x=info1.Fecha, y=info1.WBACUM,
                                 mode='lines+markers',
                                 name='lines+markers'))
        fig.show()

    elif combo.get() == "Montaje por ESP":
        fig = go.Figure(data=[
            go.Bar(name='Montaje Ponderado', x=info2.ESP, y=info2.WPOND),
            go.Bar(name='Montaje Bruto', x=info2.ESP, y=info2.WBRUTO),
            go.Bar(name='Total', x=info2.ESP, y=info2.Total)
        ])
        # Change the bar mode
        fig.update_layout(barmode='group', xaxis_tickangle=-45,
                          font=dict(
                              family="Courier New, monospace",
                              size=18,
                              color="RebeccaPurple"
                          ))
        fig.show()

def proyeccion():
    global fusion, df_total
    k = 0
    for i in fusion.Fi:
        dataframe = Calular(fusion.loc[k, 'Weight_OT'], fusion.loc[k, 'Fi'], fusion.loc[k, 'Ff']).met1()
        k = k + 1
        list_fase.append(k)
        globals()["df_fase" + str(k)] = dataframe

    combo2["values"] = list_fase
    combo2.current(0)

    #Se crea listado de fechas para index

    fi = date(2019, 10, 10)
    ff = date(2022, 4, 15)
    t = (ff - fi).days

    ft = [fi + timedelta(days=d) for d in range((ff - fi).days + 1)]  # CREAMOS LA LISTA DE FECHAS
    dft = pd.DataFrame({'Fecha': ft})
    dft.set_index('Fecha', inplace=True)

    df_total = pd.concat(
        [dft, df_fase1, df_fase2, df_fase3, df_fase4, df_fase5, df_fase6, df_fase7, df_fase8, df_fase9, df_fase10,
         df_fase11, df_fase12,
         df_fase13, df_fase14, df_fase15, df_fase16, df_fase17, df_fase18, df_fase19, df_fase20, df_fase21, df_fase22,
         df_fase23, df_fase24,
         df_fase25, df_fase26, df_fase27, df_fase28, df_fase29, df_fase30, df_fase31, df_fase32, df_fase33, df_fase34,
         df_fase35, df_fase36, df_fase37,
         df_fase38, df_fase39, df_fase40, df_fase41, df_fase42, df_fase43, df_fase44, df_fase45, df_fase46, df_fase47,
         df_fase48, df_fase49,
         df_fase50, df_fase51, df_fase52, df_fase53, df_fase54], axis=1)

    df_total = df_total.fillna(0)
    df_total["Total_1"] = df_total.sum(axis=1)
    df_total['Acum_1'] = df_total['Total_1'].cumsum()

    for i in list_fase:
        globals()["df_fase" + str(i)]['Total']=globals()["df_fase" + str(i)].sum(axis=1)
        globals()["df_fase" + str(i)]['Acum']=globals()["df_fase" + str(i)]['Total'].cumsum()


def Grafic_gen():

    fig = make_subplots(rows=2, cols=1)

    fig.append_trace(go.Scatter(
        x=df_total.index,
        y=df_total.Total_1,
        name='Montaje diario'
    ), row=1, col=1)

    fig.append_trace(go.Scatter(
        x=df_total.index,
        y=df_total.Acum_1,
        name='Montaje Acumulado'
    ), row=2, col=1)

    fig.show()

def Grafic_fase():

    x = globals()["df_fase" + str(combo2.get())].index
    y = globals()["df_fase" + str(combo2.get())].Total
    fig1 = go.Figure(data=go.Scatter(x=x, y=y))
    fig1.show()




# LECTURA Y CAMBIO DE NOMBRES A LAS COLUMNAS

root = Tk()
root.title('GESTION DE RESULTADOS PROYECTO QB2')
root.configure(bg="#BDBDBD")
root.geometry('840x765')  # Definir el tamaño de celda
root.resizable(width=0, height=0)

# Creando pestañas
nb = ttk.Notebook(root)
nb.pack(fill='both', expand='yes')  # expandir las pestañas
s = ttk.Style()
s.configure('TLabelframe', background='#BDBDBD')
General = ttk.Frame(nb, style='TLabelframe')
Regre = ttk.Frame(nb, style='TLabelframe')
nb.add(General, text='General')
nb.add(Regre, text='Proyección')

###Creando los frames
Widget(General, d_color['fondo'], 250, 116, 5, 10).marco()      #FRAME DE RESUMEN
Widget(General, d_color['fondo'], 395, 229, 5, 161).marco()     #FRAME RESUMEN POR FECHA
Widget(General, d_color['fondo'], 395, 229, 5, 439).marco()     #FRAME RESUMEN POR ESP
Widget(General, d_color['fondo'], 395, 229, 423, 161).marco()   #RESUMEN POR SEMANA
Widget(General, d_color['fondo'], 395, 229, 423, 439).marco()   #RESUMEN POR FASE

###Creando los label
Widget(General, d_color['fondo'], 1, 1, 7, 12).letra('Resumen General')
Widget(General, d_color['fondo'], 1, 1, 15, 42).letra('Total:')
Widget(General, d_color['fondo'], 1, 1, 15, 64).letra('Pond:')
Widget(General, d_color['fondo'], 1, 1, 15, 86).letra('Brut:')
Widget(General, d_color['fondo'], 1, 1, 130, 64).letra('Pond% :')
Widget(General, d_color['fondo'], 1, 1, 130, 86).letra('Brut% :')
Widget(General, d_color['fondo'], 1, 1, 0, 132).letra('Resumen Diario')
Widget(General, d_color['fondo'], 1, 1, 17, 392).letra('Total')
Widget(General, d_color['fondo'], 1, 1, 17, 670).letra('Total')
Widget(General, d_color['fondo'], 1, 1, 0, 410).letra('Resumen por ESP')
Widget(General, d_color['fondo'], 1, 1, 492, 410).letra('Resumen por Semana')
Widget(General, d_color['fondo'], 1, 1, 430, 5).letra('Inicio')
Widget(General, d_color['fondo'], 1, 1, 430, 40).letra('Final')

##Creando los botones
Widget(General, d_color['boton'], 10, 7, 260, 9).boton('Importar', importar)
Widget(General, d_color['boton'], 10, 7, 344, 9).boton('Filtrar', filtrar)
Widget(General, d_color['boton'], 13, 2, 92, 693).boton('Reporte', reporte)
Widget(General, d_color['boton'], 13, 2, 194, 693).boton('Power Bi', pbi1)
Widget(General, d_color['boton'], 13, 2, 329, 693).boton('Exp. Tekla', pintadog2)
Widget(General,d_color['boton'],13,2,574,693).boton('Gráficos',graficos)


##Creando los entry
def on_click(event):
    euser.config(state=NORMAL)
    euser.delete(0, END)


euser = Entry(General, width=10)
euser.insert(0, 'USER_FIELD')
euser.config(state=DISABLED)
euser.bind('<Button-1>', on_click)
euser.place(x=434, y=710)

##Creando CheckButton
qui1 = IntVar(value=1)
qui2 = IntVar(value=1)
qui3 = IntVar(value=1)
qui4 = IntVar(value=1)
qui5 = IntVar(value=1)
qui6 = IntVar(value=1)
chekqu1 = Checkbutton(General, text='Traslado', variable=qui1, bg=d_color['fondo'])
chekqu1.place(x=715, y=5)
chekqu2 = Checkbutton(General, text='Pre-Armado', variable=qui2, bg=d_color['fondo'])
chekqu2.place(x=715, y=30)
chekqu3 = Checkbutton(General, text='Montaje', variable=qui3, bg=d_color['fondo'])
chekqu3.place(x=715, y=55)
chekqu4 = Checkbutton(General, text='Nivelación', variable=qui4, bg=d_color['fondo'])
chekqu4.place(x=715, y=80)
chekqu5 = Checkbutton(General, text='Touch-Up', variable=qui5, bg=d_color['fondo'])
chekqu5.place(x=715, y=105)
chekqu6 = Checkbutton(General, text='Punch-List', variable=qui6, bg=d_color['fondo'])
chekqu6.place(x=715, y=130)


##Creando Combobox
combo = ttk.Combobox(General, state="readonly")
combo.place(x=676,y=712)
combo["values"] = ["Montaje Diario", "Montaje Semanal", "Montaje por ESP",'Montaje Diario Acumulado']
combo.current(0)


# creando entrada de Fechas
fechai = DateEntry(General, width=16, bg='blue', date_pattern='yyyy/MM/dd', year=2019, month=10, day=1)
fechai.place(x=480, y=5)
fechaf = DateEntry(General, width=16, bg='blue', date_pattern='yyyy/MM/dd', year=2021, month=2, day=28)
fechaf.place(x=480, y=40)




##############################################################################################################

Widget(Regre, d_color['fondo'], 395, 229, 5, 30).marco()     #FRAME RESUMEN POR FECHA

Widget(Regre, d_color['boton'], 10, 7, 460, 9).boton('Expandir', proyeccion)
Widget(Regre, d_color['boton'], 10, 7, 460, 240).boton('Grafico', Grafic_fase)
Widget(Regre, d_color['boton'], 10, 7, 460, 440).boton('Graph Gen', Grafic_gen)

##Creando Combobox
combo2 = ttk.Combobox(Regre, state="readonly")
combo2.place(x=676,y=712)
list_fase=[]
combo2["values"] = list_fase






root.mainloop()