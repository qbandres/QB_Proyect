from tkinter import *
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkcalendar import *
import numpy as np
import pandas as pd
import statsmodels.tsa.statespace._filters._conventional
import statsmodels.tsa.statespace._filters._univariate_diffuse
import statsmodels.tsa.statespace._filters._univariate
import statsmodels.tsa.statespace._filters._inversions
import statsmodels.tsa.statespace._smoothers
import statsmodels.tsa.statespace._smoothers._conventional
import statsmodels.tsa.statespace._smoothers._univariate
import statsmodels.tsa.statespace._smoothers._univariate_diffuse
import statsmodels.tsa.statespace._smoothers._classical
import statsmodels.tsa.statespace._smoothers._alternative
import sklearn.utils._weight_vector
import plotly.graph_objects as go
from scipy.optimize import fsolve
from plotly.subplots import make_subplots
from datetime import date, timedelta
from pmdarima.arima import auto_arima


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

        # Se calcula la variación de error por cada quiebre
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

    global df, dfv, df_dtr, df_dpa, df_dmo, df_dni, df_dpi, df_dpu, fusion, info_0, df_cp

    import_file_path = filedialog.askopenfilename()
    df = pd.read_excel(import_file_path, sheet_name='Data')

    df = df[['IDTekla', 'ESP', 'Barcode', 'PesoTotal(Kg)', 'Ratio', 'Traslado', 'Prearmado', 'Montaje',
             'Nivelacion,soldadura&Torque', 'Touchup', 'Punchlist','FASE']]

    df.rename(columns={'Traslado': 'DTR', 'Prearmado': 'DPA',
                       'Montaje': 'DMO', 'Nivelacion,soldadura&Torque': 'DNI', 'Touchup': 'DPI',
                       'Punchlist': 'DPU','IDTekla':'ID','PesoTotal(Kg)':'WEIGHT'},
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

    # CALCULO DE PESO BRUTO QUIEBRE AVANCE
    df['BWTR'] = np.where(df['DTR'].isnull(), 0, df.WEIGHT)
    df['BWPA'] = np.where(df['DPA'].isnull(), 0, df.WEIGHT)
    df['BWMO'] = np.where(df['DMO'].isnull(), 0, df.WEIGHT)
    df['BWNI'] = np.where(df['DNI'].isnull(), 0, df.WEIGHT)
    df['BWPI'] = np.where(df['DPI'].isnull(), 0, df.WEIGHT)
    df['BWPU'] = np.where(df['DPU'].isnull(), 0, df.WEIGHT)

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

    df_dtr = df[["ESP", "ID", 'Barcode', "WEIGHT", "Ratio", "DTR", "WTR", "ETR",'FASE']]
    df_dtr = df_dtr.dropna(subset=['DTR'])  # Elimina llas filas vacias de DTR
    df_dtr["Etapa"] = "1-Traslado"
    df_dtr = df_dtr.rename(columns={'WTR': 'WPOND', "DTR": 'Fecha', 'ETR': 'HGan'})

    df_dpa = df[["ESP", "ID", 'Barcode', "WEIGHT", "Ratio", "DPA", "WPA", "EPA",'FASE']]
    df_dpa = df_dpa.dropna(subset=['DPA'])
    df_dpa["Etapa"] = "2-Ensamble"
    df_dpa = df_dpa.rename(columns={'WPA': 'WPOND', "DPA": 'Fecha', 'EPA': 'HGan'})

    df_dmo = df[["ESP", "ID", 'Barcode', "WEIGHT", "Ratio", "DMO", "WMO", "EMO",'FASE']]
    df_dmo = df_dmo.dropna(subset=['DMO'])
    df_dmo["Etapa"] = "3-Montaje"
    df_dmo = df_dmo.rename(columns={'WMO': 'WPOND', "DMO": 'Fecha', 'EMO': 'HGan'})

    df_dni = df[["ESP", "ID", 'Barcode', "WEIGHT", "Ratio", "DNI", "WNI", "ENI",'FASE']]
    df_dni = df_dni.dropna(subset=['DNI'])
    df_dni["Etapa"] = "4-Alineamiento"
    df_dni = df_dni.rename(columns={'WNI': 'WPOND', "DNI": 'Fecha', 'ENI': 'HGan'})

    df_dpi = df[["ESP", "ID", 'Barcode', "WEIGHT", "Ratio", "DPI", "WPI", "EPI",'FASE']]
    df_dpi = df_dpi.dropna(subset=['DPI'])
    df_dpi["Etapa"] = "5-Touch Up"
    df_dpi = df_dpi.rename(columns={'WPI': 'WPOND', "DPI": 'Fecha', 'EPI': 'HGan'})

    df_dpu = df[["ESP", "ID", 'Barcode', "WEIGHT", "Ratio", "DPU", "WPU", "EPU",'FASE']]
    df_dpu = df_dpu.dropna(subset=['DPU'])
    df_dpu["Etapa"] = "6-Punch List"
    df_dpu = df_dpu.rename(columns={'WPU': 'WPOND', "DPU": 'Fecha', 'EPU': 'HGan'})

    ##CONCATENAR VERTICAL DE LAS COLUMNAS DE RESUMEN

    dfv = pd.concat(
        [df_dtr.round(1), df_dpa.round(1), df_dmo.round(1), df_dni.round(1), df_dpi.round(1), df_dpu.round(1)], axis=0)

    dfv['WBRUTO'] = np.where(dfv.Etapa != '3-Montaje', 0, dfv.WEIGHT)

    np_array = dfv.to_numpy()

    dfv = pd.DataFrame(data=np_array,
                       columns=['ESP', 'ID', 'Barcode', 'WEIGHT', 'Ratio', 'Fecha', 'WPOND', 'HGan','FASE', 'Etapa',
                                   'WBRUTO'])
    dfv["Fecha"] = pd.to_datetime(dfv.Fecha).dt.date

    temp_info0=dfv[['Fecha', 'WPOND', 'WBRUTO']]

    info_0 = temp_info0.groupby(['Fecha']).sum() / 1000

    info_0['WPACUM'] = info_0['WPOND'].cumsum()
    info_0['WBACUM'] = info_0['WBRUTO'].cumsum()

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
    fusion=fusion[['Item','Code','Fi','Ff','Dias','Weight_Tk','Ratio_est']]
    fusion['horas'] = round(fusion.Weight_Tk * fusion.Ratio_est,0)
    sumahoras=fusion['horas'].sum()
    sumapeso = round(fusion['Weight_Tk'].sum(),)

    fusion["Fi"] = pd.to_datetime(fusion.Fi).dt.date                    #Se eliminan las horas a la fecha
    fusion["Ff"] = pd.to_datetime(fusion.Ff).dt.date                    #Se eliminan las horas a la fecha

    tree5 = ttk.Treeview(Regre)
    tree5['column'] = list(fusion.columns)
    tree5['show'] = 'headings'
    # loop trhu column
    for column in tree5['column']:
        tree5.heading(column, text=column)

    df_rows = fusion.to_numpy().tolist()
    for row in df_rows:
        tree5.insert("", "end", values=row)
    tree5.place(x=5, y=31)

    tree5.column("#1", width=50,stretch=tk.NO,anchor="center")
    tree5.column("#2", width=100, stretch=tk.NO,anchor="center")
    tree5.column("#3", width=88, stretch=tk.NO,anchor="center")
    tree5.column("#4", width=88, stretch=tk.NO,anchor="center")
    tree5.column("#5", width=50, stretch=tk.NO,anchor="center")
    tree5.column("#6", width=80, stretch=tk.NO, anchor="center")
    tree5.column("#7", width=40, stretch=tk.NO, anchor="center")
    tree5.column("#8", width=70, stretch=tk.NO, anchor="center")

    desl5 = ttk.Scrollbar(Regre, orient="vertical", command=tree5.yview)
    desl5.place(x=571, y=32, height=225)
    tree5.configure(yscrollcommand=desl5.set)

    Label(Regre, text=sumahoras, bg=d_color['fondo']).place(x=520, y=258)
    Label(Regre, text=sumapeso, bg=d_color['fondo']).place(x=405, y=258)

    ####################################################################################################################
    df_cp = pd.read_excel(import_file_path, sheet_name='cp_report')

def pbi1():  # FUNCION EXPORTAR PARA PBI

    dfpbi = df[
        ['ID', 'ESP', 'Barcode', 'WEIGHT', 'DTR', 'DPA', 'DMO', 'DNI', 'DPI', 'DPU', 'WTR', 'WPA', 'WMO', 'WNI', 'WPI',
         'WPU','FASE']]

    export_file = filedialog.askdirectory()
    dfpbi.to_csv(export_file + '/Matriz.csv', index=False)
    dfv.to_csv(export_file + '/QB_PBI.csv', header=True, index=False)

def filtrar():
    global animatrix2, info1, info2, info3, Nsemana, df, info4,info2p,info2h,info2b,df_semf

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

    tree.column("#1", width=89, minwidth=89, stretch=tk.NO,anchor="center")
    tree.column("#2", width=65, minwidth=65, stretch=tk.NO,anchor="center")
    tree.column("#3", width=68, minwidth=68, stretch=tk.NO,anchor="center")
    tree.column("#4", width=79, minwidth=79, stretch=tk.NO,anchor="center")
    tree.column("#5", width=79, minwidth=79, stretch=tk.NO,anchor="center")
    ######################################frame 3#################                                                      #SELECCION POR ESP

    temp_info2a = animatrix2[['ESP', 'WPOND', 'WBRUTO']]
    temp_info2b = temp_info2a.groupby(['ESP']).sum() * 0.001

    temp_info2c= df[['ESP', 'WEIGHT']]
    temp_info2d = temp_info2c.groupby(['ESP']).sum() * 0.001

    info2=pd.concat([temp_info2d,temp_info2b],axis=1).sort_values('ESP')

    info2.reset_index(inplace=True)

    info2.rename(columns={'WEIGHT': 'Total'}, inplace=True)

    info2['Pond%'] = info2.WPOND / info2.Total * 100
    info2['Bruto%'] = info2.WBRUTO / info2.Total * 100

    info2 = info2.round(2)
    info2.fillna(0, inplace=True)

    info2p=df[['ESP','WTR', 'WPA', 'WMO', 'WNI', 'WPI','WPU']]
    info2p=info2p.groupby(['ESP']).sum() * 0.001

    info2h=df[['ESP','ETR', 'EPA', 'EMO', 'ENI', 'EPI','EPU']]
    info2h = info2h.groupby(['ESP']).sum()

    info2b=df[['ESP','BWTR', 'BWPA', 'BWMO', 'BWNI', 'BWPI','BWPU']]
    info2b = info2b.groupby(['ESP']).sum()* 0.001

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

    tree2.column("#1", width=75, minwidth=75, stretch=tk.NO,anchor="center")
    tree2.column("#2", width=55, minwidth=55, stretch=tk.NO,anchor="center")
    tree2.column("#3", width=55, minwidth=55, stretch=tk.NO,anchor="center")
    tree2.column("#4", width=55, minwidth=55, stretch=tk.NO,anchor="center")
    tree2.column("#5", width=70, minwidth=70, stretch=tk.NO,anchor="center")
    tree2.column("#6", width=70, minwidth=70, stretch=tk.NO,anchor="center")

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

    df_sem=info3[['WPACUM','WBACUM']].copy()
    info3.reset_index(inplace=True)

    info3 = info3.round(2)
    info3.fillna(0, inplace=True)


    df_semf=pd.concat([df_sem,df_cp[['prog_a','real_a','forc_a']]],axis=1)

    #Se llenan los vacios hasta el ultimo dato
    df_semf=df_semf.apply(lambda series: series.loc[:series.last_valid_index()].ffill())

    df_semf = df_semf.iloc[28:] #Eliminamos las primeras 28 filas de la data

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

    tree3.column("#1", width=75, minwidth=75, stretch=tk.NO,anchor="center")
    tree3.column("#2", width=70, minwidth=70, stretch=tk.NO,anchor="center")
    tree3.column("#3", width=70, minwidth=70, stretch=tk.NO,anchor="center")
    tree3.column("#4", width=79, minwidth=79, stretch=tk.NO,anchor="center")
    tree3.column("#5", width=79, minwidth=79, stretch=tk.NO,anchor="center")

    temp_info4a = animatrix2[['FASE', 'WPOND', 'WBRUTO']]
    temp_info4b = temp_info4a.groupby(['FASE']).sum() * 0.001

    temp_info4c= df[['FASE', 'WEIGHT']]
    temp_info4d = temp_info4c.groupby(['FASE']).sum() * 0.001

    info4=pd.concat([temp_info4d,temp_info4b],axis=1).sort_values('FASE')

    info4.reset_index(inplace=True)

    info4.rename(columns={'WEIGHT': 'Total'}, inplace=True)

    info4['Pond%'] = info4.WPOND / info4.Total * 100
    info4['Bruto%'] = info4.WBRUTO / info4.Total * 100

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

    tree4.column("#1", width=75, minwidth=75, stretch=tk.NO,anchor="center")
    tree4.column("#2", width=55, minwidth=55, stretch=tk.NO,anchor="center")
    tree4.column("#3", width=55, minwidth=55, stretch=tk.NO,anchor="center")
    tree4.column("#4", width=55, minwidth=55, stretch=tk.NO,anchor="center")
    tree4.column("#5", width=70, minwidth=70, stretch=tk.NO,anchor="center")
    tree4.column("#6", width=70, minwidth=70, stretch=tk.NO,anchor="center")


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
    info2.to_excel(writer, sheet_name='ESP_filt')
    info2p.to_excel(writer,sheet_name='ESP_Peso')
    info2h.to_excel(writer, sheet_name='ESP_Horas')
    info2b.to_excel(writer, sheet_name='ESP_Bruto')
    info3.to_excel(writer, sheet_name='Semana', index=False)
    info4.to_excel(writer, sheet_name='Fase')

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
                                 name='MONTAJE PONDERADO ACUMULADO'))
        fig.add_trace(go.Scatter(x=info1.Fecha, y=info1.WBACUM,
                                 mode='lines+markers',
                                 name='MONTAJE BRUTO ACUMULADO'))
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

    elif combo.get() == "Montaje por Fase":
        fig = go.Figure(data=[
            go.Bar(name='Montaje Ponderado', x=info4.FASE, y=info4.WPOND),
            go.Bar(name='Montaje Bruto', x=info4.FASE, y=info4.WBRUTO),
            go.Bar(name='Total', x=info4.FASE, y=info4.Total)
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
    global fusion, info_proy, info_total,info_proyh, list_fase, fg
    k = 0

    #Se crean todas las fases y su distribucion por peso
    for i in fusion.Fi:
        dataframe = Calular(fusion.loc[k, 'Weight_Tk'], fusion.loc[k, 'Fi'], fusion.loc[k, 'Ff']).met1()
        k = k + 1
        list_fase.append(k)
        globals()["df_fase" + str(k)] = dataframe

    l = 0

    #Se crean todas las fases y su distribucion por horas
    for i in fusion.Fi:
        dataframeh = Calular(fusion.loc[l, 'horas'], fusion.loc[l, 'Fi'], fusion.loc[l, 'Ff']).met1()
        l = l + 1
        globals()["df_faseh" + str(l)] = dataframeh



    combo2["values"] = list_fase
    combo2.current(0)

    #Se crea listado de fechas para index

    fi = date(2019, 10, 31)   # Fecha inicial de Repore
    ff = date(2022, 8, 15)    # Fecha estimada de termino

    ft = [fi + timedelta(days=d) for d in range((ff - fi).days + 1)]  # CREAMOS LA LISTA DE FECHAS
    dft = pd.DataFrame({'Fecha': ft})
    dft.set_index('Fecha', inplace=True)

    info_proy=dft.copy()
    info_proyh = dft.copy()

    #Concatenando las fases para el info proy por peso
    for i in list_fase:
        globals()["concat" + str(i)]=pd.concat([  info_proy,globals()["df_fase" + str(i)]  ],axis=1)
        info_proy=globals()["concat" + str(i)]

    #Concatenando las fases para el info proy por horas
    for i in list_fase:
        globals()["concath" + str(i)]=pd.concat([info_proyh,globals()["df_faseh" + str(i)]],axis=1)
        info_proyh=globals()["concath" + str(i)]

    #Añadimos a cada fase la suma acumulada por peso
    for i in list_fase:
        globals()["df_fase" + str(i)]['Total']=globals()["df_fase" + str(i)].sum(axis=1)
        globals()["df_fase" + str(i)]['Acum']=globals()["df_fase" + str(i)]['Total'].cumsum()

    #Añadimos a cada fase la suma acumulada por horas
    for i in list_fase:
        globals()["df_faseh" + str(i)]['Total']=globals()["df_faseh" + str(i)].sum(axis=1)
        globals()["df_faseh" + str(i)]['Acum']=globals()["df_faseh" + str(i)]['Total'].cumsum()


    #SUMATORIA EL DF DE PESOS
    info_proy = info_proy.fillna(0)
    info_proy["Total_1"] = info_proy.sum(axis=1)
    info_proy['Acum_1'] = info_proy['Total_1'].cumsum()

    #SUMATORIA EL DF DE HORAS
    info_proyh = info_proyh.fillna(0)
    info_proyh["Total_1"] = info_proyh.sum(axis=1)
    info_proyh['Acum_1'] = info_proyh['Total_1'].cumsum()

    info_total=pd.concat([info_proy.Acum_1,info_0[['WPACUM','WBACUM']]],axis=1)

    #Se llenan los vacios hasta el ultimo dato
    info_total=info_total.apply(lambda series: series.loc[:series.last_valid_index()].ffill())

    dfv_proy=animatrix2[['Fecha','WPOND','WBRUTO','FASE']]

    #Se filtran las fases reales
    for i in list_fase:
        f = dfv_proy[(dfv_proy['FASE'] == i)]
        globals()["df_faser" + str(i)]=f.groupby(['Fecha']).sum()
        globals()["df_faser" + str(i)]=globals()["df_faser" + str(i)][['WPOND','WBRUTO']]
        globals()["df_faser" + str(i)]['WPACUM'] = globals()["df_faser" + str(i)]['WPOND'].cumsum()*0.001
        globals()["df_faser" + str(i)]['WBACUM']=globals()["df_faser" + str(i)]['WBRUTO'].cumsum()*0.001

    fg=[]

    # concatenamos con fecha total y llenamos los vacios
    for i in list_fase:

        if globals()["df_faser" + str(i)].WPOND.max()>0:
            fii = info_0.index.min()
            ffi = info_0.index.max()
            ftt = [fii + timedelta(days=d) for d in range((ffi - fii).days + 1)]  # CREAMOS LA LISTA DE FECHAS
            dftt = pd.DataFrame({'Fecha': ftt})
            dftt.set_index('Fecha', inplace=True)
            globals()["df_faserq" + str(i)] = pd.concat([dftt, globals()["df_faser" + str(i)]['WPACUM']], axis=1)
            globals()["df_faserq" + str(i)] = globals()["df_faserq" + str(i)].fillna(method="ffill")

            fg.append(i)


    #Concatenamos todas las fases
    for i in list_fase:
        globals()["df_faset" + str(i)]=pd.concat([globals()["df_faser" + str(i)],globals()["df_fase" + str(i)]],axis=1)
        globals()["df_faset" + str(i)]=globals()["df_faset" + str(i)].sort_values('Fecha')
        globals()["df_faset" + str(i)]=globals()["df_faset" + str(i)].apply(lambda series: series.loc[:series.last_valid_index()].ffill())

def exportar():  # FUNCION EXPORTAR PARA PBI

    export_file = filedialog.askdirectory()  # Buscamos el directorio para guardar
    # Creamos una excel y le indicamos la ruta
    writer = pd.ExcelWriter(export_file + '/' + 'Report_Proyectado.xlsx')

    # Write each dataframe to a different worksheet.
    globals()["df_faset" + str(combo2.get())].to_excel(writer, sheet_name='Detalle', index=True)

    writer.save()

def Gdia_fase():

    x = globals()["df_fase" + str(combo2.get())].index
    y = globals()["df_fase" + str(combo2.get())].Total
    fig1 = go.Figure(data=go.Bar(x=x, y=y))
    fig1.update_layout(title='Plan de Montaje diario - Fase '+str(combo2.get()),
                        yaxis=dict(
                        title='Toneladas',
                        titlefont_size=16,
                        tickfont_size=14,))
    fig1.show()

def Gdia_fase_Acu():

    x = globals()["df_fase" + str(combo2.get())].index
    y = globals()["df_fase" + str(combo2.get())].Acum
    fig2 = go.Figure(data=go.Scatter(x=x, y=y))
    fig2.update_layout(title='Plan de Montaje Acumulado - Fase ' + str(combo2.get()),
                        yaxis=dict(
                        title='Toneladas',
                        titlefont_size=16,
                        tickfont_size=14,))
    fig2.show()

def Gdia_fase_real():

    x = globals()["df_faser" + str(combo2.get())].index
    y = globals()["df_faser" + str(combo2.get())].WPOND
    fig3 = go.Figure(data=go.Bar(x=x, y=y))
    fig3.update_layout(title='Montaje Real Diario - Fase ' + str(combo2.get()),
                        yaxis=dict(
                        title='Toneladas',
                        titlefont_size=16,
                        tickfont_size=14,))
    fig3.show()

def Greal_Acum():

    x = globals()["df_faser" + str(combo2.get())].index
    y = globals()["df_faser" + str(combo2.get())].WPACUM
    fig4 = go.Figure(data=go.Scatter(x=x, y=y))
    fig4.update_layout(title='Montaje Real Acumulado - Fase ' + str(combo2.get()),
                        yaxis=dict(
                        title='Toneladas',
                        titlefont_size=16,
                        tickfont_size=14,))
    fig4.show()

def Greal_vs_Proy_fase():


    fig5 = go.Figure()

    # Add traces
    fig5.add_trace(go.Scatter(x=globals()["df_faset" + str(combo2.get())].index, y=globals()["df_faset" + str(combo2.get())].WPACUM,
                             mode='lines+markers',
                             name='Montaje Ponderado Acumulado'))
    fig5.add_trace(go.Scatter(x=globals()["df_faset" + str(combo2.get())].index, y=globals()["df_faset" + str(combo2.get())].WBACUM,
                             mode='lines+markers',
                             name='Montaje Bruto Acumulado'))
    fig5.add_trace(go.Scatter(x=globals()["df_faset" + str(combo2.get())].index, y=globals()["df_faset" + str(combo2.get())].Acum,
                             mode='lines+markers',
                             name='Plan de Montaje'))

    fig5.update_layout(title='Montaje de Estructuras Real vs Proyectado Fase ' + str(combo2.get()) )

    fig5.show()

def GProyec_total():

    fig6 = make_subplots(rows=2, cols=1)

    fig6.append_trace(go.Bar(
        x=info_proy.index,
        y=info_proy.Total_1,
        name='Montaje diario'
    ), row=1, col=1)

    fig6.append_trace(go.Scatter(
        x=info_proy.index,
        y=info_proy.Acum_1,
        name='Montaje Acumulado'
    ), row=2, col=1)

    fig6.update_layout(title='PLAN GENERAL DE MONTAJE DE ESTRUCTURAS',
                        yaxis=dict(
                        title='Toneladas',
                        titlefont_size=16,
                        tickfont_size=14,))

    fig6.show()

def Greal_vs_Proy_total():
    fig7 = go.Figure(data=[
        go.Scatter(name='Montaje Ponderado', x=info_total.index, y=info_total.WPACUM),
        go.Scatter(name='Montaje Bruto', x=info_total.index, y=info_total.WBACUM),
        go.Scatter(name='PROYECTADO', x=info_total.index, y=info_total.Acum_1)
    ])
    # Change the bar mode
    fig7.update_layout(barmode='group', xaxis_tickangle=-45,
                      font=dict(
                          family="Courier New, monospace",
                          size=18,
                          color="RebeccaPurple"
                      ))
    fig7.update_layout(title='MONTAJE DE ESTRUCTURAS REAL VS PROYECTADO',
                        yaxis=dict(
                        title='Toneladas',
                        titlefont_size=14,
                        tickfont_size=14,))
    fig7.show()

def forecast():

    tempserie=info_total.copy()
    tempserie['Fecha']=tempserie.index
    tempserie.reset_index(drop=True,inplace=True)

    bserie = tempserie[['Fecha','WBACUM']].dropna()
    pserie = tempserie[['Fecha', 'WPACUM']].dropna()
    fores = tempserie[['Fecha','Acum_1']].dropna()

    d_pro=int(euser2.get())

    ind_forb = [bserie.Fecha.max() + timedelta(days=d) for d in range(d_pro)]  # CREAMOS LA LISTA DE FECHAS
    ind_forp = [pserie.Fecha.max() + timedelta(days=d) for d in range(d_pro)]  # CREAMOS LA LISTA DE FECHAS

    pserie_model = auto_arima(pserie.WPACUM, start_p=1, start_q=1,
                              max_p=4, max_q=4, m=15,
                              start_P=0, seasonal=True,
                              d=1, D=1, trace=True,
                              error_action='ignore',
                              suppress_warnings=True,
                              stepwise=True)

    pserie_fore = pserie_model.predict(n_periods=d_pro)
    pserie_fore = pd.DataFrame(pserie_fore, index=ind_forb, columns=['Prediction'])

    bserie_model = auto_arima(bserie.WBACUM, start_p=1, start_q=1,
                              max_p=4, max_q=4, m=15,
                              start_P=0, seasonal=True,
                              d=1, D=1, trace=True,
                              error_action='ignore',
                              suppress_warnings=True,
                              stepwise=True)

    bserie_fore = bserie_model.predict(n_periods=d_pro)
    bserie_fore = pd.DataFrame(bserie_fore, index=ind_forp, columns=['Prediction'])


    fig0 = go.Figure(data=[
        go.Scatter(name='PONDERADO FORESC', x=pserie_fore.index, y=pserie_fore.Prediction),
        go.Scatter(name='PONDERADO REAL', x=pserie.Fecha, y=pserie.WPACUM),
        go.Scatter(name='PROYECTADO INICIAL', x=fores.Fecha, y=fores.Acum_1),
        go.Scatter(name='BRUTO REAL', x=bserie.Fecha, y=bserie.WBACUM),
        go.Scatter(name='BRUTO FORESC', x=bserie_fore.index, y=bserie_fore.Prediction)

    ])

    # Change the bar mode
    fig0.update_layout(barmode='group', xaxis_tickangle=-45,
                       font=dict(
                           family="Courier New, monospace",
                           size=18,
                           color="RebeccaPurple"
                       ))
    fig0.update_layout(title='MONTAJE DE ESTRUCTURAS REAL VS PROYECTADO',
                       yaxis=dict(
                           title='Toneladas',
                           titlefont_size=14,
                           tickfont_size=14, ))
    fig0.show()

def hh_proy():

    hhc=info_proyh.copy()

    fig8 = make_subplots(rows=2, cols=1)

    fig8.append_trace(go.Bar(
        x=hhc.index,
        y=hhc.Total_1,
        name='HH Diario Proyecto'
    ), row=1, col=1)

    fig8.append_trace(go.Scatter(
        x=globals()["df_fase" + str(combo2.get())].index,
        y=globals()["df_fase" + str(combo2.get())].Acum,
        name='HH Acumulado - Fase ' + str(combo2.get())
    ), row=2, col=1)

    fig8.update_layout(title='HH PROYECTADAS',
                        yaxis=dict(
                        title='HH',
                        titlefont_size=16,
                        tickfont_size=14,))
    fig8.show()

def exp_hh_proy():

    hhc=info_proyh[['Total_1','Acum_1']]

    export_file = filedialog.askdirectory()  # Buscamos el directorio para guardar
    # Creamos una excel y le indicamos la ruta
    writer = pd.ExcelWriter(export_file + '/' + 'HH_proyectado.xlsx')

    # Write each dataframe to a different worksheet.
    hhc.to_excel(writer, sheet_name='Total', index=True)
    globals()["df_faseh" + str(combo2.get())].to_excel(writer, sheet_name='Detalle', index=True)

    writer.save()

def Gsemana_tot():
    fig11 = go.Figure(data=[
        go.Scatter(name='Real Bruto Terreno', x=df_semf.index, y=df_semf.WBACUM),
        go.Scatter(name='Real Pond Terreno', x=df_semf.index, y=df_semf.WPACUM),
        go.Scatter(name='Real Bruto CP', x=df_semf.index, y=df_semf.real_a),
        go.Scatter(name='Programa Total', x=df_semf.index, y=df_semf.prog_a),
        go.Scatter(name='Forecast', x=df_semf.index, y=df_semf.forc_a),

    ])
    # Change the bar mode
    fig11.update_layout(barmode='group', xaxis_tickangle=-45,
                      font=dict(
                          family="Courier New, monospace",
                          size=18,
                          color="RebeccaPurple"
                      ))
    fig11.update_layout(title='MONTAJE DE ESTRUCTURAS REAL VS PROYECTADO',
                        yaxis=dict(
                        title='Toneladas',
                        titlefont_size=14,
                        tickfont_size=14,))
    fig11.show()

def Gsuma_fase():

    fig = go.Figure()

    for i in list_fase:
        if globals()["df_faser" + str(i)].WPOND.max()>0:
            fig.add_trace(go.Scatter(
                x=globals()["df_faserq" + str(i)].index, y=globals()["df_faserq" + str(i)].WPACUM,
                hoverinfo='x+y',
                mode='lines',
                name='Montaje Fase ' + str(i),
                line=dict(width=0.5),
                stackgroup='one'  # define stack group
            ))

    fig.show()

def Gsuma_esp():

    fig = go.Figure()

    for i in list_fase:
        if globals()["df_faser" + str(i)].WPOND.max()>0:
            fig.add_trace(go.Scatter(
                x=globals()["df_faserq" + str(i)].index, y=globals()["df_faserq" + str(i)].WPACUM,
                hoverinfo='x+y',
                mode='lines',
                name='Montaje Fase ' + str(i),
                line=dict(width=0.5),
                stackgroup='one'  # define stack group
            ))

    fig.show()

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
Widget(General, d_color['fondo'], 1, 1, 440, 130).letra('Resumen por Semana')
Widget(General, d_color['fondo'], 1, 1, 440, 410).letra('Resumen por Fase')
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
combo["values"] = ["Montaje Diario", "Montaje Semanal", "Montaje por ESP",'Montaje por Fase','Montaje Diario Acumulado']
combo.current(0)

# creando entrada de Fechas
fechai = DateEntry(General, width=16, bg='blue', date_pattern='yyyy/MM/dd', year=2019, month=10, day=1)
fechai.place(x=480, y=5)
fechaf = DateEntry(General, width=16, bg='blue', date_pattern='yyyy/MM/dd', year=2022, month=8, day=28)
fechaf.place(x=480, y=40)

##############################################################################################################
###Creando los label
Widget(Regre, d_color['fondo'], 1, 1, 7, 5).letra('Detalle de Fases')
Widget(Regre, d_color['fondo'], 1, 1, 600, 125).letra('Gráficos de Generales')
Widget(Regre, d_color['fondo'], 1, 1, 600, 310).letra('Gráficos por Fases')
Widget(Regre, d_color['fondo'], 1, 1, 600, 540).letra('Gráficos de HH')

###Creando los Buttons
Widget(Regre, d_color['boton'], 10, 3, 602, 28).boton('Calcular', proyeccion)

Widget(Regre, d_color['boton'], 28, 3, 602, 148).boton('Proyectado Total', GProyec_total)
Widget(Regre, d_color['boton'], 28, 3, 602, 205).boton('Real vs Proy', Greal_vs_Proy_total)

Widget(Regre, d_color['boton'], 13, 3, 604, 335).boton('Proy Dia', Gdia_fase)
Widget(Regre, d_color['boton'], 13, 3, 706, 335).boton('Proy Acum.', Gdia_fase_Acu)
Widget(Regre, d_color['boton'], 13, 3, 604, 392).boton('Real  Dia', Gdia_fase_real)
Widget(Regre, d_color['boton'], 28, 2, 602, 449).boton('Real vs Proy', Greal_vs_Proy_fase)
Widget(Regre, d_color['boton'], 13, 3, 706, 392).boton('Real Acum.', Greal_Acum)

Widget(Regre, d_color['boton'], 28, 2, 602, 491).boton('Exportar', exportar)
Widget(Regre, d_color['boton'], 28, 2, 602, 661).boton('Forecast', forecast)
Widget(Regre, d_color['boton'], 28, 2, 602, 564).boton('Graph Report HH', hh_proy)
Widget(Regre, d_color['boton'], 28, 2, 602, 605).boton('Expor Report HH', exp_hh_proy)
Widget(Regre, d_color['boton'], 28, 2, 402, 605).boton('Gráfico Semana Total', Gsemana_tot)
Widget(Regre, d_color['boton'], 28, 2, 402, 405).boton('Gráfico Suma Fases', Gsuma_fase)
Widget(Regre, d_color['boton'], 28, 2, 302, 305).boton('Gráfico Suma Fases', Gsuma_esp)


###Creando los Frames
Widget(Regre, d_color['fondo'], 584, 228, 5, 30).marco()     #FRAME RESUMEN POR FECHA
#Widget(Regre, d_color['fondo'], 200, 158, 600, 30).marco()

##Creando los entry
def on_click2(event):
    euser2.config(state=NORMAL)
    euser2.delete(0, END)

#creando entri de periodos
euser2 = Entry(Regre, width=10)
euser2.insert(0, 'DIAS_PROYEC')
euser2.config(state=DISABLED)
euser2.bind('<Button-1>', on_click2)
euser2.place(x=602, y=710)

##Creando Combobox
combo2 = ttk.Combobox(Regre, state="readonly",width=8)
combo2.place(x=735,y=310)
list_fase=[]
combo2["values"] = list_fase

root.mainloop()