import pandas as pd
from tkinter import *
from tkinter import filedialog
import numpy as np
import winsound
from datetime import date, timedelta

# Definir Colores
d_color = {'fondo': '#BDBDBD', 'boton': 'gray', 'framew': 'gray60', 'letra': '#BDBDBD'}
pd.options.mode.chained_assignment = None  # default='warn'

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
class Semana:                                   #CREAR DATA FRAME CON LA SEMANA
    def __init__(self,df):
        self.fi='2019-04-12'
        self.T=1300
        self.df=df

    def split(self):
        s = pd.date_range(start=self.fi, periods=self.T, freq='D') # Creas el ranfo de fechas
        Nsemana = pd.DataFrame(s, columns=['Fecha'])  # Lo conveiertes en dataframe
        Nsemana['SEMANA'] = Nsemana.index
        Nsemana["Fecha"] = pd.to_datetime(Nsemana.Fecha).dt.date
        Nsemana.set_index('Fecha', inplace=True)
        Nsemana['Semana'] = Nsemana.SEMANA // 7 + 1
        Nsemana['NSem'] = Nsemana.SEMANA % 7 + 1
        del Nsemana['SEMANA']
        Nsemana['FECHA'] = Nsemana.index
        Nsemana.reset_index(drop=True, inplace=True)
        Nsemana = Nsemana[['FECHA', 'Semana','NSem']]

        self.df = self.df.merge(Nsemana, on='FECHA', how='left')  # Buscas HH de ID y lo insertas en data Horas ganadas



        return self.df
class Master:
    def __init__(self, df,Num, Q1, Q2, Q3, Q4, Q5, Q6, QP,NQ1,NQ2,NQ3,NQ4,NQ5,NQ6,FB):
        self.df = df
        self.lista = [Q1, Q2, Q3, Q4, Q5, Q6]
        self.QP = QP
        self.lisNQ=[NQ1,NQ2,NQ3,NQ4,NQ5,NQ6]
        self.FB=FB
        self.Num=Num+1

    def develop(self):
        self.df = self.df[self.df.RATIO.notnull()]  # LIMPIAMOS DATOS QUE ESTEN NULOS EN EL RATIO
        self.df = self.df[self.df.CANT.notnull()]  # LIMPIAMOS LOS DATOS QUE ESTEN NULOS EN EL PESO

        Nqui = np.arange(1, self.Num, 1)
        lisQ = [str.format('Q{0}', x) for x in Nqui]
        lisTCQ = [str.format('TCQ{0}', x) for x in Nqui]
        lisTEQ = [str.format('TEQ{0}', x) for x in Nqui]
        lisCQ = [str.format('CQ{0}', x) for x in Nqui]
        lisBCQ = [str.format('BCQ{0}', x) for x in Nqui]
        lisEQ = [str.format('EQ{0}', x) for x in Nqui]


        for i in lisQ:
            self.df[i] = pd.to_datetime(self.df[i])

        # CALCULO DE PESOS TOTALES DE SEGUN PONDERACION

        k = 0
        for i in lisTCQ:
            self.df[i] = self.df.CANT * self.lista[k] #PRINT self.df['TCQ1']
            k+=1
        k=0
        for i in lisTEQ:
            self.df[i] = self.df.CANT * self.lista[k] * self.df.RATIO  # PRINT self.df['TEQ1']
            k += 1

        # CALCULO DE PESO SEGUN AVANCE
        k = 0
        for i in lisCQ:
            self.df[i] = np.where(self.df[lisQ[k]].isnull(), 0, self.df.CANT * self.lista[k])  # PRINT self.df['CQ1']
            k += 1

        # CALCULO DE PESO BRUTO QUIEBRE AVANCE
        k = 0
        for i in lisBCQ:
            self.df[i] = np.where(self.df[lisQ[k]].isnull(), 0, self.df.CANT)  # PRINT self.df['BCQ1']
            k += 1

        df_base = self.df

        # CALCULO DE HH EARNED SEGUN AVANCE
        k = 0
        for i in lisEQ:
            self.df[i] = np.where(self.df[lisQ[k]].isnull(), 0, self.df.CANT * self.lista[k] * self.df.RATIO)  # PRINT self.df['EQ1']
            k += 1
        self.df['CBRUTO'] = np.where(self.df[self.QP].isnull(), 0, self.df.CANT)
        self.df['CPOND']=self.df['CQ1']

        k = 0
        for i in lisCQ[:-1]:
            self.df['CPOND']=self.df['CPOND']+self.df[lisCQ[k+1]]
            k += 1

        #SEPARAMOS LOS PESOS DEL AVANCE
        dflineaM = pd.DataFrame(columns=["TAG", "CANT","DIAMETER",'FLUIDCODE', "RATIO",'MLPOND','FECHA', 'HHGan'])

        k=0
        for i in lisQ:
            globals()["df_" + str(k+1)]= self.df[["TAG", "CANT",'DIAMETER','FLUIDCODE', "RATIO", lisQ[k], lisCQ[k], lisEQ[k]]]
            globals()["df_" + str(k+1)] = globals()["df_" + str(k+1)].dropna(subset=[lisQ[k]])
            globals()["df_" + str(k + 1)] = globals()["df_" + str(k + 1)].rename(columns={lisCQ[k]: 'MLPOND', lisQ[k]: 'FECHA', lisEQ[k]: 'HHGan'})
            globals()["df_" + str(k + 1)]['Etapa'] = self.lisNQ[k]
            dflineaM = dflineaM.append(globals()["df_" + str(k+1)])
            k += 1



        dflineaM["FECHA"] = pd.to_datetime(dflineaM.FECHA).dt.date

        dflineaM['MLBRUTO'] = np.where(dflineaM.Etapa != self.FB, 0, dflineaM.CANT)
        dflineaM = Semana(dflineaM).split()  # Insertamos la Semana con class


        return dflineaM,df_base
class Masterelect:
    def __init__(self, df,Num, Q1, Q2, Q3, Q4, Q5, Q6, QP,NQ1,NQ2,NQ3,NQ4,NQ5,NQ6,FB):
        self.df = df
        self.lista = [Q1, Q2, Q3, Q4, Q5, Q6]
        self.QP = QP
        self.lisNQ=[NQ1,NQ2,NQ3,NQ4,NQ5,NQ6]
        self.FB=FB
        self.Num=Num+1

    def develop(self):
        self.df = self.df[self.df.RATIO.notnull()]  # LIMPIAMOS DATOS QUE ESTEN NULOS EN EL RATIO
        self.df = self.df[self.df.CANT.notnull()]  # LIMPIAMOS LOS DATOS QUE ESTEN NULOS EN EL PESO

        Nqui = np.arange(1, self.Num, 1)
        lisQ = [str.format('Q{0}', x) for x in Nqui]
        lisTCQ = [str.format('TCQ{0}', x) for x in Nqui]
        lisTEQ = [str.format('TEQ{0}', x) for x in Nqui]
        lisCQ = [str.format('CQ{0}', x) for x in Nqui]
        lisBCQ = [str.format('BCQ{0}', x) for x in Nqui]
        lisEQ = [str.format('EQ{0}', x) for x in Nqui]


        for i in lisQ:
            self.df[i] = pd.to_datetime(self.df[i])

        # CALCULO DE PESOS TOTALES DE SEGUN PONDERACION

        k = 0
        for i in lisTCQ:
            self.df[i] = self.df.CANT * self.lista[k] #PRINT self.df['TCQ1']
            k+=1
        k=0
        for i in lisTEQ:
            self.df[i] = self.df.CANT * self.lista[k] * self.df.RATIO  # PRINT self.df['TEQ1']
            k += 1

        # CALCULO DE PESO SEGUN AVANCE
        k = 0
        for i in lisCQ:
            self.df[i] = np.where(self.df[lisQ[k]].isnull(), 0, self.df.CANT * self.lista[k])  # PRINT self.df['CQ1']
            k += 1

        # CALCULO DE PESO BRUTO QUIEBRE AVANCE
        k = 0
        for i in lisBCQ:
            self.df[i] = np.where(self.df[lisQ[k]].isnull(), 0, self.df.CANT)  # PRINT self.df['BCQ1']
            k += 1

        d_cable_base = self.df

        # CALCULO DE HH EARNED SEGUN AVANCE
        k = 0
        for i in lisEQ:
            self.df[i] = np.where(self.df[lisQ[k]].isnull(), 0, self.df.CANT * self.lista[k] * self.df.RATIO)  # PRINT self.df['EQ1']
            k += 1
        self.df['CBRUTO'] = np.where(self.df[self.QP].isnull(), 0, self.df.CANT)
        self.df['CPOND']=self.df['CQ1']

        k = 0
        for i in lisCQ[:-1]:
            self.df['CPOND']=self.df['CPOND']+self.df[lisCQ[k+1]]
            k += 1

        #SEPARAMOS LOS PESOS DEL AVANCE
        d_cablem = pd.DataFrame(columns=["Service", "Cable_Code","CANT",'RATIO', 'HHGan'])

        k=0
        for i in lisQ:
            globals()["df_" + str(k+1)]= self.df[["Service", "Cable_Code",'CANT', "RATIO", lisQ[k], lisCQ[k], lisEQ[k]]]
            globals()["df_" + str(k+1)] = globals()["df_" + str(k+1)].dropna(subset=[lisQ[k]])
            globals()["df_" + str(k + 1)] = globals()["df_" + str(k + 1)].rename(columns={lisCQ[k]: 'MLPOND', lisQ[k]: 'FECHA', lisEQ[k]: 'HHGan'})
            globals()["df_" + str(k + 1)]['Etapa'] = self.lisNQ[k]
            d_cablem = d_cablem.append(globals()["df_" + str(k+1)])
            k += 1



        d_cablem["FECHA"] = pd.to_datetime(d_cablem.FECHA).dt.date

        d_cablem['MLBRUTO'] = np.where(d_cablem.Etapa != self.FB, 0, d_cablem.CANT)
        d_cablem = Semana(d_cablem).split()  # Insertamos la Semana con class


        return d_cablem,d_cable_base
class Restr:
    def __init__(self,df):
        self.df=df
    def add(self):
        self.df.loc[self.df.CATEGORIA != 'HH_RESTR', "RESTRICCION"] = None
        self.df['HHGast'] = np.where(self.df['CATEGORIA']=='HH_RESTR', self.df['HHGast']*(-1),self.df['HHGast'])


        return self.df
def import_OOCC():
    global nOOCC,mOOCC,dfIDOOCC

    ###CODIGO MECANICA#############################

    import_file_path = filedialog.askopenfilename()
    dfe = pd.read_excel(import_file_path,sheet_name='HHGan')
    dfs = pd.read_excel(import_file_path, sheet_name='HHGast')
    dfID = pd.read_excel(import_file_path, sheet_name='ID')

    Quiebre = {'TRASLADO': 0.03, 'FIERRO': 0.2, 'MOLDAJE/INSERTOS': 0.3, 'HORMIGON': 0.25, 'DESCIMBRE': 0.08, 'PUNCH': 0.04
               ,'MONTAJE':0.5,'JUNTAS':0.15,'NIVELACION':0.15,'PUNCH LIST':0.05}  # Quiebres Mecanica General

    dfIDOOCC = dfID[['ELEMENTO','SUB_ELEMENTO','CANTIDAD ITEM','ALCANCE TOTAL QUIEBRE','FACTOR']]  # Solo obtener TAG Y Horas de los equpos.

    # dfe[dfe == 0] = 'nan'                                       #Reemplazar con NaN los Zeros
    # dfs[dfs == 0] = 'nan'                                       #Reemplazar con NaN los Zeros

    dfe["FECHA"] = pd.to_datetime(dfe.FECHA).dt.date  # Conviertes fecha en formato sin horas
    dfs["FECHA"] = pd.to_datetime(dfs.FECHA).dt.date  # Conviertes fecha en formato sin horas



    nOOCC = dfe[['FECHA', 'SUB_ELEMENTO', 'QUIEBRE', 'SUPERVISOR', 'Cant', 'ACTIVIDAD']]  # Data frame de horas Ganadas
    nOOCC['RATIOQ'] = nOOCC['QUIEBRE'].map(Quiebre)  # Creas columnas según Diccionario

    dfIDOOCC1 = dfIDOOCC[['SUB_ELEMENTO', 'FACTOR']]  # Solo obtener TAG Y Horas de los equpos.
    nOOCC = nOOCC.merge(dfIDOOCC1, on='SUB_ELEMENTO', how='left')  # Buscas HH de ID y lo insertas en data Horas ganadas


    nOOCC = nOOCC.dropna(0)

    print(nOOCC)

    #nOOCC['HH_EARNED']=nOOCC['Cant']*nOOCC['RATIOQ']*nOOCC['FACTOR']


    nOOCC = nOOCC.dropna()

    mOOCC = dfs[['FECHA', 'SUB_ELEMENTO', 'QUIEBRE', 'SUPERVISOR', 'ACTIVIDAD', 'Capataz', 'MM', 'M1', 'M2', 'Ayudante',
             'Soldador']]  # Data frame de Horas Gastadas

    mOOCC = mOOCC.melt(id_vars=["FECHA", "SUB_ELEMENTO", 'QUIEBRE', 'SUPERVISOR', 'ACTIVIDAD'],
               var_name="CATEGORIA",
               value_name="HH_SPENT")

    mOOCC = mOOCC.dropna()


    f = 1999  # Frequency
    d = 900  # Duration
    winsound.Beep(f, d)
def import_STEELM():

    global dfv, df_base

    ########CODIGO STEEL###############

    import_file_path = filedialog.askopenfilename()
    df_master = pd.read_excel(import_file_path,sheet_name='Reporte',skiprows=7)

    d_pon = {'TR': 0.05, 'PA': 0.1, 'MO': 0.45, 'NI': 0.2, 'PI': 0.1, 'PU': 0.1}  # PONDERACIONES STEEL
    df_master = df_master[['IDTekla', 'ESP', 'Barcode', 'PesoTotal(Kg)', 'Ratio', 'Traslado', 'Prearmado', 'Montaje',
                           'Nivelacion,soldadura&Torque', 'Touchup', 'Punchlist', 'FASE', 'Clasificación']]

    df_master.rename(columns={'Traslado': 'DTR', 'Prearmado': 'DPA',
                              'Montaje': 'DMO', 'Nivelacion,soldadura&Torque': 'DNI', 'Touchup': 'DPI',
                              'Punchlist': 'DPU', 'IDTekla': 'ID', 'PesoTotal(Kg)': 'WEIGHT'},
                     inplace=True)

    df_master = df_master[df_master.Ratio.notnull()]  # LIMPIAMOS DATOS QUE ESTEN NULOS EN EL RATIO
    df_master = df_master[df_master.WEIGHT.notnull()]  # LIMPIAMOS LOS DATOS QUE ESTEN NULOS EN EL PESO

    # CALCULO DE PESOS TOTALES DE SEGUN PONDERACION

    df_master['TOTAL_WTR'] = df_master.WEIGHT * d_pon['TR']
    df_master['TOTAL_WPA'] = df_master.WEIGHT * d_pon['PA']
    df_master['TOTAL_WMO'] = df_master.WEIGHT * d_pon['MO']
    df_master['TOTAL_WNI'] = df_master.WEIGHT * d_pon['NI']
    df_master['TOTAL_WPI'] = df_master.WEIGHT * d_pon['PI']
    df_master['TOTAL_WPU'] = df_master.WEIGHT * d_pon['PU']

    # CALCULO DE HH EARNED TOTALES SEGUN MODERATION

    df_master['TOTAL_ETR'] = df_master.WEIGHT * d_pon['TR'] * df_master.Ratio / 1000
    df_master['TOTAL_EPA'] = df_master.WEIGHT * d_pon['PA'] * df_master.Ratio / 1000
    df_master['TOTAL_EMO'] = df_master.WEIGHT * d_pon['MO'] * df_master.Ratio / 1000
    df_master['TOTAL_ENI'] = df_master.WEIGHT * d_pon['NI'] * df_master.Ratio / 1000
    df_master['TOTAL_EPI'] = df_master.WEIGHT * d_pon['PI'] * df_master.Ratio / 1000
    df_master['TOTAL_EPU'] = df_master.WEIGHT * d_pon['PU'] * df_master.Ratio / 1000

    # CALCULO DE PESO SEGUN AVANCE
    df_master['WTR'] = np.where(df_master['DTR'].isnull(), 0, df_master.WEIGHT * d_pon['TR'])
    df_master['WPA'] = np.where(df_master['DPA'].isnull(), 0, df_master.WEIGHT * d_pon['PA'])
    df_master['WMO'] = np.where(df_master['DMO'].isnull(), 0, df_master.WEIGHT * d_pon['MO'])
    df_master['WNI'] = np.where(df_master['DNI'].isnull(), 0, df_master.WEIGHT * d_pon['NI'])
    df_master['WPI'] = np.where(df_master['DPI'].isnull(), 0, df_master.WEIGHT * d_pon['PI'])
    df_master['WPU'] = np.where(df_master['DPU'].isnull(), 0, df_master.WEIGHT * d_pon['PU'])

    # CALCULO DE PESO BRUTO QUIEBRE AVANCE
    df_master['BWTR'] = np.where(df_master['DTR'].isnull(), 0, df_master.WEIGHT)
    df_master['BWPA'] = np.where(df_master['DPA'].isnull(), 0, df_master.WEIGHT)
    df_master['BWMO'] = np.where(df_master['DMO'].isnull(), 0, df_master.WEIGHT)
    df_master['BWNI'] = np.where(df_master['DNI'].isnull(), 0, df_master.WEIGHT)
    df_master['BWPI'] = np.where(df_master['DPI'].isnull(), 0, df_master.WEIGHT)
    df_master['BWPU'] = np.where(df_master['DPU'].isnull(), 0, df_master.WEIGHT)

    df_base = df_master[['ID', 'ESP', 'WEIGHT', 'Ratio', 'FASE', 'Clasificación', 'BWTR', 'BWPA', 'BWMO', 'BWNI', 'BWPI', 'BWPU']]
    df_base.fillna(0,inplace=True)
    df_base['WEIGHT'] = df_base['WEIGHT']* 0.001
    df_base.loc['BWPA'] = df_base['BWPA'] * 0.001
    df_base.loc['BWMO'] = df_base['BWMO'] * 0.001
    df_base.loc['BWNI'] = df_base['BWNI'] * 0.001
    df_base.loc['BWPI'] = df_base['BWPI'] * 0.001
    df_base.loc['BWPU'] = df_base['BWPU'] * 0.001

    df_base.dropna(inplace=True)
    df_base.rename(columns={'FASE': 'ZONA'},
                       inplace=True)

    # CALCULO DE HH EARNED SEGUN AVANCE
    df_master['ETR'] = np.where(df_master['DTR'].isnull(), 0, df_master.WEIGHT * d_pon['TR'] * df_master.Ratio / 1000)
    df_master['EPA'] = np.where(df_master['DPA'].isnull(), 0, df_master.WEIGHT * d_pon['PA'] * df_master.Ratio / 1000)
    df_master['EMO'] = np.where(df_master['DMO'].isnull(), 0, df_master.WEIGHT * d_pon['MO'] * df_master.Ratio / 1000)
    df_master['ENI'] = np.where(df_master['DNI'].isnull(), 0, df_master.WEIGHT * d_pon['NI'] * df_master.Ratio / 1000)
    df_master['EPI'] = np.where(df_master['DPI'].isnull(), 0, df_master.WEIGHT * d_pon['PI'] * df_master.Ratio / 1000)
    df_master['EPU'] = np.where(df_master['DPU'].isnull(), 0, df_master.WEIGHT * d_pon['PU'] * df_master.Ratio / 1000)

    df_master['WBRUTO'] = np.where(df_master['DMO'].isnull(), 0, df_master.WEIGHT)
    df_master['WPOND'] = df_master.WTR + df_master.WPA + df_master.WMO + df_master.WNI + df_master.WPI + df_master.WPU

    ##########################SEPARAMOS LOS PESOS POR AVANCE DE CADA ETAPA

    df_dtr = df_master[["ESP", "ID", 'Barcode', "WEIGHT", "Ratio", "DTR", "WTR", "ETR", 'FASE', 'Clasificación']]
    df_dtr = df_dtr.dropna(subset=['DTR'])  # Elimina llas filas vacias de DTR
    df_dtr["Etapa"] = "1-Traslado"
    df_dtr = df_dtr.rename(columns={'WTR': 'WPOND', "DTR": 'Fecha', 'ETR': 'HGan'})

    df_dpa = df_master[["ESP", "ID", 'Barcode', "WEIGHT", "Ratio", "DPA", "WPA", "EPA", 'FASE', 'Clasificación']]
    df_dpa = df_dpa.dropna(subset=['DPA'])
    df_dpa["Etapa"] = "2-Ensamble"
    df_dpa = df_dpa.rename(columns={'WPA': 'WPOND', "DPA": 'Fecha', 'EPA': 'HGan'})

    df_dmo = df_master[["ESP", "ID", 'Barcode', "WEIGHT", "Ratio", "DMO", "WMO", "EMO", 'FASE', 'Clasificación']]
    df_dmo = df_dmo.dropna(subset=['DMO'])
    df_dmo["Etapa"] = "3-Montaje"
    df_dmo = df_dmo.rename(columns={'WMO': 'WPOND', "DMO": 'Fecha', 'EMO': 'HGan'})

    df_dni = df_master[["ESP", "ID", 'Barcode', "WEIGHT", "Ratio", "DNI", "WNI", "ENI", 'FASE', 'Clasificación']]
    df_dni = df_dni.dropna(subset=['DNI'])
    df_dni["Etapa"] = "4-Alineamiento"
    df_dni = df_dni.rename(columns={'WNI': 'WPOND', "DNI": 'Fecha', 'ENI': 'HGan'})

    df_dpi = df_master[["ESP", "ID", 'Barcode', "WEIGHT", "Ratio", "DPI", "WPI", "EPI", 'FASE', 'Clasificación']]
    df_dpi = df_dpi.dropna(subset=['DPI'])
    df_dpi["Etapa"] = "5-Touch_Up"
    df_dpi = df_dpi.rename(columns={'WPI': 'WPOND', "DPI": 'Fecha', 'EPI': 'HGan'})

    df_dpu = df_master[["ESP", "ID", 'Barcode', "WEIGHT", "Ratio", "DPU", "WPU", "EPU", 'FASE', 'Clasificación']]
    df_dpu = df_dpu.dropna(subset=['DPU'])
    df_dpu["Etapa"] = "6-Punch_List"
    df_dpu = df_dpu.rename(columns={'WPU': 'WPOND', "DPU": 'Fecha', 'EPU': 'HGan'})

    ##CONCATENAR VERTICAL DE LAS COLUMNAS DE RESUMEN

    dfv = pd.concat(
        [df_dtr.round(1), df_dpa.round(1), df_dmo.round(1), df_dni.round(1), df_dpi.round(1), df_dpu.round(1)], axis=0)


    dfv['WBRUTO'] = np.where(dfv.Etapa != '3-Montaje', 0, dfv.WEIGHT)

    np_array = dfv.to_numpy()
    dfv = pd.DataFrame(data=np_array,
                       columns=['ESP', 'ID', 'Barcode', 'WEIGHT', 'Ratio', 'FECHA', 'WPOND', 'HHGan', 'FASE',
                                'Clasificación', 'Etapa',
                                'WBRUTO'])
    dfv["FECHA"] = pd.to_datetime(dfv.FECHA).dt.date

    dfv = Semana(dfv).split()  # Insertamos la Semana con class


    dfv['WPOND'] = dfv['WPOND']*0.001
    dfv['WBRUTO'] = dfv['WBRUTO']*0.001
    dfv['WEIGHT'] = dfv['WEIGHT'] * 0.001

    dfv.dropna(subset=['HHGan'], inplace=True)
    dfv['Disc'] = 'Steel'

    dfv.rename(columns={'FASE': 'ZONA'},
                       inplace=True)

    Widget(root, d_color['fondo'], 1, 1, 140, 38).letra('STEEL-M')

    f = 1999  # Frequency
    d = 900  # Duration
    winsound.Beep(f, d)
def import_STEELR():

    global df_Steel_HH,st_person,rest_steel

    import_file_path = filedialog.askopenfilename()
    df_Steel_HH = pd.read_excel(import_file_path,sheet_name='HHGast')


    df_Steel_HH = df_Steel_HH[
        ['FECHA', 'FASE', 'QUIEBRE', 'SUPERVISOR', 'ACTIVIDAD', 'Capataz', 'MM', 'M1', 'M2', 'Ayudante',
         'Soldador','HH_RESTR','RESTRICCION']]  # Data frame de Horas Gastadas

    df_Steel_HH = df_Steel_HH.melt(id_vars=["FECHA", "FASE", 'QUIEBRE', 'SUPERVISOR', 'ACTIVIDAD','RESTRICCION'],
                                   var_name="CATEGORIA",
                                   value_name="HHGast")
    df_Steel_HH["FECHA"] = pd.to_datetime(df_Steel_HH.FECHA).dt.date
    df_Steel_HH = Restr(df_Steel_HH).add()  # Insertamos las HH restr limpiar
    df_Steel_HH.dropna(subset=['HHGast'],inplace=True)

    df_Steel_HH['Disc'] = 'Steel'
    df_Steel_HH = Semana(df_Steel_HH).split()

    df_Steel_HH.rename(columns={'FASE': 'ZONA'},
               inplace=True)

    Widget(root, d_color['fondo'], 1, 1, 140, 64).letra('STEEL-R')

    rest_steel=df_Steel_HH[df_Steel_HH.CATEGORIA=='HH_RESTR']
    rest_steel=rest_steel[['FECHA','CATEGORIA','HHGast','Semana','Disc','RESTRICCION']]

    f = 1999  # Frequency
    d = 900  # Duration
    winsound.Beep(f, d)
def import_MG():
    global nMG,mMG, dfID, maMG,mg_person,rest_MG
    ###CODIGO MECANICA#############################

    import_file_path = filedialog.askopenfilename()
    dfe = pd.read_excel(import_file_path,sheet_name='HHGan')
    dfs = pd.read_excel(import_file_path, sheet_name='HHGast')
    dfID = pd.read_excel(import_file_path, sheet_name='ID')

    Quiebre = {'TRASLADO': 0.1, 'MONTAJE': 0.5, 'ALIN_TOR_SOLD': 0.35, 'PUNCH_PROT': 0.5}  # Quiebres Mecanica General


    dfe = dfe.merge(dfID[['TAG_EQUIPO','HH']], on='TAG_EQUIPO', how='left')  # Buscas HH de ID y lo insertas en data Horas ganadas

    # dfe[dfe == 0] = 'nan'                                       #Reemplazar con NaN los Zeros
    # dfs[dfs == 0] = 'nan'                                       #Reemplazar con NaN los Zeros

    dfe["FECHA"] = pd.to_datetime(dfe.FECHA).dt.date  # Conviertes fecha en formato sin horas
    dfs["FECHA"] = pd.to_datetime(dfs.FECHA).dt.date  # Conviertes fecha en formato sin horas

    list_eq = list(set(dfe["TAG_EQUIPO"].tolist()))  # Creas lista de los equipos insertados
    list_qu = list(set(dfe["QUIEBRE"].tolist()))  # Lista de los quiebres insertados

    m = pd.DataFrame()
    n = pd.DataFrame()
    p = 0

    # Vamos a ordenar y aplicar el porcentaje de avance
    for i in list_eq:
        d = dict(tuple(dfe.groupby('TAG_EQUIPO')))  # Separamos el DF según Tag_Equipo
        e = dict(tuple(d[i].groupby('QUIEBRE')))
        globals()["list_qu" + str(i)] = list(set(d[i]["QUIEBRE"].tolist()))
        for k in globals()["list_qu" + str(i)]:
            e[k]['code'] = p
            p = p + 1
            e[k]['delta'] = e[k].AVAN_QUIEBRE_PORC.diff().fillna(
                e[k].AVAN_QUIEBRE_PORC)  # Resta y se rellena los vacios columna Avance
            n = pd.concat([n, e[k]], axis=0)


    n['FACTOR'] = n['QUIEBRE'].map(Quiebre)  # Creas columnas según Diccionario
    n['HHGan'] = n.HH * n.FACTOR * n.delta  # Calculas las HH Gaadas por Item

    nMG = n[['FECHA', 'TAG_EQUIPO', 'QUIEBRE', 'SUPERVISOR', 'HHGan']]  # Data frame de horas Ganadas

    #nMG.dropna(subset=['HHGan'], inplace=True)
    nMG = Semana(nMG).split()  # Insertamos la Semana con class
    nMG = nMG.merge(dfID[['TAG_EQUIPO', 'ZONA']], on='TAG_EQUIPO',how='left')  # Buscas HH de ID y lo insertas en data Horas ganadas

    m = dfs[['FECHA', 'TAG_EQUIPO', 'QUIEBRE', 'SUPERVISOR', 'Capataz', 'MM', 'M1', 'M2', 'Ayudante','RESTRICCION','HH_RESTR',
             'Soldador']]  # Data frame de Horas Gastadas

    mMG = m.melt(id_vars=["FECHA", "TAG_EQUIPO", 'QUIEBRE', 'SUPERVISOR','RESTRICCION'],
               var_name="CATEGORIA",
               value_name="HHGast")

    mMG = Semana(mMG).split()  # Insertamos la Semana con class
    mMG = mMG.merge(dfID[['TAG_EQUIPO', 'ZONA']], on='TAG_EQUIPO',
                    how='left')  # Buscas HH de ID y lo insertas en data Horas ganadas

    mMG = Restr(mMG).add()  # Insertamos las HH restr limpiar

    mMG.dropna(subset=['HHGast'],inplace=True)

    nMG['Disc'] = 'MG'
    mMG['Disc'] = 'MG'

    nMG=nMG[nMG['FECHA']>date(2021, 6, 22)]
    mMG = mMG[mMG['FECHA'] > date(2021, 6, 22)]

    rest_MG=mMG[mMG.CATEGORIA=='HH_RESTR']
    rest_MG=rest_MG[['FECHA','CATEGORIA','HHGast','Semana','Disc','RESTRICCION']]


    Widget(root, d_color['fondo'], 1, 1, 148, 94).letra('MG310')

    f = 1999  # Frequency
    d = 900  # Duration
    winsound.Beep(f, d)
def import_PIPING():

    global mPIPING,nPIPING,Totcode,pip_person,rest_pip

    Quiebre = {'U_TRASLADO': 0.05, 'U_TENDIDO': 0.05, 'U_EMPLANTILLADO': 0.23, 'U_SOLDADURA': 0.3,'U_REVESTIMIENTO': 0.17,
               'U_PRUEBA': 0.15,'U_PUNCH_LIST': 0.05,'A_TRASLADO': 0.1,'A_MONTAJE':0.1, 'A_EMPLANTILLADO': 0.1, 'A_SOLDADURA': 0.5,
               'A_PRUEBA': 0.15,'A_PUNCH_LIST': 0.05}  # Quiebres Mecanica General

    changQ = {'A_TRASLADO': '1-Traslado','A_MONTAJE':'2-Montaje', 'A_EMPLANTILLADO': '3-Emplantillado', 'A_SOLDADURA': '4-Soldadura',
               'A_PRUEBA': '5-Prueba','A_PUNCH_LIST': '6-Punch List','U_TRASLADO': 'a-Traslado', 'U_TENDIDO': 'b-Tendido', 'U_EMPLANTILLADO': 'c_EMPLANTILLADO', 'U_SOLDADURA': 'd-Soldadura','U_REVESTIMIENTO':'e-Revestimiento',
               'U_PRUEBA': 'f-Prueba','U_PUNCH_LIST': 'g-Prueba','SOPORTES':'SOPORTES','APOYO':'APOYO'}  # Quiebres Piping

    import_file_path = filedialog.askopenfilename()
    dfe = pd.read_excel(import_file_path,sheet_name='HHGan')                #Importar HHGan
    dfBulk = pd.read_excel(import_file_path, sheet_name='Bulk')
    dfBulk.rename(columns={'CLLENGTH':'CANT'},inplace=True)

    dflinea = pd.read_excel(import_file_path, sheet_name='Linea')
    dfsoporte = pd.read_excel(import_file_path, sheet_name='Soporte1')
    dfvalvu = pd.read_excel(import_file_path, sheet_name='Valvulas')
    mPIPING= pd.read_excel(import_file_path, sheet_name='HHGast')


    #LINEAS
    dflinea = dflinea[['TAG','FLUIDCODE','DIAMETER','CLLENGTH','LINENUM','DESCRIPTION_ESP','RATIO','TRASLADO','MONTAJE','EMPLANTILLADO','SOLDADURA','PRUEBA','PUNCH_LIST']]

    dflinea.rename(columns={'TRASLADO': 'Q1', 'MONTAJE': 'Q2',
                       'EMPLANTILLADO': 'Q3', 'SOLDADURA': 'Q4', 'PRUEBA': 'Q5',
                       'PUNCH_LIST': 'Q6','IDTekla':'ID','CLLENGTH':'CANT'},
              inplace=True)

    nLIN_A,nLIN_B = Master(dflinea,6, 0.1, 0.1, 0.1, 0.5, 0.15, 0.05, 'Q4','1-Traslado','2-Montaje','3-Emplantillado','4-Soldadura','5-Prueba','6-Punch List','4-Soldadura').develop()

    
    nLIN_A = nLIN_A.merge(dflinea[['TAG','DESCRIPTION_ESP']], on='TAG',
                    how='left')  # Buscas HH de ID y lo insertas en data Horas ganadas
    #nLIN_B = nLIN_B.merge(dflinea[['TAG','DESCRIPTION_ESP']], on='TAG',
    #                how='left')  # Buscas HH de ID y lo insertas en data Horas ganadas


    #SOPORTES
    dfsoporte = dfsoporte[['TAG','PESO','RATIO','FLUIDCODE','DIAMETER','TRASLADO','MONTAJE','TOUCH_UP','PUNCH']]

    dfsoporte.rename(columns={'TRASLADO': 'Q1', 'MONTAJE': 'Q2',
                       'TOUCH_UP': 'Q3', 'PUNCH': 'Q4','PESO':'CANT'},
              inplace=True)

    nSOP_A,nSOP_B = Master(dfsoporte,4, 0.2, 0.6, 0.15, 0.05, 0, 0.0, 'Q2','1-Traslado','2-Montaje','3-Touch Up','4-Punch Lis','5-NN','6-NN','2-Montaje').develop()


    nSOP_A['Etapa']='SOPORTES'
    print(nSOP_A)

    nSOP_A = nSOP_A.merge(dflinea[['TAG','DESCRIPTION_ESP']], on='TAG',
                    how='left')  # Buscas HH de ID y lo insertas en data Horas ganadas
    #nSOP_B = nSOP_B.merge(dflinea[['TAG','DESCRIPTION_ESP']], on='TAG',
    #                how='left')  # Buscas HH de ID y lo insertas en data Horas ganadas

    nSOP_A['DESCRIPTION_ESP']="Soportes"
    #nSOP_B['DESCRIPTION_ESP'] = "Soportes"


    #VALVULAS
    dfvalvu = dfvalvu[['TAG','CANT','FLUIDCODE','DIAMETER', 'RATIO','TRASLADO','MONTAJE','TOUCH_UP','PUNCH']]

    dfvalvu.rename(columns={'TRASLADO': 'Q1', 'MONTAJE': 'Q2',
                       'TOUCH_UP': 'Q3', 'PUNCH': 'Q4'},
              inplace=True)

    nVAL_A,nVAL_B = Master(dfvalvu,4, 0.2, 0.6, 0.15, 0.05, 0, 0.0, 'Q2','1-Traslado','2-Montaje','3-Touch Up','4-Punch Lis','5-NN','6-NN','2-Montaje').develop()
    nVAL_A['Etapa']='VALVULAS'

    nVAL_A = nVAL_A.merge(dflinea[['TAG','DESCRIPTION_ESP']], on='TAG',
                    how='left')  #
    #nVAL_B = nVAL_B.merge(dflinea[['TAG','DESCRIPTION_ESP']], on='TAG',
    #                how='left')  #


    nVAL_A['DESCRIPTION_ESP']="Valvula"
    #nVAL_B['DESCRIPTION_ESP'] = "Valvula"

    # Agregamos Itemd de tipo
    nLIN_A['Tipo']='Linea'
    #nLIN_B['Tipo']='Linea'
    nSOP_A['Tipo']='Soporte'
    #nSOP_B['Tipo']='Soporte'
    nVAL_A['Tipo']='Válvula'
    #nVAL_B['Tipo']='Válvula'

    nPIPING=pd.concat([nLIN_A,nSOP_A,nVAL_A],axis=0)

    del nPIPING['TAG']

    #hhGAN bULK
    dfe["FECHA"] = pd.to_datetime(dfe.FECHA).dt.date  # Conviertes fecha en formato sin horas
    dfe.rename(columns={'AVANCE (metrado diario)': 'CANT', 'QUIEBRE': 'Etapa'},
              inplace=True)

    dfe = dfe.merge(dfBulk[['TAG','FLUIDCODE', 'RATIO','DESCRIPTION_ESP']], on='TAG',
                    how='left')  #


    dfe['FACTOR'] = dfe['Etapa'].map(Quiebre)  # Creas columnas según Diccionario


    dfe['MLPOND']=dfe.CANT*dfe.FACTOR
    dfe['HHGan']=dfe.MLPOND*dfe.RATIO
    dfe['MLBRUTO'] = np.where(dfe.Etapa != 'U_SOLDADURA', 0, dfe.CANT)
    dfe = Semana(dfe).split()  # Insertamos la Semana con class
    dfe['Tipo'] = 'Linea UG'

    dfe = dfe[['CANT', 'RATIO','MLPOND','FECHA', 'HHGan','Etapa','MLBRUTO','Semana','FLUIDCODE','DIAMETER', 'DESCRIPTION_ESP','Tipo']]


    nPIPING = nPIPING.append(dfe)                                                                                       #Agregamos HHGan BUlk a las HH de matriz
    nPIPING.dropna(subset=['HHGan'], inplace=True)                                                                      #Limpiamos ala información



    Totcode=pd.concat([dflinea[['FLUIDCODE','DIAMETER','CANT']],dfBulk[['FLUIDCODE','DIAMETER','CANT']]],axis=0)  #Data base de metrados totales

    dfs = pd.read_excel(import_file_path, sheet_name='HHGast')                                                          #IMportamos horas gastadas.

    dfs["FECHA"] = pd.to_datetime(dfs.FECHA).dt.date                                                                    # Conviertes fecha en formato sin horas

    m = dfs[['FECHA', 'FLUIDCODE', 'DIAMETER', 'QUIEBRE', 'SUPERVISOR', 'Capataz', 'MM', 'M1', 'M2', 'Ayudante','RESTRICCION','HH_RESTR',
             'Soldador']]  # Data frame de Horas Gastadas

    mPIPING = m.melt(id_vars=["FECHA",'FLUIDCODE','DIAMETER', 'QUIEBRE', 'SUPERVISOR','RESTRICCION'],
               var_name="CATEGORIA",
               value_name="HHGast")

    mPIPING = Semana(mPIPING).split()                                                                                   # Insertamos la Semana con class
    mPIPING['Etapa'] = mPIPING['QUIEBRE'].map(changQ)                                                                   # Creas columnas según Diccionario

    mPIPING = Restr(mPIPING).add()  # Insertamos las HH restr limpiar

    mPIPING.dropna(subset=['HHGast'],inplace=True)

    nPIPING['Disc'] = 'PIP'
    mPIPING['Disc'] = 'PIP'

    rest_pip=mPIPING[mPIPING.CATEGORIA=='HH_RESTR']
    rest_pip=rest_pip[['FECHA','CATEGORIA','HHGast','Semana','Disc','RESTRICCION']]

    Widget(root, d_color['fondo'], 1, 1, 168, 130).letra('PIP')

    f = 1999  # Frequency
    d = 900  # Duration
    winsound.Beep(f, d)
def import_ELECT():
    global mELECT, nELECT, Totcode, pip_person, rest_pip

    Quiebre = {'U_TRASLADO': 0.05, 'U_TENDIDO': 0.05, 'U_EMPLANTILLADO': 0.23, 'U_SOLDADURA': 0.3,
               'U_REVESTIMIENTO': 0.17,
               'U_PRUEBA': 0.15, 'U_PUNCH_LIST': 0.05, 'A_TRASLADO': 0.1, 'A_MONTAJE': 0.1, 'A_EMPLANTILLADO': 0.1,
               'A_SOLDADURA': 0.5,
               'A_PRUEBA': 0.15, 'A_PUNCH_LIST': 0.05}  # Quiebres Mecanica General

    changQ = {'A_TRASLADO': '1-Traslado', 'A_MONTAJE': '2-Montaje', 'A_EMPLANTILLADO': '3-Emplantillado',
              'A_SOLDADURA': '4-Soldadura',
              'A_PRUEBA': '5-Prueba', 'A_PUNCH_LIST': '6-Punch List', 'U_TRASLADO': 'a-Traslado',
              'U_TENDIDO': 'b-Tendido', 'U_EMPLANTILLADO': 'c_EMPLANTILLADO', 'U_SOLDADURA': 'd-Soldadura',
              'U_REVESTIMIENTO': 'e-Revestimiento',
              'U_PRUEBA': 'f-Prueba', 'U_PUNCH_LIST': 'g-Prueba', 'SOPORTES': 'SOPORTES',
              'APOYO': 'APOYO'}  # Quiebres Piping

    import_file_path = filedialog.askopenfilename()
    d_cable = pd.read_excel(import_file_path, sheet_name='Cables')  # Importar HHGAN OF CABLES
    d_EPC = pd.read_excel(import_file_path, sheet_name='EPC')  # Importar HHGAN OF CABLES

    # CABLES
    d_cable= d_cable[
        ['Service', 'Cable_Code', 'ENGR_LGTH', 'Ubicación', 'RATIO', 'TRASLADO', 'TENDIDO',
         'CONEXIONADO', 'INSPECCION_PRUEBAS', 'PUNCH_LIST']]

    d_cable.rename(columns={'TRASLADO': 'Q1', 'TENDIDO': 'Q2',
                            'CONEXIONADO': 'Q3', 'INSPECCION_PRUEBAS': 'Q4', 'PUNCH_LIST': 'Q5', 'ENGR_LGTH': 'CANT'},
                   inplace=True)

    nCAB_A, nCAB_B = Masterelect(d_cable, 5, 0.1, 0.1, 0.1, 0.5, 0.15, 0, 'Q2', '1-Traslado', '2-Tendido',
                            '3-Conexionado', '4-Pruebas', '5-Punch List', '6-xxxx', '2-Tendido').develop()

    # EPC
    d_EPC= d_EPC[
        ['Sub_area', 'Partida', 'Cantidad', 'ESP', 'RATIO', 'TRASLADO', 'CANALIZACION',
         'SOPORTE_FAB', 'SOPORTE_MON', 'PUNCH_LIST']]

    d_EPC.rename(columns={'TRASLADO': 'Q1', 'CANALIZACION': 'Q2',
                            'SOPORTE_FAB': 'Q3', 'SOPORTE_MON': 'Q4', 'PUNCH_LIST': 'Q5', 'Cantidad': 'CANT'},
                   inplace=True)

    nEPC_A, nEPC_B = Masterelect(d_cable, 5, 0.1, 0.1, 0.1, 0.5, 0.15, 0, 'Q2', '1-Traslado', '2-Tendido',
                            '3-Conexionado', '4-Pruebas', '5-Punch List', '6-xxxx', '2-Tendido').develop()

    print(d_EPC)


    '''
    dfBulk = pd.read_excel(import_file_path, sheet_name='Bulk')
    dfBulk.rename(columns={'CLLENGTH': 'CANT'}, inplace=True)

    dflinea = pd.read_excel(import_file_path, sheet_name='Linea')
    dfsoporte = pd.read_excel(import_file_path, sheet_name='Soporte1')
    dfvalvu = pd.read_excel(import_file_path, sheet_name='Valvulas')
    mPIPING = pd.read_excel(import_file_path, sheet_name='HHGast')







    nLIN_A = nLIN_A.merge(dflinea[['TAG', 'DESCRIPTION_ESP']], on='TAG',
                          how='left')  # Buscas HH de ID y lo insertas en data Horas ganadas
    # nLIN_B = nLIN_B.merge(dflinea[['TAG','DESCRIPTION_ESP']], on='TAG',
    #                how='left')  # Buscas HH de ID y lo insertas en data Horas ganadas

    # SOPORTES
    dfsoporte = dfsoporte[['TAG', 'PESO', 'RATIO', 'FLUIDCODE', 'DIAMETER', 'TRASLADO', 'MONTAJE', 'TOUCH_UP', 'PUNCH']]

    dfsoporte.rename(columns={'TRASLADO': 'Q1', 'MONTAJE': 'Q2',
                              'TOUCH_UP': 'Q3', 'PUNCH': 'Q4', 'PESO': 'CANT'},
                     inplace=True)

    nSOP_A, nSOP_B = Master(dfsoporte, 4, 0.2, 0.6, 0.15, 0.05, 0, 0.0, 'Q2', '1-Traslado', '2-Montaje', '3-Touch Up',
                            '4-Punch Lis', '5-NN', '6-NN', '2-Montaje').develop()

    nSOP_A['Etapa'] = 'SOPORTES'
    print(nSOP_A)

    nSOP_A = nSOP_A.merge(dflinea[['TAG', 'DESCRIPTION_ESP']], on='TAG',
                          how='left')  # Buscas HH de ID y lo insertas en data Horas ganadas
    # nSOP_B = nSOP_B.merge(dflinea[['TAG','DESCRIPTION_ESP']], on='TAG',
    #                how='left')  # Buscas HH de ID y lo insertas en data Horas ganadas

    nSOP_A['DESCRIPTION_ESP'] = "Soportes"
    # nSOP_B['DESCRIPTION_ESP'] = "Soportes"

    # VALVULAS
    dfvalvu = dfvalvu[['TAG', 'CANT', 'FLUIDCODE', 'DIAMETER', 'RATIO', 'TRASLADO', 'MONTAJE', 'TOUCH_UP', 'PUNCH']]

    dfvalvu.rename(columns={'TRASLADO': 'Q1', 'MONTAJE': 'Q2',
                            'TOUCH_UP': 'Q3', 'PUNCH': 'Q4'},
                   inplace=True)

    nVAL_A, nVAL_B = Master(dfvalvu, 4, 0.2, 0.6, 0.15, 0.05, 0, 0.0, 'Q2', '1-Traslado', '2-Montaje', '3-Touch Up',
                            '4-Punch Lis', '5-NN', '6-NN', '2-Montaje').develop()
    nVAL_A['Etapa'] = 'VALVULAS'

    nVAL_A = nVAL_A.merge(dflinea[['TAG', 'DESCRIPTION_ESP']], on='TAG',
                          how='left')  #
    # nVAL_B = nVAL_B.merge(dflinea[['TAG','DESCRIPTION_ESP']], on='TAG',
    #                how='left')  #

    nVAL_A['DESCRIPTION_ESP'] = "Valvula"
    # nVAL_B['DESCRIPTION_ESP'] = "Valvula"

    # Agregamos Itemd de tipo
    nLIN_A['Tipo'] = 'Linea'
    # nLIN_B['Tipo']='Linea'
    nSOP_A['Tipo'] = 'Soporte'
    # nSOP_B['Tipo']='Soporte'
    nVAL_A['Tipo'] = 'Válvula'
    # nVAL_B['Tipo']='Válvula'

    nPIPING = pd.concat([nLIN_A, nSOP_A, nVAL_A], axis=0)

    del nPIPING['TAG']

    # hhGAN bULK
    dfe["FECHA"] = pd.to_datetime(dfe.FECHA).dt.date  # Conviertes fecha en formato sin horas
    dfe.rename(columns={'AVANCE (metrado diario)': 'CANT', 'QUIEBRE': 'Etapa'},
               inplace=True)

    dfe = dfe.merge(dfBulk[['TAG', 'FLUIDCODE', 'RATIO', 'DESCRIPTION_ESP']], on='TAG',
                    how='left')  #

    dfe['FACTOR'] = dfe['Etapa'].map(Quiebre)  # Creas columnas según Diccionario

    dfe['MLPOND'] = dfe.CANT * dfe.FACTOR
    dfe['HHGan'] = dfe.MLPOND * dfe.RATIO
    dfe['MLBRUTO'] = np.where(dfe.Etapa != 'U_SOLDADURA', 0, dfe.CANT)
    dfe = Semana(dfe).split()  # Insertamos la Semana con class
    dfe['Tipo'] = 'Linea UG'

    dfe = dfe[['CANT', 'RATIO', 'MLPOND', 'FECHA', 'HHGan', 'Etapa', 'MLBRUTO', 'Semana', 'FLUIDCODE', 'DIAMETER',
               'DESCRIPTION_ESP', 'Tipo']]

    nPIPING = nPIPING.append(dfe)  # Agregamos HHGan BUlk a las HH de matriz
    nPIPING.dropna(subset=['HHGan'], inplace=True)  # Limpiamos ala información

    Totcode = pd.concat([dflinea[['FLUIDCODE', 'DIAMETER', 'CANT']], dfBulk[['FLUIDCODE', 'DIAMETER', 'CANT']]],
                        axis=0)  # Data base de metrados totales

    dfs = pd.read_excel(import_file_path, sheet_name='HHGast')  # IMportamos horas gastadas.

    dfs["FECHA"] = pd.to_datetime(dfs.FECHA).dt.date  # Conviertes fecha en formato sin horas

    m = dfs[['FECHA', 'FLUIDCODE', 'DIAMETER', 'QUIEBRE', 'SUPERVISOR', 'Capataz', 'MM', 'M1', 'M2', 'Ayudante',
             'RESTRICCION', 'HH_RESTR',
             'Soldador']]  # Data frame de Horas Gastadas

    mPIPING = m.melt(id_vars=["FECHA", 'FLUIDCODE', 'DIAMETER', 'QUIEBRE', 'SUPERVISOR', 'RESTRICCION'],
                     var_name="CATEGORIA",
                     value_name="HHGast")

    mPIPING = Semana(mPIPING).split()  # Insertamos la Semana con class
    mPIPING['Etapa'] = mPIPING['QUIEBRE'].map(changQ)  # Creas columnas según Diccionario

    mPIPING = Restr(mPIPING).add()  # Insertamos las HH restr limpiar

    mPIPING.dropna(subset=['HHGast'], inplace=True)

    nPIPING['Disc'] = 'PIP'
    mPIPING['Disc'] = 'PIP'

    rest_pip = mPIPING[mPIPING.CATEGORIA == 'HH_RESTR']
    rest_pip = rest_pip[['FECHA', 'CATEGORIA', 'HHGast', 'Semana', 'Disc', 'RESTRICCION']]
    
    '''

    Widget(root, d_color['fondo'], 1, 1, 168, 130).letra('ELECT')

    f = 1999  # Frequency
    d = 900  # Duration
    winsound.Beep(f, d)
def export():



    nMGR = nMG[['FECHA', 'HHGan', 'Disc']]                                                                              #Filtras las HH Gan
    mMGR = mMG[['FECHA', 'HHGast', 'Disc']]                                                                             #Filtras las HH Gast

    nPIPINGR = nPIPING[['FECHA', 'HHGan', 'Disc']]
    mPIPINGR = mPIPING[['FECHA', 'HHGast', 'Disc']]

    dfvc = dfv[['FECHA', 'HHGan', 'Disc']]


    df_Steel_HHc = df_Steel_HH[['FECHA', 'HHGast', 'Disc']]
    #dfvc.columns = ['FECHA', 'HHGan', 'Disc']

    GlobGan = pd.concat([nMGR, dfvc,nPIPINGR], axis=0)                                                                  #Concatenmos las las HH Ganadas
    GlobGan.dropna(subset=['FECHA'],inplace=True)                                                                       #Limpiamos la información
    GlobGan=GlobGan[GlobGan['FECHA']> date(2021, 6, 15) ]
    GlobGan = Semana(GlobGan).split()                                                                                   # Insertamos la Semana con class



    GlobGas = pd.concat([mMGR, df_Steel_HHc,mPIPINGR], axis=0)                                                          # Concatenamos las HH Gastadas
    GlobGas = Semana(GlobGas).split()                                                                                   # Insertamos la Semana con class

    GlobGas_t = GlobGas[GlobGas.HHGast > 0]                                                                             #Retiaramos las HH de restricción

    GlobPerso1 = pd.concat([mMG[['FECHA','CATEGORIA','HHGast','Semana','NSem','Disc']], mPIPING[['FECHA','CATEGORIA','HHGast','Semana','NSem','Disc']],df_Steel_HH[['FECHA','CATEGORIA','HHGast','Semana','NSem','Disc']]], axis=0)
    GlobPerso1 = GlobPerso1[GlobPerso1.HHGast > 0]

    tempGlob=GlobPerso1.groupby(['Disc','Semana'])['NSem'].max()
    tempGlob=pd.DataFrame(tempGlob)
    tempGlob.reset_index(inplace=True)
    tempGlob.rename(columns={'NSem': 'FACT'},
                       inplace=True)
    tempGlob['Name'] = tempGlob["Disc"] + tempGlob["Semana"].astype(str)
    GlobPerso1['Name']=GlobPerso1["Disc"] + GlobPerso1["Semana"].astype(str)

    GlobPerso1 = GlobPerso1.merge(tempGlob[['Name','FACT']], on='Name',
                    how='left')
    GlobPerso1['Npers']=GlobPerso1['HHGast']/(11*GlobPerso1.FACT)

    H_Rest=pd.concat([rest_steel,rest_pip,rest_MG],axis=0)

    H_Rest.dropna(subset=['RESTRICCION'],inplace=True)


    export_file = filedialog.askdirectory()  # Buscamos el directorio para guardar
    writer = pd.ExcelWriter(export_file + '/' + 'QB2_HH.xlsx')  # Creamos una excel y le indicamos la ruta


    # Exportar Global
    GlobGan.to_excel(writer, sheet_name='Tot_Gan', index=True)
    GlobGas.to_excel(writer, sheet_name='Tot_Gas', index=True)
    GlobGas_t.to_excel(writer, sheet_name='Tot_Gas_t', index=True)

    # Exportar Personal
    GlobPerso1.to_excel(writer, sheet_name='Personal', index=True)
    H_Rest.to_excel(writer, sheet_name='Restricicones', index=True)


    '''
    # Exportar OOCC
    nOOCC.to_excel(writer, sheet_name='OOCC_Gan', index=True)
    mOOCC.to_excel(writer, sheet_name='OOCC_Gas', index=True)
    dfIDOOCC.to_excel(writer, sheet_name='OOCC_AllEquip', index=True)
    
    '''
    # Exportar MG
    nMG.to_excel(writer, sheet_name='MG_Gan', index=True)
    mMG.to_excel(writer, sheet_name='MG_Gas', index=True)
    dfID[['TAG_EQUIPO','HH','ZONA']].to_excel(writer, sheet_name='MG_AllEquip', index=True)

    # Exportar Steel
    dfv.to_excel(writer, sheet_name='ST_Gan', index=True)
    df_base.to_excel(writer, sheet_name='ST_Base', index=True)
    df_Steel_HH.to_excel(writer, sheet_name='ST_Gas', index=True)

    # Exportar PIPING

    nPIPING.to_excel(writer, sheet_name='PIP_Gan', index=True)
    mPIPING.to_excel(writer, sheet_name='PIP_Gas', index=True)
    Totcode.to_excel(writer, sheet_name='PIP_TAG', index=True)


    Widget(root, d_color['fondo'], 1, 1, 168, 178).letra('OK')

    writer.save()

    f = 1999  # Frequency
    d = 900  # Duration
    winsound.Beep(f, d)

root = Tk()
root.title('GESTION DE RESULTADOS PROYECTO QB2')
root.configure(bg="#BDBDBD")
root.geometry('340x365')  # Definir el tamaño de celda
root.resizable(width=0, height=0)

###Creando los frames
#Widget(root, d_color['fondo'], 150, 116, 5, 10).marco()      #FRAME DE RESUMEN


###Creando los label
Widget(root, d_color['fondo'], 1, 1, 7, 12).letra('Resumen General')


##Creando los botones
#Widget(root, d_color['boton'], 15, 1, 200, 10).boton('OOCC',import_OOCC)
Widget(root, d_color['boton'], 15, 1, 200, 35).boton('STEEL-MASTER', import_STEELM)
Widget(root, d_color['boton'], 15, 1, 200, 65).boton('STEEL-RECURSOS', import_STEELR)
Widget(root, d_color['boton'], 15, 1, 200, 95).boton('MG', import_MG)
Widget(root, d_color['boton'], 15, 1, 200, 125).boton('PIPING', import_PIPING)
#Widget(root, d_color['boton'], 15, 1, 200, 145).boton('ELECTRICIDAD',import_ELECT)
Widget(root, d_color['boton'], 15, 1, 200, 175).boton('EXPORTAR',export)

root.mainloop()


