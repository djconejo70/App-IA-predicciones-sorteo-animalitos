import tensorflow as te
import pandas as pd
import datetime as dt
import numpy as np
import warnings
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
import warnings
warnings.filterwarnings("ignore")
from collections import Counter
sw=0
array_rv=[]
swb=0
array_rvb=[]

while sw==0:
    df_read= pd.read_excel('granja.xlsx')
    df = pd.DataFrame(df_read)
    df.columns = ['resultado']
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(df['resultado'].values.reshape(-1,1))
    prediction_days = 1 # Number of days the neural network will predict
    x_train, y_train = [], []
    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x-prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2)) # To prevent overfitting
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2)) # To prevent overfitting
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=200, batch_size=38, shuffle=False, validation_split=0.1, verbose=0)
    prediction_prices = model.predict(x_train, verbose=0)
    prediction_prices = scaler.inverse_transform(prediction_prices)
    r0=prediction_prices[0]
    r01=float(r0)
    # print('Granjita:')
    v=str(r01)
    v1=v[3:5]
    # print(v1, "v1")
    rv=int(v1)
    rv=str(rv)
    rv1=int(v1)


    

    if rv1>37:
        v3=v[3]
        v4=v[4]
        v5=v[5]
        # print('v3 y v4-v5: ', v3, v4,v5)
        # rv12=int(v3)+int(v4)
        # rv=str(rv12)
        
        if rv1 >37 and rv1 <54:
            v3="0"
        if rv1 >53 and rv1 <69:
            v3="1"
        if rv1 >68 and rv1 <85:
            v3="2"
        if rv1 >85 and rv1 <=99:
            v3="3"
        rv=str(v3)+str(v4)
        # print('rv= ', rv)
        
    bale=str(rv1)
    balen=len(bale)
    if balen==1:
        # print("cuando 0 es el primer digito")
        v3="0"
        v4=str(rv1)
        rv=str(v3)+str(v4)
    
    #Aqui hago el ciclo y verifico el primer repetido
    array_rv.append(rv)
    print(array_rv)    
    #visited = set()
    #dup1 = {x for x in array_rv if x in visited or (visited.add(x) or False)}
    
    dup1 = {x for x in array_rv if array_rv.count(x) > 3}
    
    
    
    if dup1:
        print(dup1)
        # print(dup1, type(dup1))
        for e in dup1:
            # print(e, type(e))
            sw=1
            rv=str(e)           
            # print('rv valor antes de comparar', rv)
            if rv=="00":
                print('Dato Granjita: Ballena ', '(',rv,')')
            if rv=="01":
                print('Dato Granjita: Carnero ', '(',rv,')')
            if rv=="02":
                print('Dato Granjita: Toro ','(',rv,')')
            if rv=="03":
                print('Dato Granjita: Ciempies ','(',rv,')')
            if rv=="04":
                print('Dato Granjita: Alacran ', '(',rv,')')
            if rv=="05":
                print('Dato Granjita: Leon ', '(',rv,')')
            if rv=="06":
                print('Dato Granjita: Rana ', '(',rv,')')
            if rv=="07":
                print('Dato Granjita: Perico ', '(',rv,')')
            if rv=="08":
                print('Dato Granjita: Raton ', '(',rv,')')
            if rv=="09":
                print('Dato Granjita: Aguila ', '(',rv,')')
            if rv=="10":
                print('Dato Granjita: Tigre ', '(',rv,')')
            if rv=="11":
                print('Dato Granjita: Gato ','(',rv,')')
            if rv=="12":
                print('Dato Granjita: Caballo ','(',rv,')')
            if rv=="13":
                print('Dato Granjita: Mono ', '(',rv,')')
            if rv=="14":
                print('Dato Granjita: Paloma ','(',rv,')')
            if rv=="15":
                print('Dato Granjita: Zorro ', '(',rv,')')
            if rv=="16":
                print('Dato Granjita: Oso', '(',rv,')')
            if rv=="17":
                print('Dato Granjita: Pavo ', '(',rv,')')
            if rv=="18":
                print('Dato Granjita: Burro ', '(',rv,')')
            if rv=="19":
                print('Dato Granjita: Chivo ', '(',rv,')')
            if rv=="20":
                print('Dato Granjita: Cochino','(',rv,')')
            if rv=="21":
                print('Dato Granjita: Gallo ', '(',rv,')')
            if rv=="22":
                print('Dato Granjita: Camello ','(',rv,')')
            if rv=="23":
                print('Dato Granjita: Cebra ', '(',rv,')')
            if rv=="24":
                print('Dato Granjita: Iguana ', '(',rv,')')
            if rv=="25":
                print('Dato Granjita: Gallina ', '(',rv,')')
            if rv=="26":
                print('Dato Granjita: Vaca ', '(',rv,')')
            if rv=="27":
                print('Dato Granjita: Perro', '(',rv,')')
            if rv=="28":
                print('Dato Granjita: Zamuro ', '(',rv,')')
            if rv=="29":
                print('Dato Granjita: Elefante ', '(',rv,')')
            if rv=="30":
                print('Dato Granjita: Caiman ', '(',rv,')')
            if rv=="31":
                print('Dato Granjita: Lapa ', '(',rv,')')
            if rv=="32":
                print('Dato Granjita: Ardilla ','(',rv,')')
            if rv=="33":
                print('Dato Granjita: Pescado ', '(',rv,')')
            if rv=="34":
                print('Dato Granjita: Venado ', '(',rv,')')
            if rv=="35":
                print('Dato Granjita: Jirafa ','(',rv,')')
            if rv=="36":
                print('Dato Granjita: Culebra ', '(',rv,')')
            if rv=="37":
                print('Dato Granjita: Delfin ', '(0)')



while swb==0:
    dfla_read= pd.read_excel('lotoactivo.xlsx')
    dfla = pd.DataFrame(dfla_read)
    dfla.columns = ['resultado']
    scalerla = MinMaxScaler(feature_range=(0,1))
    scaled_datala = scalerla.fit_transform(dfla['resultado'].values.reshape(-1,1))
    prediction_daysla = 1 # Number of days the neural network will predict
    x_trainla, y_trainla = [], []
    for x in range(prediction_daysla, len(scaled_datala)):
        x_trainla.append(scaled_datala[x-prediction_daysla:x, 0])
        y_trainla.append(scaled_datala[x, 0])        
    x_trainla, y_trainla = np.array(x_trainla), np.array(y_trainla)
    x_trainla = np.reshape(x_trainla, (x_trainla.shape[0], x_trainla.shape[1], 1))
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_trainla.shape[1], 1)))
    model.add(Dropout(0.2)) # To prevent overfitting
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2)) # To prevent overfitting
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_trainla, y_trainla, epochs=100, batch_size=38, shuffle=False, validation_split=0.1, verbose=0)
    prediction_pricesla = model.predict(x_trainla, verbose=0)
    prediction_pricesla = scalerla.inverse_transform(prediction_pricesla)
    r0la=prediction_pricesla[0]
    # print('Loto:', r0la)

    r01b=float(r0la)
    vb=str(r01b)
    # print('v:',v, type(v))
    v1b=vb[3:5]
    # print('v1b: ', v1b)
    rvb=int(v1b)
    rvb=str(rvb)
    rv1b=int(v1b)
    if rv1b>37:
        v3b=vb[3]
        v4b=vb[4]
        v5b=vb[5]
        # print('v3b y v4b: ', v3b, v4b)
        rv12b=int(v3b)+int(v4b)
        # rvb=str(rv12b)
        # print('rvb= ', rvb)
        if rv1b >37 and rv1b <54:
            v3b="0"
        if rv1b >53 and rv1b <69:
            v3b="1"
        if rv1b >68 and rv1b <85:
            v3b="2"
        if rv1b >85 and rv1b <=99:
            v3b="3"
        rvb=str(v3b)+str(v4b)
        # print('rv= ', rvb)
    ble=str(rv1b)
    blen=len(ble)
    if blen==1:
        # print("cuando 0 es el primer digito")
        v3b="0"
        v4b=str(rv1b)
        rvb=str(v3b)+str(v4b)
        
        
    array_rvb.append(rvb)
    print(array_rvb)
    #visited = set()
    #dup2 = {x for x in array_rvb if x in visited or (visited.add(x) or False)}
    dup2 = {x for x in array_rvb if array_rvb.count(x) > 3}
    if dup2:
        for e in dup2:
            # print(e, type(e))
            swb=1
            rvb=str(e)           
            if rvb=="00":
                print('Dato LottoActivo: Ballena ', '(',rvb,')')
            if rvb=="01":
                print('Dato LottoActivo: Carnero ', '(',rvb,')')
            if rvb=="02":
                print('Dato LottoActivo: Toro ', '(',rvb,')')
            if rvb=="03":
                print('Dato LottoActivo: Ciempies ', '(',rvb,')')
            if rvb=="04":
                print('Dato LottoActivo: Alacran ', '(',rvb,')')
            if rvb=="05":
                print('Dato LottoActivo: Leon ', '(',rvb,')')
            if rvb=="06":
                print('Dato LottoActivo: Rana ', '(',rvb,')')
            if rvb=="07":
                print('Dato LottoActivo: Perico ', '(',rvb,')')
            if rvb=="08":
                print('Dato LottoActivo: Raton ', '(',rvb,')')
            if rvb=="09":
                print('Dato LottoActivo: Aguila ', '(',rvb,')')
            if rvb=="10":
                print('Dato LottoActivo: Tigre ', '(',rvb,')')
            if rvb=="11":
                print('Dato LottoActivo: Gato ', '(',rvb,')')
            if rvb=="12":
                print('Dato LottoActivo: Caballo ', '(',rvb,')')
            if rvb=="13":
                print('Dato LottoActivo: Mono ', '(',rvb,')')
            if rvb=="14":
                print('Dato LottoActivo: Paloma ', '(',rvb,')')
            if rvb=="15":
                print('Dato LottoActivo: Zorro ', '(',rvb,')')
            if rvb=="16":
                print('Dato LottoActivo: Oso ', '(',rvb,')')
            if rvb=="17":
                print('Dato LottoActivo: Pavo ', '(',rvb,')')
            if rvb=="18":
                print('Dato LottoActivo: Burro ', '(',rvb,')')
            if rvb=="19":
                print('Dato LottoActivo: Chivo ', '(',rvb,')')
            if rvb=="20":
                print('Dato LottoActivo: Cochino ', '(',rvb,')')
            if rvb=="21":
                print('Dato LottoActivo: Gallo ', '(',rvb,')')
            if rvb=="22":
                print('Dato LottoActivo: Camello ', '(',rvb,')')
            if rvb=="23":
                print('Dato LottoActivo: Cebra ', '(',rvb,')')
            if rvb=="24":
                print('Dato LottoActivo: Iguana ', '(',rvb,')')
            if rvb=="25":
                print('Dato LottoActivo: Gallina ', '(',rvb,')')
            if rvb=="26":
                print('Dato LottoActivo: Vaca ', '(',rvb,')')
            if rvb=="27":
                print('Dato LottoActivo: Perro ', '(',rvb,')')
            if rvb=="28":
                print('Dato LottoActivo: Zamuro ', '(',rvb,')')
            if rvb=="29":
                print('Dato LottoActivo: Elefante ', '(',rvb,')')
            if rvb=="30":
                print('Dato LottoActivo: Caiman ', '(',rvb,')')
            if rvb=="31":
                print('Dato LottoActivo: Lapa ', '(',rvb,')')
            if rvb=="32":
                print('Dato LottoActivo: Ardilla ', '(',rvb,')')
            if rvb=="33":
                print('Dato LottoActivo: Pescado ', '(',rvb,')')
            if rvb=="34":
                print('Dato LottoActivo: Venado ', '(',rvb,')')
            if rvb=="35":
                print('Dato LottoActivo: Jirafa ', '(',rvb,')')
            if rvb=="36":
                print('Dato LottoActivo: Culebra ', '(',rvb,')')
            if rvb=="37":
                print('Dato LottoActivo : Delfin ', '(0)')

           









