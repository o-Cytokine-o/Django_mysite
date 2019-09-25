import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class janken():
    janken_array = np.array([[1,0,0],[0,1,0],[0,0,1]])
    janken_class = ['グー','チョキ','パー']

    def create_model(self,Jprev_set):
        #モデルの初期化
        model = keras.Sequential()
        
        model.add(keras.layers.Dense(units=50,
                                    input_dim=Jprev_set.shape[1],
                                    kernel_initializer='glorot_uniform',
                                    bias_initializer='zeros',
                                    activation='tanh'))
        
        model.add(keras.layers.Dense(units=50,
                                    input_dim=50,
                                    kernel_initializer='glorot_uniform',
                                    bias_initializer='zeros',
                                    activation='tanh'))
        
        model.add(keras.layers.Dense(units=3,
                                    input_dim=50,
                                    kernel_initializer='glorot_uniform',
                                    bias_initializer='zeros',
                                    activation='softmax'))
        
        optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-7)
        
        model.compile(optimizer=optimizer,loss='categorical_crossentropy')
        
        return model

    def run(self):
        janken_array = np.array([[1,0,0],[0,1,0],[0,0,1]])
        janken_class = ['グー','チョキ','パー']

        #過去何回分の手を覚えているか
        n = 3

        Jprev = np.zeros(3*n*2)

        #過去の手をランダムに初期化
        for i in range(2*n):
            j = np.random.randint(0,3)
            Jprev[3*i:3*i+3] = janken_array[j]

        #現在の手をランダムに初期化
        j = janken_array[np.random.randint(0,3)]

        Jprev_set = np.array([Jprev])
        Jnow_set = np.array([j])

        model = self.create_model(Jprev_set)
        model.summary()
        history = model.fit(Jprev_set,Jnow_set)

        print("1:グー、2:チョキ、3:パー")

        win = 0
        draw = 0
        lose = 0

        try:
            while True:
                try:
                    j = int(input())-1
                    
                except(SyntaxError,NameError,ValueError):
                    continue
                    
                if j<0 or j>2:
                    continue
                
                Jprev_set = np.array([Jprev])
                Jnow_set = np.array([janken_array[j]])
                
                #コンピュータが過去の手から人間の現在の手を予測
                jpredict = model.predict_classes(Jprev_set)
                print(Jnow_set)
                
                #人間の手を代入
                your_choice = j
                
                #予測を元にコンピュータが決めた手
                comp_choice = (jpredict[0]+2)%3
                
                #お互いの手の表示
                print("あなた："+janken_class[your_choice]+", コンピュータ："+janken_class[comp_choice])
                
                #スコアを記録
                if your_choice == comp_choice:
                    draw += 1
                elif your_choice == (comp_choice+1)%3:
                    lose += 1
                else:
                    win += 1
                    
                #勝敗の結果表示
                print("あなたの勝ち: {0}, 負け：{1}, あいこ： {2}".format(win,lose,draw))
                
                #過去の手（入力データ）と現在の手（ターゲット）とでオンライン学習
                history = model.fit(Jprev_set,Jnow_set)
                
                #過去の手の末尾に現在のコンピュータの手を追加
                Jprev = np.append(Jprev[3:], janken_array[comp_choice])
                
                #過去の手の末尾に現在の人間の手を追加
                Jprev = np.append(Jprev[3:], janken_array[comp_choice])
                
        except KeyboardInterrupt:
            pass
        