import numpy as np
import tensorflow as tf
import json
from tensorflow import keras
from tensorflow.keras import layers

class janken():
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

    model = create_model(Jprev_set)

    win = 0
    draw = 0
    lose = 0

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
        
    #予測して結果を返す関数
    def NewGame(self):
        Jprev_set,Jnow_set = self.readJson()
        jpredict = model.predict_classes(Jprev_set)
        return jpredict
    
    #手の入力と予測結果をもとに結果を返す関数（データの更新もする）
    #inputは1~3の値を受け取る
    def sess_predict(self,Jnow_input,jpredict):
        
        Jprev_set,Jnow_set = self.readJson()

        #人間の手を代入
        your_choice = Jnow_input
        
        #予測を元にコンピュータが決めた手
        comp_choice = (jpredict[0]+2)%3

        if your_choice == comp_choice:
            self.draw += 1
        elif your_choice == (comp_choice+1)%3:
            self.lose += 1
        else:
            self.win += 1

        Jprev = Jprev_set[0]

        #過去の手の末尾に現在のコンピュータの手を追加
        Jprev = np.append(Jprev[3:], janken_array[comp_choice])
        #過去の手の末尾に現在の人間の手を追加
        Jprev = np.append(Jprev[3:], janken_array[comp_choice])

        Jprev_set = np.array([Jprev])
        Jnow_set = np.array([self.janken_array[Jnow_input]])
        self.updateJson(Jprev_set,Jnow_set)


    #学習させる関数
    def sess_train(self):
        Jprev_set,Jnow_set = self.readJson()
        #過去の手（入力データ）と現在の手（ターゲット）とでオンライン学習
        history = self.model.fit(Jprev_set,Jnow_set)

    #Json読み込み関数
    def readJson(self):
        with open('polls/janken_data.json','r') as f:
            jd = json.load(f)
            Jprev_set = np.array(jd['jprev'])
            Jnow_set = np.array(jd['jnow'])

            return Jprev_set,Jnow_set

    #Json更新（じゃんけんデータ更新）
    def updateJson(self,Jprev_set,Jnow_set):
        jsond = {
            'jprev':Jprev_set.tolist(),
            'jnow':Jnow_set.tolist()
            }
        with open('polls/janken_data.json','w') as f:
            json.dump(jsond,f,indent='\t')


    def run(self):
        try:
            while True:
                try:
                    j = int(input())-1
                    
                except(SyntaxError,NameError,ValueError):
                    continue
                    
                if j<0 or j>2:
                    continue

                #ランダムに初期化したデータを代入
                Jprev_set = np.array([self.Jprev])
                #入力した手を代入
                Jnow_set = np.array([janken_array[j]])
                
                #コンピュータが過去の手から人間の現在の手を予測
                jpredict = model.predict_classes(Jprev_set)
                
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
        