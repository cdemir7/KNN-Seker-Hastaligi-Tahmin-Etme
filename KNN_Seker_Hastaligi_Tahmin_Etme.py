import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


#Outcome = 1 Diyabet Hastası
#Outcome = 0 Sağlıklı
data = pd.read_csv("diabetes.csv")
#print(data.head())


#Verileri hasta ve sağlıklı bireyler olarak ayırıp işleme başlamadan önce görselleştirelim.
seker_hastalari = data[data.Outcome == 1]
saglikli_bireyler = data[data.Outcome == 0]
plt.scatter(saglikli_bireyler.Age, saglikli_bireyler.Glucose, color="green", label="Sağlıklı Bireyler", alpha=0.4)
plt.scatter(seker_hastalari.Age, seker_hastalari.Glucose, color="red", label="SŞeker Hastaları", alpha=0.4)
plt.xlabel("Yaş")
plt.ylabel("Glukoz Miktarı")
plt.legend()
#plt.show()


#Eksenleri belirleyelim
#Y eksenini hasta-sağlıklı olarak belirliyoruz.
y = data.Outcome.values

#X ekseni için Outcome sütununu çıkartıyoruz.
x_ham_veri = data.drop(["Outcome"], axis=1)

#KNN algoritmasında yüksek verilerin küçük verileri ezmemesi için normalizasyon yapıyoruz.
#Bu şekilde yapay zeka algoritmamız daha doğru şekilde çalışabilecek.
x = (x_ham_veri - np.min(x_ham_veri))/(np.max(x_ham_veri) - np.min(x_ham_veri))
#print(x)


#Şimdi train datası ile test datamızı ayırıyoruz.
#Burada tran ile modelimizi eğitip, test verisi ile test edeceğiz.
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1)

#KNN veri modelimizi oluşturalım.
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print("K=3 için Test verilerimizin doğrulama testi sonucu: ", knn.score(x_test,y_test))

#K değerini kaç yapmalıyız?
#En iyi K değerini belirleme
sayac = 1
for k in range(1,21):
    knn_yeni = KNeighborsClassifier(n_neighbors=k)
    knn_yeni.fit(x_train,y_train)
    print(f"K={sayac} için Test verilerimizin doğrulama testi sonucu: {knn_yeni.score(x_test, y_test)}")
    sayac += 1