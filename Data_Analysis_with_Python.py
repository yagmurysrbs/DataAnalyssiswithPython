###############################################
# PYTHON İLE VERİ ANALİZİ (DATA ANALYSIS WITH PYTHON)
###############################################
# - NumPy(numerical python)
# - Pandas
# - Veri Görselleştirme: Matplotlib(düşük seviye) & Seaborn(yüksek seviye)
# - Gelişmiş Fonksiyonel Keşifçi Veri Analizi (Advanced Functional Exploratory Data Analysis)

#############################################
# NUMPY
#############################################
# Neden NumPy? (Why Numpy?)
#bilimsel Hesaplamalar için
#Verimli veri saklama-vektörel operasyonlar
#Yüksek seviyeden vvektörel işlemler
#Hızlı şekilde arrayler üzerinde çalışma imkanı sağlar

# NumPy Array'i Oluşturmak (Creating Numpy Arrays)
# NumPy Array Özellikleri (Attibutes of Numpy Arrays)
# Yeniden Şekillendirme (Reshaping)
# Index Seçimi (Index Selection)
# Slicing
# Fancy Index
# Numpy'da Koşullu İşlemler (Conditions on Numpy)
# Matematiksel İşlemler (Mathematical Operations)

#############################################
# Neden NumPy?
#############################################
import numpy as np
#numpy kütüphanesi dahil edilir

a = [1, 2, 3, 4]
b = [2, 3, 4, 5]

#klasik yol iki listenin elemanlarını çarpma
ab = []
for i in range(0, len(a)):
    ab.append(a[i] * b[i])
ab

#numpy ile iki listenin elemanlarını çarpma
a = np.array([1, 2, 3, 4])
b = np.array([2, 3, 4, 5])
a * b
#numpy ile direkt çarptık


#############################################
# NumPy Array'i Oluşturmak (Creating Numpy Arrays)
#############################################
import numpy as np

np.array([1, 2, 3, 4, 5])
#liste üzerinden bir numpy array oluşturur
type(np.array([1, 2, 3, 4, 5]))

np.zeros(10, dtype=int)
#girilen sayı adedince sıfır  array oluşturur
np.random.randint(0, 10, size=10)
#girilen aralıkta , size boyutlu array oluşturur

np.random.normal(10, 4, (3, 4))
# ortalama 10,standrt sapma 4 olan, 3 satır 4 sutundan olusan array

#############################################
# NumPy Array Özellikleri (Attibutes of Numpy Arrays)
#############################################
import numpy as np

# ndim: boyut sayısı
# shape: boyut bilgisi
# size: toplam eleman sayısı
# dtype: array veri tipi

a = np.random.randint(10, size=5)
#0 dan 10 a kadar olan sayılardan rasgele 5 değerli liste gelir

a.ndim
a.shape
a.size
a.dtype

#############################################
# Yeniden Şekillendirme (Reshaping)
#############################################
import numpy as np

np.random.randint(1, 10, size=9)
np.random.randint(1, 10, size=9).reshape(3, 3)
#reshape girilen boyutlarda yeniden uyarlar

ar = np.random.randint(1, 10, size=9)
ar.reshape(3, 3)


#############################################
# Index Seçimi (Index Selection)
#############################################
import numpy as np
a = np.random.randint(10, size=10)
a
a[0]
a[0:5]
a[0] = 999

#iki boyutlu array oluşturma
#3 satır , 5 sutun bilgisini ifade eder
m = np.random.randint(10, size=(3, 5))
m
m[0, 0]
m[1, 1]
# satır, sutun

m[2, 3] = 999

m[2, 3] = 2.9
#float ifade eklemek istedik ,numpt tek tip bilgisi tuttuğu için
#2.9 float değeri 2 olan int yuvarlayıp atadı

m[:, 0]
m[1, :]
m[0:2, 0:3 ]
# 2 ye kadar 3 e kadar sınırlar dahil değil

#############################################
# Fancy Index
#############################################
import numpy as np

v = np.arange(0, 30, 3)
v #0 dan 30 a kadar 3 er artarak olusan array
v[1]
v[4]

catch = [1, 2, 3]
v[catch]
#arraydan listedeki indexleri getirir

#############################################
# Numpy'da Koşullu İşlemler (Conditions on Numpy)
#############################################
import numpy as np
v = np.array([1, 2, 3, 4, 5])

#######################
# Klasik döngü ile
#######################
#listede 3 den kücük elemanları bulma
ab = []
for i in v:
    if i < 3:
        ab.append(i)

#######################
# Numpy ile
#######################
v < 3
#arrayde tüm elamanlar 3 den buyuk kucuk mu bakar true false verir

v[v < 3]
#kosulu sağlayan değerleri listeden getirir
v[v > 3]
v[v != 3]
v[v == 3]
v[v >= 3]

#############################################
# Matematiksel İşlemler (Mathematical Operations)
#############################################
import numpy as np
v = np.array([1, 2, 3, 4, 5])

v / 5
v * 5 / 10
v ** 2
v - 1
#arrayde tüm elemanlara sırayla islemler uygulanır

#cıkarma
np.subtract(v, 1)
#toplama
np.add(v, 1)
#ortalama
np.mean(v)
#toplam alma
np.sum(v)
np.min(v)
np.max(v)
#varyans
np.var(v)
#bu islemlerde cıktı verir ancak aklıcı tutulmaz
#bunun için islemi atamak gerekir

v = np.subtract(v, 1)
v
#######################
# NumPy ile İki Bilinmeyenli Denklem Çözümü
#######################

# 5*x0 + x1 = 12
# x0 + 3*x1 = 10

a = np.array([[5, 1], [1, 3]])
#katsayılar arrayi
b = np.array([12, 10])
#sonuclar array

np.linalg.solve(a, b)
#dneklem cözme

###############################
