##################################################
# Pandas Alıştırmalar
##################################################

import numpy as np
import seaborn as sns
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

#########################################
# Görev 1: Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.
#########################################
df = sns.load_dataset("titanic")
df.head()
df.shape

df.describe()
df.describe(include='all')
df.isnull().sum()

#########################################
# Görev 2: Yukarıda tanımlanan Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.
#########################################

df["sex"].value_counts()
#df["sex"].value_counts(dropna=False)
df["deck"].value_counts(dropna=False)
#df.groupby("sex").count()
#df.groupby("sex").size()


#########################################
# Görev 3: Her bir sutuna ait unique değerlerin sayısını bulunuz.
#########################################

df.nunique()
#df.nunique(dropna=False)
#df["pclass"].nunique()

#########################################
# Görev 4: pclass değişkeninin unique değerleri bulunuz.
#########################################

df["pclass"].unique()

#type(df["pclass"].unique())

#########################################
# Görev 5:  pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.
#########################################

df[["pclass","parch"]].nunique()

#########################################
# Görev 6: embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz. Tekrar tipini kontrol ediniz.
#########################################

df["embarked"].dtype
df["embarked"] = df["embarked"].astype("category")
df["embarked"].dtype
df.info()


#########################################
# Görev 7: embarked değeri C olanların tüm bilgelerini gösteriniz.
#########################################

df[df["embarked"]=="C"].head(10)

#########################################
# Görev 8: embarked değeri S olmayanların tüm bilgelerini gösteriniz.
#########################################
df[df["embarked"] != "S"].head(10)

df[df["embarked"] != "S"]["embarked"].unique()
#"(2, object)" ifadesi, toplamda iki kategorinin olduğunu ve kategorilerin veri türünün "object" olduğunu belirtir.

df[~(df["embarked"] == "S")]["embarked"].unique()


#########################################
# Görev 9: Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.
#########################################

df[(df["age"]<30) & (df["sex"]=="female")].head()
# df.query("age < 30 and sex == 'female'").head()

#########################################
# Görev 10: Fare'i 500'den büyük veya yaşı 70 den büyük yolcuların bilgilerini gösteriniz.
#########################################

df[(df["fare"] > 500 ) | (df["age"] > 70 )].head()
#df.query("fare > 500 or age > 70").head()

#########################################
# Görev 11: Her bir değişkendeki boş değerlerin toplamını bulunuz.
#########################################
df.isnull().sum() #isnull() fonksiyonunun kendisi de isna() kullaniyor, ikisi de aynı
df.isna().sum()

#########################################
# Görev 12: who değişkenini dataframe'den düşürün.
#########################################
df.drop("who", axis=1, inplace=True)

#########################################
# Görev 13: deck değişkenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz.
#########################################
df["deck"].mode()
type(df["deck"].mode())
df["deck"].mode()[0]

#df.deck.value_counts(dropna=False)
#df["deck"].mode(dropna=False)[0]

df["deck"].fillna(df["deck"].mode()[0], inplace=True)
df["deck"].isnull().sum()


#########################################
# Görev 14: age değikenindeki boş değerleri age değişkenin medyanı ile doldurun.
#########################################

df["age"].fillna(df["age"].median(),inplace=True)
df.isnull().sum()

#########################################
# Görev 15: survived değişkeninin Pclass ve Cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.
#########################################

df.groupby(["pclass","sex"]).agg({"survived": ["sum","count","mean"]})
#df.pivot_table(values="survived", index=["pclass", "sex"], aggfunc=["sum", "count", "mean"])

#########################################
# Görev 16:  30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 verecek bir fonksiyon yazınız.
# Yazdığınız fonksiyonu kullanarak titanik veri setinde age_flag adında bir değişken oluşturunuz oluşturunuz. (apply ve lambda yapılarını kullanınız)
#########################################

def age_30(age):
    if age<30:
        return 1
    else:
        return 0

df["age_flag"] = df["age"].apply(lambda x : age_30(x))


df["age_flag2"] = df["age"].apply(lambda x: 1 if x<30 else 0)
df.head(10)
#df.drop("age_flag2",axis=1,inplace=True)


#########################################
# Görev 17: Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.
#########################################

df = sns.load_dataset("tips")
df.head()
df.shape


#########################################
# Görev 18: Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill  değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#########################################
df.time.value_counts(dropna=False)
df.groupby("time").agg({"total_bill": ["sum","min","mean","max"]})

#########################################
# Görev 19: Günlere ve time göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#########################################
df.groupby(["day","time"]).agg({"total_bill": ["sum","min","mean","max"]})
#df.pivot_table("total_bill",["day","time"],aggfunc=[np.sum,np.min,np.mean,np.max])

#########################################
# Görev 20:Lunch zamanına ve kadın müşterilere ait total_bill ve tip değerlerinin day'e göre toplamını, min, max ve ortalamasını bulunuz.
#########################################

df[(df["time"] == "Lunch") & (df["sex"] == "Female")].groupby("day").agg({"total_bill": ["sum","min","max","mean"],
                                                                          "tip":  ["sum","min","max","mean"]})


#########################################
# Görev 21: size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir?
#########################################

df.loc[(df["size"] < 3) & (df["total_bill"] >10 ) , "total_bill"].mean() # 17.184965034965035
#df[(df["size"] < 3) & (df["total_bill"] >10 )]["total_bill"].mean()


#########################################
# Görev 22: total_bill_tip_sum adında yeni bir değişken oluşturun. Her bir müşterinin ödediği totalbill ve tip in toplamını versin.
#########################################

df["total_bill_tip_sum"] = df["total_bill"] + df["tip"]
df.head()

#df["total_bill_tip_sum2"] = df["total_bill"].add(df["tip"])
#df.drop("total_bill_tip_sum2",axis=1,inplace=True)


#########################################
# Görev 23: total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir dataframe'e atayınız.
#########################################

new_df = df.sort_values("total_bill_tip_sum", ascending=False)[:30]
new_df.shape



### BONUS

#########################################
# Total_bill değişkeninin kadın ve erkek için ayrı ayrı ortalamasını bulun.
# Bulduğunuz ortalamaların altında olanlara 0, üstünde ve eşit olanlara 1 verildiği yeni bir total_bill_flag değişkeni oluşturun.
# Dikkat !! Female olanlar için kadınlar için bulunan ortalama dikkate alınacak, male için ise erkekler için bulunan ortalama.
# Parametre olarak cinsiyet ve total_bill alan bir fonksiyon yazarak başlayın. (If-else koşulları içerecek)
#########################################







































