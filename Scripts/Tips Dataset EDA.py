##################################################################
# Tips Veri Seti (Görev 1-17)
##################################################################

# Genel Görevler: #

# Veri setini yükleme ve ilk bilgilere bakma.
# Temel özet istatistikler ve eksik değer kontrolü.
# Veriyi gruplayarak istatistikler hesaplama.
# Belirli koşullara göre veriyi filtreleme.
# Yeni özellikler oluşturma.Veriyi sıralama.
# Temel görselleştirmeler yapma.

##################################################################
# Kütüphanelerin İçe Aktarılması ve Görüntü Ayarları
##################################################################

# Gerekli kütüphaneleri import ediyoruz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# scipy.stats bu bölüm için gerekli olmayabilir, ihtiyaca göre eklenebilir.


# Pandas gösterim ayarlarını yapılandırıyoruz
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Görselleştirme stillerini ayarlıyoruz (isteğe bağlı olarak güncellenebilir)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")
plt.rcParams['figure.figsize'] = [8, 6]

##################################################################
# Veri Setini Yükleme
##################################################################

# Görev 1: Tips veri setini yüklüyoruz
df = sns.load_dataset('tips')

# Veri setine genel bir bakış
print("\nFirst 5 rows:\n")
display(df.head())

print("\nDataset Shape:\n")
print(df.shape)

print("\nData Information:\n")
df.info()

##################################################################
# Temel Veri Keşfi
##################################################################

# Görev 2: Nümerik Sütunların Tanımlayıcı İstatistiklerini Görüntüleme
print("\nSummary statistics for numerical variables:\n")
display(df.describe())

# Görev 3: Kategorik Sütunların Tanımlayıcı İstatistiklerini Görüntüleme
print("\nSummary statistics for categorical variables:\n")
display(df.describe(include=['category']))

# Görev 4: Tüm sütunlardaki benzersiz değer sayılarını buluyoruz
print("\nNumber of Unique Values in All Columns:\n")
print(df.nunique())

# Görev 5: Eksik değerleri kontrol ediyoruz
print("\nMisisng Value Analysis:\n")
missing_values = df.isnull().sum().sort_values(ascending=False)
missing_percent = (missing_values / len(df)) * 100
missing_data = pd.concat([missing_values, missing_percent], axis=1,
                         keys=['Missing Count', 'Missing Percent'])
print(missing_data[missing_data['Missing Count'] > 0])

# Görev 6: Belirli bir sütunun benzersiz değerlerini alıyoruz (Örnek: 'day' sütunu)
print("\nUnique values in 'day' column:\n")
print(df['day'].unique())

##################################################################
# Veri Gruplama ve Agregasyon
##################################################################

# Görev 7: 'time' kategorisine göre total_bill istatistiklerini (sum, min, max, mean) alıyoruz
print("\nTotal_bill statistics by 'Time':\n")
print(df.groupby('time')['total_bill'].agg(['sum', 'min', 'max', 'mean']))

# Görev 8: 'time' ve 'day' bazlı total_bill istatistiklerini alıyoruz
print("\n\nTotal_bill statistics by 'time' and 'day'\n")
print(df.groupby(['time', 'day'])['total_bill'].agg(['sum', 'min', 'max', 'mean']))

# Görev 9: Lunch zamanındaki kadın müşterilere göre total_bill ve tip istatistiklerini alıyoruz
print("\nTotal_bill and type statistics for female customers at lunch time:\n")
filtered_df = df[(df['time'] == 'Lunch') & (df['sex'] == 'Female')]
print(filtered_df.groupby('day')[['total_bill', 'tip']].agg(['sum', 'min', 'max', 'mean']))

# Alternatif Yaklaşım: Koşullu filtreleme sonrası groupby ve agg
# print(df[df['time'] == 'Lunch'].groupby(['day', 'sex'])[['total_bill', 'tip']].agg(['sum', 'min', 'max', 'mean']).loc[:, 'Female', :]) # .loc[:, 'Female', :] ile sadece Female seçilir

##################################################################
# Veri Filtreleme ve Seçimi
##################################################################

# Görev 10: size < 3 ve total_bill > 10 koşulundaki siparişlerin ortalamasını hesaplıyoruz
print("\nAverage total_bill value of orders in records size < 3 and total_bill > 10:\n")
print(df.loc[(df['size'] < 3) & (df['total_bill'] > 10), 'total_bill'].mean())

# Alternatif Yaklaşım: [] kullanarak filtreleme
# print(df[(df['size'] < 3) & (df['total_bill'] > 10)]['total_bill'].mean())

##################################################################
# Veri Filtreleme ve Seçimi
##################################################################

# Görev 11: total_bill ve tip toplamını veren 'total_bill_tip_sum' sütununu oluşturuyoruz
df['total_bill_tip_sum'] = df['total_bill'] + df['tip']
print("\nAdded 'total_bill_tip_sum' column:\n")
print(df.head())

# Alternatif Yaklaşım: assign() metodunu kullanarak yeni sütun oluşturma (orijinal DataFrame'i değiştirmez, yeni bir kopya döndürür)
# new_df = df.assign(total_bill_tip_sum=df['total_bill'] + df['tip'])
# print(new_df.head())

##################################################################
# Veriyi Sıralama
##################################################################

# Görev 12: 'total_bill_tip_sum'a göre sıralayıp en yüksek 30 kaydı seçiyoruz
top30_df = df.sort_values('total_bill_tip_sum', ascending=False).head(30)
print("\nSize of top 30 records sorted by 'total_bill_tip_sum':", top30_df.shape)
print("\nTop 30 records sorted by 'total_bill_tip_sum':\n")
display(top30_df)

# Alternatif Yaklaşım: nlargest() metodunu kullanma (tek bir sütuna göre en büyük N değeri için daha performanslı olabilir)
# top30_df = df.nlargest(30, 'total_bill_tip_sum')
# print(top30_df.shape)

##################################################################
# Görsel Keşifsel Veri Analizi (EDA):
##################################################################

# Veri setini yüklüyoruz (eğer daha önce yüklenmediyse)
  #if 'df' not in globals():
    #df = sns.load_dataset('tips')

# Görev 13: Cinsiyet Dağılımını Pasta Grafiği ile Görselleştirme
print("\nGender Distribution Pie Chart:")
df['sex'].value_counts().plot(kind='pie', autopct='%0.1f%%')
plt.title('Gender Distribution')
plt.ylabel('') # y ekseni etiketini kaldır
plt.show()

# Görev 14: Günlere Göre Müşteri Sayısı Dağılımını Çubuk Grafiği ile Görselleştirme
print("\nNumber of Customers by Day:")
sns.countplot(x='day', data=df, order=['Thur', 'Fri', 'Sat', 'Sun'])
plt.title('Number of Customers by Day')
plt.xlabel('Day of the Week')
plt.ylabel('Count')
plt.show()

# Görev 15: Zaman Dilimine Göre Müşteri Sayısı Dağılımını Çubuk Grafiği ile Görselleştirme
print("\nNumber of Customers by Time Period:")
sns.countplot(x='time', data=df)
plt.title('Number of Customers by Time')
plt.xlabel('Time of Day')
plt.ylabel('Count')
plt.show()

# Görev 16: Toplam Hesap ve Bahşiş Arasındaki İlişkiyi Dağılım Grafiği ile Görselleştirme
print("\nTotal Bill and Tip Relationship:")
sns.scatterplot(x='total_bill', y='tip', data=df)
plt.title('Total Bill vs Tip')
plt.xlabel('Total Bill')
plt.ylabel('Tip Amount')
plt.show()

# Görev 17: Güne Göre Bahşiş Dağılımını Kutu Grafiği ile Görselleştirme
print("\nTip Distribution by Day:")
sns.boxplot(x='day', y='tip', data=df, order=['Thur', 'Fri', 'Sat', 'Sun'])
plt.title('Tip Distribution by Day')
plt.xlabel('Day of the Week')
plt.ylabel('Tip Amount')
plt.show()



