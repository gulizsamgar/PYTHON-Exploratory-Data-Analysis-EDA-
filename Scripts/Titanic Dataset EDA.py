##################################################################
# Titanic Veri Seti (Görev 1-32)
##################################################################

# Genel Görevler: #

# Veri setini yükleme ve inceleme.
# Eksik değerleri ele alma ve veri temizliği.
# Veriyi gruplama ve özet istatistikler alma.
# Yeni özellikler oluşturma (Özellik Mühendisliği).
# Değişken dağılımlarını ve ilişkilerini görselleştirme (Tek ve İki Değişkenli Analiz).
# Sayısal değişkenlerin eğikliğini ve korelasyonlarını inceleme.

##################################################################
# Kütüphanelerin İçe Aktarılması ve Görüntü Ayarları
##################################################################

# Gerekli kütüphaneleri import ediyoruz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# Pandas gösterim ayarlarını yapılandırıyoruz
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


# Görselleştirme stillerini ayarlıyoruz
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")
plt.rcParams['figure.figsize'] = [12, 8]

##################################################################
# Veri Setini Yükleme
##################################################################

# Görev 1: Titanic veri setini yüklüyoruz
df = sns.load_dataset('titanic')

# Açıklama: ilk 5 satırı görüntülüyoruz
_df_head = df.head()
print("\nFirst 5 rows:")
display(_df_head)
# Açıklama: veri setinin boyutunu (satır, sütun) olarak kontrol ediyoruz
_df_shape = df.shape
print("\nDataset Shape:",_df_shape)
# Açıklama: veri setinin genel bilgilerini ekrana yazdırıyoruz
print("\nData Information:")
_df_info = df.info()

##################################################################
# Temel Veri Keşfi
##################################################################

# Görev 2: Nümerik Sütunların Tanımlayıcı İstatistiklerini Görüntülüyoruz
print("\nDescriptive Statistics:\n")
print(df.describe())

# Görev 3: Kadın ve erkek yolcu sayısını buluyoruz
print("\nNumber of Females:")
print(df['sex'].value_counts())

# Alternatif Yaklaşım: groupby kullanarak sayım
# print(df.groupby('sex').size())

# Alternatif Yaklaşım: groupby kullanarak size ile count
# print(df.groupby('sex')['sex'].count())

# Görev 4: Her sütundaki benzersiz değer sayısını buluyoruz
print("\nNumber of Unique Values:")
print(df.nunique())

# Görev 5: 'pclass' sütununun benzersiz değerlerini alıyoruz
print("\nUnique Values:")
print(df['pclass'].unique())

# Görev 6: 'pclass' ve 'parch' sütunlarının benzersiz değer sayılarını buluyoruz
print("\nNumber of Unique Values:")
print(df[['pclass', 'parch']].nunique())

##################################################################
# Veri Tipi Dönüşümü
##################################################################

# Görev 7: 'embarked' sütununun veri tipini kontrol edip kategorik tipe dönüştürüyoruz
print(df['embarked'].dtype)
df['embarked'] = df['embarked'].astype('category')
print(df['embarked'].dtype)

# Alternatif Yaklaşım: categorical dtype belirtmeden dönüştürme
# df['embarked'] = df['embarked'].astype(str) # String'e dönüştürme örneği

# Alternatif Yaklaşım: pd.Categorical ile daha spesifik kontrol
# df['embarked'] = pd.Categorical(df['embarked'])

##################################################################
# Veri Filtreleme ve Seçimi
##################################################################

# Görev 8: 'embarked' değeri 'C' olan kayıtları gösteriyoruz
print(df[df['embarked'] == 'C'].head(10))

# Alternatif Yaklaşım: .loc[] ile aynı işlevi görme
# print(df.loc[df['embarked'] == 'C'].head(10))

# Görev 9: 'embarked' değeri 'S' olmayan kayıtları gösteriyoruz
print(df[df['embarked'] != 'S'].head(10))

# Alternatif Yaklaşım: .loc[] ile aynı işlevi görme
# print(df.loc[df['embarked'] != 'S'].head(10))

# Alternatif Yaklaşım: .isin() ve negasyon (~) kullanma
# print(df[~df['embarked'].isin(['S'])].head(10))

# Görev 10: Yaşı 30'dan küçük ve kadın yolcuları seçiyoruz
print(df[(df['age'] < 30) & (df['sex'] == 'female')].head())

# Alternatif Yaklaşım: .loc[] ile aynı işlevi görme
# print(df.loc[(df['age'] < 30) & (df['sex'] == 'female')].head())

# Görev 11: 'fare' > 500 veya yaşı 70'ten büyük yolcuları seçiyoruz
print(df[(df['fare'] > 500) | (df['age'] > 70)].head())

# Alternatif Yaklaşım: .loc[] ile aynı işlevi görme
# print(df.loc[(df['fare'] > 500) | (df['age'] > 70)].head())

##################################################################
# Veri Temizliği
##################################################################

# Görev 12: Her sütundaki boş değerlerin toplamını buluyoruz
print("\nMissing Values Analysis:")
missing_values = df.isnull().sum().sort_values(ascending=False)
missing_percent = (missing_values / len(df)) * 100
missing_data = pd.concat([missing_values, missing_percent], axis=1,
 keys=['Missing Count', 'Missing Percent'])
print(missing_data[missing_data['Missing Count'] > 0])

# Alternatif Yaklaşım: .info() ile genel bilgi alma (eksik olmayan sayısını gösterir)
# print(df.info())

# Görev 13: 'who' sütununu DataFrame'den çıkarıyoruz
if 'who' in df.columns:
    df.drop('who', axis=1, inplace=True)
    print("'who' column was successfully extracted.")
else:
    print("Column 'who' not found in DataFrame.")

# Alternatif Yaklaşım: columns parametresiyle düşürme
# if 'who' in df.columns:
#     df.drop(columns=['who'], inplace=True)
#     print("'who' sütunu başarıyla çıkarıldı (columns parametresiyle).")

# Görev 14: 'deck' sütunundaki NaN değerleri en sık görülen değer (mode) ile dolduruyoruz
# Boş (NaN) değer sayısını kontrol et
print("Missing values:",df['deck'].isnull().sum())
# En çok tekrar eden değeri (modu) bul
mode_deger = df['deck'].mode()[0]
# Boş (NaN) değerleri bu mod değeri ile doldur
df['deck'] = df['deck'].fillna(mode_deger)
# Boş (NaN) değer sayısını tekrar kontrol et
print("Missing data after data filling:",df['deck'].isnull().sum())

# Alternatif Yaklaşım: inplace=True kullanma
# df['deck'].fillna(df['deck'].mode()[0], inplace=True)

# Görev 15: 'age' sütunundaki NaN değerleri medyan ile dolduruyoruz
# Boş (NaN) değer sayısını kontrol et
print("Missing values:",df['age'].isnull().sum())
# Medyan değeri bul
medyan_deger = df['age'].median()
# Boş (NaN) değerleri medyan ile doldur
df['age'] = df['age'].fillna(medyan_deger)
# Boş (NaN) değer sayısını tekrar kontrol et
print("Missing data after data filling:",df['age'].isnull().sum())

# Alternatif Yaklaşım: inplace=True kullanma
# df['age'].fillna(df['age'].median(), inplace=True)

# Alternatif Yaklaşım: Ortalama ile doldurma
# df['age'].fillna(df['age'].mean(), inplace=True)

# Alternatif Yaklaşım: Belirli bir değerle doldurma (örneğin 0)
# df['age'].fillna(0, inplace=True)

##################################################################
# Veri Gruplama ve Gruplara Göre Özet İstatistikler
##################################################################

# Görev 16: 'survived' değişkenini pclass ve sex kırılımında sum, count, mean ile analiz ediyoruz
print(df.groupby(['pclass', 'sex'])['survived'].agg(['sum', 'count', 'mean']))

# Alternatif Yaklaşım: Farklı agregasyon fonksiyonları listesi kullanma
# print(df.groupby(['pclass', 'sex'])['survived'].agg([np.sum, 'count', np.mean])) # numpy fonksiyonlarını kullanma

# Alternatif Yaklaşım: Birden fazla sütuna aynı agregasyonları uygulama
# print(df.groupby(['pclass', 'sex'])[['survived', 'fare']].agg(['mean']))

##################################################################
# Özellik Mühendisliği
##################################################################

# Görev 17: 'family_size' adında yeni bir sütun ekliyoruz (sibsp + parch + 1)
df['family_size'] = df['sibsp'] + df['parch'] + 1
print(df[['sibsp', 'parch', 'family_size']].head())

# Görev 18: Yaşa göre 'child' (çocuk) ve 'adult' (yetişkin) olarak iki kategoriye ayıran yeni bir sütun oluşturuyoruz
df['age_category'] = np.where(df['age'] < 18, 'child', 'adult')
print(df[['age', 'age_category']].head())

# Alternatif Yaklaşım: .apply() kullanma
# df['age_category'] = df['age'].apply(lambda x: 'child' if x < 18 else 'adult')
# print(df[['age', 'age_category']].head())

# Alternatif Yaklaşım: fonksiyon kullanma
#def  age_category(age):
#   if age< "18":
#        return 'child'
#    else:
#        return 'adult'
#df["age_category"] = df["age"].apply(lambda x: age_category(x))
#df.head()

##################################################################
# Tek Değişkenli (Univariate) Analiz - Kategorik Özellikler
##################################################################

# Görev 19: Hayatta Kalma Dağılımını Analiz Etme ve Görselleştirme
print("\nSurvival Distribution:")
survival_counts = df['survived'].value_counts()
print(survival_counts)
plt.figure(figsize=(8, 6))
sns.countplot(x='survived', data=df, palette=['red', 'green'], hue='survived', legend=False)
plt.title('Survival Distribution')
plt.xticks([0, 1], ['Did not survive', 'Survived'])
plt.ylabel('Count')
plt.show()

# Görev 20: Cinsiyet Dağılımını Analiz Etme ve Görselleştirme
print("\nPassenger Class Distribution:")
print(df['pclass'].value_counts())
plt.figure(figsize=(8, 6))
sns.countplot(x='pclass', data=df)
plt.title('Passenger Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# Görev 21: Yolcu Sınıfı Dağılımını Analiz Etme ve Görselleştirme
print("\nGender Distribution:")
print(df['sex'].value_counts())
plt.figure(figsize=(8, 6))
sns.countplot(x='sex', data=df)
plt.title('Gender Distribution')
plt.ylabel('Count')
plt.show()

# Görev 22: Biniş Limanı Dağılımını Analiz Etme ve Görselleştirme
print("\nEmbarkation Port Distribution:")
print(df['embarked'].value_counts())
plt.figure(figsize=(8, 6))
sns.countplot(x='embarked', data=df)
plt.title('Embarkation Port Distribution')
plt.xlabel('Port (C=Cherbourg, Q=Queenstown, S=Southampton)')
plt.ylabel('Count')
plt.show()

##################################################################
# İki Değişkenli (Bivariate) Analiz - Hayatta Kalma ve Kategorik Özellikler
##################################################################

plt.figure(figsize=(16, 10))

# Görev 23: Yolcu Sınıfına Göre Hayatta Kalma Dağılımını Görselleştirme
plt.subplot(2, 2, 1)
sns.countplot(x='pclass', hue='survived', data=df)
plt.title('Survival by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])

# Görev 24: Cinsiyete Göre Hayatta Kalma Dağılımını Görselleştirme
plt.subplot(2, 2, 2)
sns.countplot(x='sex', hue='survived', data=df)
plt.title('Survival by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])

# Görev 25: Biniş Limanına Göre Hayatta Kalma Dağılımını Görselleştirme
plt.subplot(2, 2, 3)
sns.countplot(x='embarked', hue='survived', data=df)
plt.title('Survival by Embarkation Port')
plt.xlabel('Port')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])

# Görev 26: Yaş Grubuna Göre Hayatta Kalma Dağılımını Görselleştirme
plt.subplot(2, 2, 4)
# 'Age_Group' sütunu oluşturulmamış olabilir, kontrol edelim veya oluşturalım
if 'age_category' in df.columns:
    sns.countplot(x='age_category', hue='survived', data=df)
    plt.title('Survival by Age Group')
    plt.xlabel('Age Group')
    plt.ylabel('Count')
    plt.legend(title='Survived', labels=['No', 'Yes'])
else:
    print("Warning: Column 'age_category' not found. Please ensure that you have created the column 'age_category'.")


plt.tight_layout()
plt.show()

##################################################################
# İki Değişkenli (Bivariate) Analiz - Hayatta Kalma ve Sayısal Özellikler
##################################################################

plt.figure(figsize=(16, 6))

# Görev 26: Yaş Dağılımını Hayatta Kalma Durumuna Göre Görselleştirme
plt.subplot(1, 2, 1)
sns.boxplot(x='survived', y='age', data=df) # 'Survived' yerine 'survived', 'Age' yerine 'age'
plt.title('Age Distribution by Survival')
plt.xlabel('Survived') # 'Survived' yerine 'survived'
plt.ylabel('Age') # 'Age' yerine 'age'

# Görev 27: Ücret Dağılımını Hayatta Kalma Durumuna Göre Görselleştirme
plt.subplot(1, 2, 2)
sns.boxplot(x='survived', y='fare', data=df) # 'Survived' yerine 'survived', 'Fare' yerine 'fare'
plt.title('Fare Distribution by Survival')
plt.xlabel('Survived') # 'Survived' yerine 'survived'
plt.ylabel('Fare') # 'Fare' yerine 'fare'

plt.tight_layout()
plt.show()

##################################################################
# Çok Değişkenli (Multivariate) Analiz - Korelasyon ve Çift Grafik
##################################################################

# Görev 28: Korelasyon Matrisini Hesaplama ve Görselleştirme
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 6))
correlation = numeric_df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Görev 29: Sayısal Özelliklerin Çift Grafiğini Oluşturma
sns.pairplot(df[['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']], hue='survived')
plt.suptitle('Pairplot of Numerical Variables by Survival', y=1.02)
plt.show()

##################################################################
# Sayısal Özelliklerin Eğiklik (Skewness) Analizi
##################################################################

# Görev 30: Çarpıklık Kontrolü ve Görselleştirme
print("\nSkewness Analysis:")
# Sütun isimlerinin küçük harfli olduğundan emin olalım
numeric_cols = ['age', 'fare', 'sibsp', 'parch']
for col in numeric_cols:
    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
        skewness = df[col].skew()
        if skewness > 0:
            print(f"{col} is positively skewed: {skewness:.4f}")
        elif skewness < 0:
            print(f"{col} is negatively skewed: {skewness:.4f}")
        else:
            print(f"{col} is normally distributed: {skewness:.4f}")

        # Eğriliği görselleştirme
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f'Distribution of {col}')

        plt.subplot(1, 2, 2)
        stats.probplot(df[col].dropna(), dist="norm", plot=plt)
        plt.title(f'Q-Q Plot of {col}')

        plt.tight_layout()
        plt.show()
    elif col not in df.columns:
        print(f"Uyarı: '{col}' sütunu veri setinde bulunamadı.")
    else:
        print(f"Uyarı: '{col}' sütunu nümerik bir sütun değil.")

##################################################################
# Çarpık Verileri Dönüştürme
##################################################################

# Görev 31: Ücret Değişkeninin Dağılımını Dönüştürme ve Görselleştirme
print("\nTransforming Skewed Variables:")
plt.figure(figsize=(15, 5)) # Grafik figürünün boyutunu ayarla (genişlik, yükseklik)

# Orijinal Ücret Dağılımı Grafiği
plt.subplot(1, 3, 1) # 1x3'lük bir ızgarada ilk alt grafiği seç
sns.histplot(df['fare'], kde=True) # Orijinal 'fare' sütununun histogramını çiz (yoğunluk eğrisi ile)
plt.title('\nOriginal Fare Distribution')

# Log-Dönüştürülmüş Ücret Dağılımı Grafiği
plt.subplot(1, 3, 2) # 1x3'lük ızgarada ikinci alt grafiği seç
sns.histplot(np.log1p(df['fare']), kde=True) # log1p dönüşümü uygulanmış 'fare' sütununun histogramını çiz
plt.title('Log-Transformed Fare Distribution')

# Karekök-Dönüştürülmüş Ücret Dağılımı Grafiği
plt.subplot(1, 3, 3) # 1x3'lük ızgarada üçüncü alt grafiği seç
sns.histplot(np.sqrt(df['fare']), kde=True)
plt.title('\nSquare Root-Transformed Fare Distribution')

plt.tight_layout()
plt.show()

##################################################################
# Çoklu Doğrusallığın (Multicollinearity) Tespiti
##################################################################

# Görev 32: Çoklu Doğrusallık Tespiti
print("\nMulticollinearity Detection:")

# Korelasyon matrisini hesapla (eğer daha önce hesaplanmadıysa)
  # Nümerik sütunları seç
  #numeric_df = df.select_dtypes(include=[np.number])
# Korelasyon matrisini hesapla
  #correlation = numeric_df.corr()

high_corr_pairs = []
for i in range(len(correlation.columns)):
    for j in range(i):
        # Korelasyon katsayısı |r| > 0.5 olan çiftleri belirle (Threshold of 0.5)
        if abs(correlation.iloc[i, j]) > 0.5:
            high_corr_pairs.append((correlation.columns[i], correlation.columns[j], correlation.iloc[i, j]))

print("Highly correlated pairs (|r| > 0.5):")
if high_corr_pairs:
    for var1, var2, corr in high_corr_pairs:
        print(f"{var1} and {var2}: {corr:.4f}")  # .4f: corr değişkeninin bir ondalık sayı (f) olarak formatlanacağını ve virgülden sonra 4 basamak (.4) gösterileceğini belirtir.
else:
    print("No highly correlated pairs found with |r| > 0.5")

