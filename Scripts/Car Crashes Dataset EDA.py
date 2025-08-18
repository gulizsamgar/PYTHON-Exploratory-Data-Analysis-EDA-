##################################################################
# Car Crashes Veri Seti (Görev 1-4)
##################################################################

# Genel Görevler: #

# Veri setini yükleme.
# Sütun adlarını belirli kurallara göre listeleme (nümerikler için ön ek, 'no' içermeyenler için son ek).
# Belirli sütunları hariç tutarak yeni bir DataFrame oluşturma.

##################################################################
# Kütüphanelerin İçe Aktarılması ve Görüntü Ayarları
##################################################################

# Gerekli kütüphaneleri import ediyoruz
import pandas as pd
import seaborn as sns

# Pandas gösterim ayarlarını yapılandırıyoruz
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

##################################################################
# Veri Setini Yükleme
##################################################################

# Görev 1: car_crashes veri setini yüklüyoruz
df = sns.load_dataset('car_crashes')

# Açıklama: veri setinin sütun isimlerini kontrol ediyoruz
_df_columns = df.columns
print(_df_columns)
# Açıklama: veri setinin genel bilgilerini ekrana yazdırıyoruz
_df_info = df.info()

##################################################################
# Sütun Adlarını Değiştirme
##################################################################

# Görev 2: Nümerik değişken isimlerini büyük harfe çevirip başına 'NUM_' eki ekliyoruz
numeric_cols = [
    'NUM_' + col.upper() if df[col].dtype != 'O' else col.upper()
    for col in df.columns
]
print(numeric_cols)

# Görev 3: İsmi 'no' içermeyen sütunların sonuna '_FLAG' eki ekliyoruz
flagged_cols = [
    col.upper() + '_FLAG' if 'no' not in col else col.upper()
    for col in df.columns
]
print(flagged_cols)

# DataFrame sütunlarını doğrudan değiştirmek için alternatif (Dikkat: Orijinal df değişir)
    # Yeni sütun isimleri için bir dictionary oluşturulması gerekir
    # numeric_mapping = {col: 'NUM_' + col.upper() if df[col].dtype != 'O' else col.upper() for col in df.columns}
    # df.rename(columns=numeric_mapping, inplace=True)
    # print(df.head())

##################################################################
# Yeni Bir DataFrame Oluşturma
##################################################################

# Görev 4: 'abbrev' ve 'no_previous' dışındaki sütunlarla yeni DataFrame oluşturuyoruz
og_list = ['abbrev', 'no_previous']
new_cols = [col for col in df.columns if col not in og_list]
new_df = df[new_cols]
print(new_df.head())

# DataFrame sütunlarını doğrudan değiştirmek için alternatif (Dikkat: Orijinal df değişir)
    # Yeni sütun isimleri için bir dictionary oluşturulması gerekir
    # numeric_mapping = {col: 'NUM_' + col.upper() if df[col].dtype != 'O' else col.upper() for col in df.columns}
    # df.rename(columns=numeric_mapping, inplace=True)

