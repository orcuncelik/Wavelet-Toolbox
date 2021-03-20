from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi

from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
import librosa
import numpy as np
import pywt
from src import Features, Database


class DiscreteWT(QMainWindow):

    def __init__(self):
        QMainWindow.__init__(self)
        loadUi("../ui/Discrete.ui", self)
        self.setWindowTitle("Discrete Wavelet Analysis")
        self.setWindowIcon(QtGui.QIcon('../icon/iDiscrete Wavelet Analysiscon.png'))
        self.setFixedSize(991, 830)
        Features.center(self)

        self.MplWidget.canvas.axes.remove()                 #ortada oluşan modeli siliyor
        self.clear_count = 0                                #currently how many canvas we have
        self.load_check = False                             #Sinyal Yüklenmiş Mi?
        self.analyze_check = False                          #İşlenmiş mi

        self.signals = []   #Sinyaller
        self.audio = []      #Ses Dosyaları
        self.time = []        #Zaman
        #self.addToolBar(NavigationToolbar(self.MplWidget.canvas, self))  # Sinyalleri daha iyi görmek için toolbar

        self.actionClose.triggered.connect(self.close_window)                               #Sayfayı Kapatma
        self.actionFolder.triggered.connect(self.load_folder)                               #Menu Bardan Dosya Yükleme
        self.actionImage.triggered.connect(self.load_signal)                                #Menu Bardan Sinyal Yükleme
        self.pushButton_1D_Analyze.clicked.connect(self.analyze_signal)                     #Analiz butonu
        self.progressBar.setValue(0)                                                        #progressBar değeri ilk başta 0

        self.pushButton_save_to_db.clicked.connect(self.save_to_db)                         #Veritabanına kaydetme


        wave_types = pywt.wavelist(kind='discrete')
        self.comboBox_1D_Type.addItems(wave_types)                                          #Wavelet Typeları için comboBox oluşturuluyor
        for i in range(1,20):
            self.comboBox_1D_Level.addItem(str(i) , i)

        self.comboBox_1D_Level.currentIndexChanged.connect(self.on_combobox_changed, self.comboBox_1D_Level.currentIndex())

        self.checkBox_1D_Average.setChecked(True)
        self.checkBox_1D_Entropy.setChecked(True)
        self.checkBox_1D_Kurtosis.setChecked(True)
        self.checkBox_1D_Max.setChecked(True)
        self.checkBox_1D_Median.setChecked(True)
        self.checkBox_1D_Min.setChecked(True)
        self.checkBox_1D_Skewness.setChecked(True)
        self.checkBox_1D_StandartDeviation.setChecked(True)

    def close_window(self):
        self.close()

    def on_combobox_changed(self, value):
        if(value >= 7):
            Features.message("Plotting is not supported on levels higher than 5",QMessageBox.Information)

    def load_signal(self):
        self.all_signals, self.filter, self.load_check = Features.load_signal(self)
        self.progressBar.setValue(0)
        if(self.load_check):
            if(len(self.all_signals) > 1):                                      #Çoklu Veri Girişi
                self.label_1D_DataName.setText("Multiple Signals")              #Veri Adı
            else:
                name = self.all_signals[0].split(sep ='/')
                self.label_1D_DataName.setText(name[-1])                        #Veri Adını oluşturma
        else:
            Features.message("You have to load at least 1 signal", QMessageBox.Warning)
            self.label_1D_DataName.setText("No signal selected")

    def load_folder(self):                                                      #Dosya Yükleme
        self.all_signals, self.load_check = Features.load_folder(self)
        if (self.load_check):
            Features.message(str(len(self.all_signals)) + " signals ready to be processed !", QMessageBox.Information)
            if (len(self.all_signals) > 1):
                self.label_1D_DataName.setText("Multiple Signals")
            else:
                name = self.all_signals[0].split(sep='/')
                self.label_1D_DataName.setText(name[-1])
        else:
            Features.message("You have to load at least 1 signal", QMessageBox.Warning)

    def analyze_signal(self):           #Veriyi Analiz Etme
        self.average, self.entropy, self.kurtosis, self.max_v, self.median, self.min_v, self.skewness, self.standart_dev = Features.check_statistics(self)
        self.wavelet_type, self.wavelet_level = Features.discrete_check_wavelet(self)
        if(self.load_check and (self.average or self.entropy or self.kurtosis or self.max_v or self.median or self.min_v or self.skewness or self.standart_dev)):   #işlenecek müzik olması durumu

            self.analyze_check = True
            col, self.header, self.db_header = Features.init_table(self, len(self.all_signals))
            self.db_matrix = np.zeros((len(self.all_signals), 8 * (self.wavelet_level + 1)))

            for iter in range(0 , len(self.all_signals)):
                print(self.all_signals[iter])
                self.signals.clear()
                self.audio, self.sample = librosa.load(self.all_signals[iter])
                self.signals.append(self.audio)
                self.time = np.arange(0, len(self.audio)) / self.sample

                coeffs = pywt.wavedec(self.audio, self.wavelet_type, level = self.wavelet_level)         #Wavelet Analiz

                for i in range(0 , self.wavelet_level + 1):
                    self.signals.append(coeffs[i])                                                       #Sinyalleri Coef içine atma

                if (iter == 0 and self.wavelet_level < 8):                                               #İlk sinyali yazdırma
                    Features.discrete_plot_signal(self)
                self.db_matrix[iter] = Features.insertTable(self, iter, col)

                self.progressBar.setValue(((iter+1)/len(self.all_signals))*100)
                #level = 3 ise signals içinde  5 (4 analiz edilmiş + 1 source)
        else:
            if not self.load_check :
                Features.message("You have to load at least 1 signal", QMessageBox.Warning)
            else:
                Features.message("You have to pick at least 1 statistic function", QMessageBox.Warning)

    def save_to_db(self):
        if(self.load_check and self.analyze_check):
            w_name = str(self.wavelet_type)
            table_name = "Db_GTZAN_function_" + w_name + "_Degree_" + str(self.wavelet_level)

            Database.create_table(Database.database_name, table_name, self.db_header, "")                                                    # İstatistiksel fonksiyonlar için tablo oluşturma

            for index in range(0, len(self.all_signals)):
                name = self.all_signals[index].split(sep='/')                                                                                #Sinyalin adı
                Database.delete_row(Database.database_name, table_name, name[-1], "")
                Database.add_values_to_table(Database.database_name, table_name, name[-1], self.db_header, self.db_matrix[index], "")       #Db'ye değerleri girme
            Features.message("Your Data Saved Succesfully", QMessageBox.Information)

        else:
            if (self.load_check):
                Features.message("You have to analyze the signals first", QMessageBox.Warning)
            else:
                Features.message("You have to load at least 1 signal", QMessageBox.Warning)


    def run(self):
        wave_types = pywt.wavelist(kind='discrete')
        levels = [1,2,3]
        signals = []
        self.average, self.entropy, self.kurtosis, self.max_v, self.median, self.min_v, self.skewness, self.standart_dev = Features.check_statistics(self)
        table_names = Database.get_table_names(Database.database_name)

        for wave_func in wave_types:
            if wave_func.find("sym") == -1:
                for level in levels:
                    self.wavelet_level = level
                    self.wavelet_type = wave_func
                    col, self.header, self.db_header = Features.init_table(self, len(self.all_signals))
                    self.db_matrix = np.zeros((len(self.all_signals), 8 * (self.wavelet_level + 1)))

                    if not any("Db_GTZAN_function_" + str(wave_func) + "_Degree_" + str(level) in s for s in table_names):

                        for iter in range(0, len(self.all_signals)):
                            self.signals.clear()
                            self.audio, self.sample = librosa.load(self.all_signals[iter])
                            self.signals.append(self.audio)
                            self.time = np.arange(0, len(self.audio)) / self.sample

                            coeffs = pywt.wavedec(self.audio, self.wavelet_type, level=self.wavelet_level)  # wavelet analyze

                            for i in range(0, self.wavelet_level + 1):
                                self.signals.append(coeffs[i])  # adding signals array to coeffs

                            self.db_matrix[iter] = Features.insertTable(self, iter,col)  # level = 3 ise signals içinde  5 (4 analiz edilmiş + 1 kaynak)

                        w_name = str(wave_func)
                        table_name = "Db_GTZAN_function_" + w_name + "_Degree_" + str(level)
                        Database.create_table(Database.database_name, table_name, self.db_header, "")

                        for index in range(0, len(self.all_signals)):
                            name = self.all_signals[index].split(sep='/')  # name of signal
                            Database.delete_row(Database.database_name, table_name, name[-1], "")
                            Database.add_values_to_table(Database.database_name, table_name, name[-1], self.db_header, self.db_matrix[index], "")  # adding db to values
                        print(table_name)
                    else:
                        print("Db_GTZAN_function_" + str(wave_func) + "_Degree_" + str(level) + ' passed')
            else:
                print(wave_func + " passed")