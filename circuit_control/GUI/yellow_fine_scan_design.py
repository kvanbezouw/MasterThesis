# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GUI/yellow_fine_scan_design.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(275, 501)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Form.sizePolicy().hasHeightForWidth())
        Form.setSizePolicy(sizePolicy)
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.comboBox_pd_detection_fine_scan = QtWidgets.QComboBox(Form)
        self.comboBox_pd_detection_fine_scan.setObjectName("comboBox_pd_detection_fine_scan")
        self.comboBox_pd_detection_fine_scan.addItem("")
        self.comboBox_pd_detection_fine_scan.addItem("")
        self.comboBox_pd_detection_fine_scan.addItem("")
        self.comboBox_pd_detection_fine_scan.addItem("")
        self.comboBox_pd_detection_fine_scan.addItem("")
        self.comboBox_pd_detection_fine_scan.addItem("")
        self.comboBox_pd_detection_fine_scan.addItem("")
        self.comboBox_pd_detection_fine_scan.addItem("")
        self.verticalLayout_4.addWidget(self.comboBox_pd_detection_fine_scan)
        self.comboBox_MZ_detection = QtWidgets.QComboBox(Form)
        self.comboBox_MZ_detection.setObjectName("comboBox_MZ_detection")
        self.comboBox_MZ_detection.addItem("")
        self.comboBox_MZ_detection.addItem("")
        self.comboBox_MZ_detection.addItem("")
        self.comboBox_MZ_detection.addItem("")
        self.comboBox_MZ_detection.addItem("")
        self.comboBox_MZ_detection.addItem("")
        self.comboBox_MZ_detection.addItem("")
        self.comboBox_MZ_detection.addItem("")
        self.verticalLayout_4.addWidget(self.comboBox_MZ_detection)
        self.comboBox_test_detection = QtWidgets.QComboBox(Form)
        self.comboBox_test_detection.setObjectName("comboBox_test_detection")
        self.comboBox_test_detection.addItem("")
        self.comboBox_test_detection.addItem("")
        self.comboBox_test_detection.addItem("")
        self.comboBox_test_detection.addItem("")
        self.comboBox_test_detection.addItem("")
        self.comboBox_test_detection.addItem("")
        self.comboBox_test_detection.addItem("")
        self.comboBox_test_detection.addItem("")
        self.verticalLayout_4.addWidget(self.comboBox_test_detection)
        self.lineEdit_sweep_range = QtWidgets.QLineEdit(Form)
        self.lineEdit_sweep_range.setObjectName("lineEdit_sweep_range")
        self.verticalLayout_4.addWidget(self.lineEdit_sweep_range)
        self.textedit_center_wavelength = QtWidgets.QLineEdit(Form)
        self.textedit_center_wavelength.setMaximumSize(QtCore.QSize(100, 16777215))
        self.textedit_center_wavelength.setObjectName("textedit_center_wavelength")
        self.verticalLayout_4.addWidget(self.textedit_center_wavelength)
        self.comboBox_plotting_fine_scan = QtWidgets.QComboBox(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox_plotting_fine_scan.sizePolicy().hasHeightForWidth())
        self.comboBox_plotting_fine_scan.setSizePolicy(sizePolicy)
        self.comboBox_plotting_fine_scan.setObjectName("comboBox_plotting_fine_scan")
        self.comboBox_plotting_fine_scan.addItem("")
        self.comboBox_plotting_fine_scan.addItem("")
        self.comboBox_plotting_fine_scan.addItem("")
        self.comboBox_plotting_fine_scan.addItem("")
        self.comboBox_plotting_fine_scan.addItem("")
        self.comboBox_plotting_fine_scan.addItem("")
        self.comboBox_plotting_fine_scan.addItem("")
        self.comboBox_plotting_fine_scan.addItem("")
        self.comboBox_plotting_fine_scan.addItem("")
        self.comboBox_plotting_fine_scan.addItem("")
        self.verticalLayout_4.addWidget(self.comboBox_plotting_fine_scan)
        self.lineEdit_avg_over_x_steps = QtWidgets.QLineEdit(Form)
        self.lineEdit_avg_over_x_steps.setObjectName("lineEdit_avg_over_x_steps")
        self.verticalLayout_4.addWidget(self.lineEdit_avg_over_x_steps)
        self.ps_scaling = QtWidgets.QLineEdit(Form)
        self.ps_scaling.setObjectName("ps_scaling")
        self.verticalLayout_4.addWidget(self.ps_scaling)
        self.lineEdit_amplification = QtWidgets.QLineEdit(Form)
        self.lineEdit_amplification.setObjectName("lineEdit_amplification")
        self.verticalLayout_4.addWidget(self.lineEdit_amplification)
        self.gridLayout.addLayout(self.verticalLayout_4, 2, 1, 1, 1)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.detection_label = QtWidgets.QLabel(Form)
        self.detection_label.setObjectName("detection_label")
        self.verticalLayout_3.addWidget(self.detection_label)
        self.label = QtWidgets.QLabel(Form)
        self.label.setObjectName("label")
        self.verticalLayout_3.addWidget(self.label)
        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_3.addWidget(self.label_3)
        self.label_5 = QtWidgets.QLabel(Form)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_3.addWidget(self.label_5)
        self.label_wavelength = QtWidgets.QLabel(Form)
        self.label_wavelength.setObjectName("label_wavelength")
        self.verticalLayout_3.addWidget(self.label_wavelength)
        self.plot_label = QtWidgets.QLabel(Form)
        self.plot_label.setObjectName("plot_label")
        self.verticalLayout_3.addWidget(self.plot_label)
        self.label_4 = QtWidgets.QLabel(Form)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_3.addWidget(self.label_4)
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_3.addWidget(self.label_2)
        self.label_6 = QtWidgets.QLabel(Form)
        self.label_6.setObjectName("label_6")
        self.verticalLayout_3.addWidget(self.label_6)
        self.gridLayout.addLayout(self.verticalLayout_3, 2, 0, 1, 1)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.btn_scan = QtWidgets.QPushButton(Form)
        self.btn_scan.setMaximumSize(QtCore.QSize(100, 16777215))
        self.btn_scan.setObjectName("btn_scan")
        self.verticalLayout_2.addWidget(self.btn_scan)
        self.gridLayout.addLayout(self.verticalLayout_2, 0, 0, 1, 1)
        self.btn_fit = QtWidgets.QPushButton(Form)
        self.btn_fit.setMaximumSize(QtCore.QSize(100, 16777215))
        self.btn_fit.setObjectName("btn_fit")
        self.gridLayout.addWidget(self.btn_fit, 0, 1, 1, 1)
        self.verticalLayout_9 = QtWidgets.QVBoxLayout()
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_9.addItem(spacerItem)
        self.btn_save_fit = QtWidgets.QPushButton(Form)
        self.btn_save_fit.setMinimumSize(QtCore.QSize(100, 0))
        self.btn_save_fit.setMaximumSize(QtCore.QSize(100, 16777215))
        self.btn_save_fit.setObjectName("btn_save_fit")
        self.verticalLayout_9.addWidget(self.btn_save_fit)
        self.btn_save_power_meas = QtWidgets.QPushButton(Form)
        self.btn_save_power_meas.setMinimumSize(QtCore.QSize(100, 0))
        self.btn_save_power_meas.setMaximumSize(QtCore.QSize(100, 16777215))
        self.btn_save_power_meas.setObjectName("btn_save_power_meas")
        self.verticalLayout_9.addWidget(self.btn_save_power_meas)
        self.btn_save = QtWidgets.QPushButton(Form)
        self.btn_save.setMinimumSize(QtCore.QSize(100, 0))
        self.btn_save.setMaximumSize(QtCore.QSize(100, 16777215))
        self.btn_save.setObjectName("btn_save")
        self.verticalLayout_9.addWidget(self.btn_save)
        self.gridLayout.addLayout(self.verticalLayout_9, 4, 1, 1, 1)
        self.verticalLayout_8 = QtWidgets.QVBoxLayout()
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.label_save_note = QtWidgets.QLabel(Form)
        self.label_save_note.setObjectName("label_save_note")
        self.verticalLayout_8.addWidget(self.label_save_note)
        self.info_text = QtWidgets.QTextEdit(Form)
        self.info_text.setMinimumSize(QtCore.QSize(50, 0))
        self.info_text.setMaximumSize(QtCore.QSize(100, 400))
        self.info_text.setObjectName("info_text")
        self.verticalLayout_8.addWidget(self.info_text)
        self.gridLayout.addLayout(self.verticalLayout_8, 4, 0, 1, 1)
        self.btn_power_measurement = QtWidgets.QPushButton(Form)
        self.btn_power_measurement.setObjectName("btn_power_measurement")
        self.gridLayout.addWidget(self.btn_power_measurement, 1, 0, 1, 1)
        self.btn_fine_scan_loop = QtWidgets.QPushButton(Form)
        self.btn_fine_scan_loop.setObjectName("btn_fine_scan_loop")
        self.gridLayout.addWidget(self.btn_fine_scan_loop, 1, 1, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.comboBox_pd_detection_fine_scan.setItemText(0, _translate("Form", "ai0"))
        self.comboBox_pd_detection_fine_scan.setItemText(1, _translate("Form", "ai1"))
        self.comboBox_pd_detection_fine_scan.setItemText(2, _translate("Form", "ai2"))
        self.comboBox_pd_detection_fine_scan.setItemText(3, _translate("Form", "ai3"))
        self.comboBox_pd_detection_fine_scan.setItemText(4, _translate("Form", "ai4"))
        self.comboBox_pd_detection_fine_scan.setItemText(5, _translate("Form", "ai5"))
        self.comboBox_pd_detection_fine_scan.setItemText(6, _translate("Form", "ai6"))
        self.comboBox_pd_detection_fine_scan.setItemText(7, _translate("Form", "ai7"))
        self.comboBox_MZ_detection.setItemText(0, _translate("Form", "ai0"))
        self.comboBox_MZ_detection.setItemText(1, _translate("Form", "ai1"))
        self.comboBox_MZ_detection.setItemText(2, _translate("Form", "ai2"))
        self.comboBox_MZ_detection.setItemText(3, _translate("Form", "ai3"))
        self.comboBox_MZ_detection.setItemText(4, _translate("Form", "ai4"))
        self.comboBox_MZ_detection.setItemText(5, _translate("Form", "ai5"))
        self.comboBox_MZ_detection.setItemText(6, _translate("Form", "ai6"))
        self.comboBox_MZ_detection.setItemText(7, _translate("Form", "ai7"))
        self.comboBox_test_detection.setItemText(0, _translate("Form", "ai0"))
        self.comboBox_test_detection.setItemText(1, _translate("Form", "ai1"))
        self.comboBox_test_detection.setItemText(2, _translate("Form", "ai2"))
        self.comboBox_test_detection.setItemText(3, _translate("Form", "ai3"))
        self.comboBox_test_detection.setItemText(4, _translate("Form", "ai4"))
        self.comboBox_test_detection.setItemText(5, _translate("Form", "ai5"))
        self.comboBox_test_detection.setItemText(6, _translate("Form", "ai6"))
        self.comboBox_test_detection.setItemText(7, _translate("Form", "ai7"))
        self.lineEdit_sweep_range.setText(_translate("Form", "2"))
        self.textedit_center_wavelength.setText(_translate("Form", "1520"))
        self.comboBox_plotting_fine_scan.setItemText(0, _translate("Form", "Tapered vs detuning"))
        self.comboBox_plotting_fine_scan.setItemText(1, _translate("Form", "PDdisk vs V"))
        self.comboBox_plotting_fine_scan.setItemText(2, _translate("Form", "Det vs V"))
        self.comboBox_plotting_fine_scan.setItemText(3, _translate("Form", "MZ vs V"))
        self.comboBox_plotting_fine_scan.setItemText(4, _translate("Form", "Test vs V"))
        self.comboBox_plotting_fine_scan.setItemText(5, _translate("Form", "power_scan_first"))
        self.comboBox_plotting_fine_scan.setItemText(6, _translate("Form", "power_scan_middle"))
        self.comboBox_plotting_fine_scan.setItemText(7, _translate("Form", "power_scan_last"))
        self.comboBox_plotting_fine_scan.setItemText(8, _translate("Form", "power_scan_first_run_2"))
        self.comboBox_plotting_fine_scan.setItemText(9, _translate("Form", "power_scan_ref"))
        self.lineEdit_avg_over_x_steps.setText(_translate("Form", "4"))
        self.ps_scaling.setText(_translate("Form", "1"))
        self.lineEdit_amplification.setText(_translate("Form", "1"))
        self.detection_label.setText(_translate("Form", "Tap. fib.  detection"))
        self.label.setText(_translate("Form", "Mach Zend. detection"))
        self.label_3.setText(_translate("Form", "Test detection"))
        self.label_5.setText(_translate("Form", "Sweep range [V]"))
        self.label_wavelength.setText(_translate("Form", "Wavelength [nm]"))
        self.plot_label.setText(_translate("Form", "Plot"))
        self.label_4.setText(_translate("Form", "Avg over # runs"))
        self.label_2.setText(_translate("Form", "Power scaling"))
        self.label_6.setText(_translate("Form", "Amplification"))
        self.btn_scan.setText(_translate("Form", "scan"))
        self.btn_fit.setText(_translate("Form", "Fit"))
        self.btn_save_fit.setText(_translate("Form", "save fit"))
        self.btn_save_power_meas.setText(_translate("Form", "save power meas"))
        self.btn_save.setText(_translate("Form", "save"))
        self.label_save_note.setText(_translate("Form", "save note:"))
        self.btn_power_measurement.setText(_translate("Form", "Power Meas."))
        self.btn_fine_scan_loop.setText(_translate("Form", "Fine scan loop"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
