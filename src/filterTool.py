# Imports
#

from Lib.random import random, seed

# Qt Modules
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QFileDialog,QWidget, QGridLayout,QPushButton, QApplication, QLabel, QCheckBox
from src.ui.filterToolGUI_v2 import Ui_Form

# Matplotlib Modules
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# Python Modules
import numpy as np
import scipy.signal as ss
from enum import Enum
import array as arr

# SymPy modules

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sympy import I

# ?
import math

# My Own Modules
from src.backend.Stages.SimpleHs import SimpleHs, w_domain, w_domain_blank
from src.backend.Filter.Filter import FilterData

from src.backend.Filter.LowPass import LowPass
from src.backend.Filter.HighPass import HighPass
from src.backend.Filter.BandPass import BandPass
from src.backend.Filter.BandReject import BandReject
from src.backend.Filter.GroupDelay import GroupDelay

#from src.backend.Approx.Gauss import Gauss
from src.backend.Approx.Legendre import Legendre
from src.backend.Approx.Butterworth import Butterworth
from src.backend.Approx.Gauss import Gauss

DEBUG = False

t_domain = np.linspace(0,0.001,10000)
test = [1]*5
SIG_FIGURE = 6

class filterType(Enum):
    LP=0
    HP=1
    BP=2
    BS=3
    GD=4

class approxTypeALL(Enum):
    Butterworth = 0
    Gauss = 1
    Legendre = 2

class FilterTool(QWidget,Ui_Form):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)
        self.setWindowTitle("TP GRUPAL 4.5 - TEORÍA DE CIRCUITOS")
        self.setWindowIcon(QtGui.QIcon('py.png'))

        seed()

        self.__setConstants()
        self.__init_graphs()
        self.__setCallbacks()
        self.__manageDebug()

        self.__showHideState_CreateNewStage()
        self.__refreshFilterMakerGraphs()
        self.__refreshStagesGraphs()

    def __createFilter(self):
        if self.checkBox_Test.isChecked() == False:
            gain = self.doubleSpinBox_Gain.value()
            Aa = self.doubleSpinBox_Aa.value()
            Ap = self.doubleSpinBox_Ap.value()
            fa_minus = self.doubleSpinBox_fa_minus.value()
            fp_minus = self.doubleSpinBox_fp_minus.value()
            fa_plus = self.doubleSpinBox_fa_plus.value()
            fp_plus = self.doubleSpinBox_fp_plus.value()
            ft = self.doubleSpinBox_ft.value()
            tol = self.doubleSpinBox_Tolerance.value()
            gDelay = self.doubleSpinBox_GroupDelay.value()

            denorm = self.doubleSpinBox_denorm.value()
            if self.checkBox_Nmin.isChecked():
                Nmin = self.spinBox_Nmin.value()
            else:
                Nmin = None
            if self.checkBox_Nmax.isChecked():
                Nmax = self.spinBox_Nmax.value()
            else:
                Nmax = None
            if self.checkBox_Qmax.isChecked():
                Qmax = self.doubleSpinBox_Qmax.value()
            else:
                Qmax = None

            Name = self.lineEdit_name.text()

            filtertype = self.comboBox_filterType.currentIndex()
            if filtertype == filterType.LP.value:
                name_filterType = "LowPass"
                newFilter = LowPass(Aa,fa_minus,Ap,fp_minus,gain,Nmax,Nmin,Qmax,denorm)
            elif filtertype == filterType.HP.value:
                name_filterType = "HighPass"
                newFilter = HighPass(Aa,fa_minus,Ap,fp_minus,gain,Nmax,Nmin,Qmax,denorm)
            elif filtertype == filterType.BP.value:
                name_filterType = "BandPass"
                newFilter = BandPass(Aa,fa_minus,fa_plus,Ap,fp_minus,fp_plus,gain,Nmax,Nmin,Qmax,denorm)
            elif filtertype == filterType.BS.value:
                name_filterType = "BandStop"
                newFilter = BandReject(Aa,fa_minus,fa_plus,Ap,fp_minus,fp_plus,gain,Nmax,Nmin,Qmax,denorm)
            else:
                name_filterType = "GroupDelay"
                newFilter = GroupDelay(ft,gDelay,tol,gain,Nmax,Nmin,Qmax,denorm)

            valid,message = newFilter.validate()
            if not valid:
                self.__error_message(message)
                return

            approxtype = self.comboBox_approximation.currentIndex()
            if approxtype == approxTypeALL.Gauss.value:
                name_approxType = "Gauss"
                newApprox = Gauss(newFilter)
            elif approxtype == approxTypeALL.Butterworth.value:
                name_approxType = "Butterworth"
                newApprox = Butterworth(newFilter)
            elif approxtype == approxTypeALL.Legendre.value:
                name_approxType = "Legendre"
                newApprox = Legendre(newFilter)
            else:
                self.__error_message("Invalid Approximation Type")
                return

            #TODO VERIFICAR Q LA APROX SEA VALIDA

            #w,mag,pha = newApprox.calculate()
            newTransFunc = newApprox.get_MagAndPhaseWithGain()
            orden = len(newApprox.get_Qs())
            fullname = Name + " - " + name_filterType + " - " + name_approxType + " - ORDER: " + str(orden)
            self.myFilters.append([fullname,newApprox,newTransFunc,True])
            self.comboBox_YourFilters.addItem(fullname)
            self.comboBox_SelectYourFilter.addItem(fullname)
        else:
            newApprox = myFilterTest()
            fullname = "TEST" + str(self.testvar1)
            self.testvar1 += 1
            w,mag,pha = newApprox.calculate()
            newTransFunc = [w,mag,pha]
            self.myFilters.append([fullname,newApprox,newTransFunc,True])
            self.comboBox_YourFilters.addItem(fullname)
            self.comboBox_SelectYourFilter.addItem(fullname)

        self.__refreshFilterMakerGraphs()

    def __setCallbacks(self):
        self.pushButton_CanceNewStage.clicked.connect(self.__cancelNewStage)
        self.pushButton_CreateNewStage.clicked.connect(self.__crateNewStage)
        self.pushButton_1stOrderPole.clicked.connect(self.__clicked_1stOrderPole)
        self.pushButton_SecondOrderPoles.clicked.connect(self.__clicked_secondOrderPoles)
        self.pushButton_ComplexPoles.clicked.connect(self.__clicked_complexPoles)
        self.pushButton_RealPoles.clicked.connect(self.__clicked_realPoles)
        self.pushButton_SelectRealPole.clicked.connect(self.__selectRealPole)
        self.pushButton_SelectRealPoles.clicked.connect(self.__selectRealPoles)
        self.pushButton_SelectComplexPoles.clicked.connect(self.__selectComplexPoles)
        self.pushButton_FINISH1stOrder.clicked.connect(self.__finish1stOrderPole)
        self.pushButton_AddZero_1stOrder.clicked.connect(self.__clicked_addZero)
        self.pushButton_FINISH2ndOrder.clicked.connect(self.__finish2ndOrderPole)
        self.pushButton_Add1Zero.clicked.connect(self.__clicked_add1Zero)
        self.pushButton_Add2Zeros.clicked.connect(self.__clicked_add2Zeros)
        self.pushButton_ComplexConjZeros.clicked.connect(self.__clicked_complexZeros)
        self.pushButton_RealZeros.clicked.connect(self.__clicked_realZeros)
        self.pushButton_SelectRealZero.clicked.connect(self.__finishRealZero)
        self.pushButton_SelectRealZeros.clicked.connect(self.__finishRealZeros)
        self.pushButton_SelectComplexZeros.clicked.connect(self.__finishComplexZeros)

        self.comboBox_filterType.currentIndexChanged.connect(self.__showAndHideParameters)
        self.__showAndHideParameters()

        self.pushButton_createFilter.clicked.connect(self.__createFilter)

        self.comboBox_YourFilters.currentIndexChanged.connect(self.__indexChanged_yourFilters)
        self.checkBox_VisibleFilter.clicked.connect(self.__clicked_visibleFilter)
        self.__indexChanged_yourFilters()
        self.pushButton_EraseFilter.clicked.connect(self.__clicked_EraseFilter)

        self.checkBox_SelectedFilterVisible.clicked.connect(self.__refreshStagesGraphs)
        #####################################################################

        self.comboBox_SelectYourFilter.currentIndexChanged.connect(self.__indexChanged_SelectYourFilter)
        self.__indexChanged_SelectYourFilter()
        self.comboBox_YourStages.currentIndexChanged.connect(self.__indexChanged_SelectYourStage)
        self.pushButton_StageUpdateGain.clicked.connect(self.__updateStageGain)
        self.pushButton_DeleteStage.clicked.connect(self.__deleteCurrentStage)
        self.checkBox_SelectedStageVisible.clicked.connect(self.__changeStageVisibility)
        self.checkBox_VisibleStagesSubtotalVisible.clicked.connect(self.__refreshStagesGraphs)
        self.checkBox_CratedStagesSubtotalVisible.clicked.connect(self.__refreshStagesGraphs)
        self.pushButton_showAllStages.clicked.connect(self.__showAllStages)
        self.pushButton_hideAllStages.clicked.connect(self.__hideAllStages)
        #####################################################################
        self.pushButton_TEST.clicked.connect(self.__test)
        self.pushButton_TEST_2.clicked.connect(self.__test2)

    def __indexChanged_yourFilters(self):
        if self.comboBox_YourFilters.currentIndex() == 0:
            self.checkBox_VisibleFilter.setDisabled(True)
        else:
            self.checkBox_VisibleFilter.setDisabled(False)
            i = self.comboBox_YourFilters.currentIndex()-1
            self.checkBox_VisibleFilter.setChecked(self.myFilters[i][3])

    def __indexChanged_SelectYourFilter(self):
        #TODO ARREGLAR POLOS REALES TOMADOS COMO COMPLEJOS ( Y CONJUGADOS TOMADOS DIFERENTES )!!!
        #TODO AGREGAR BARRA DESPLAZADORA
        self.comboBox_YourStages.setCurrentIndex(0)
        self.__cleanThisComboBox(self.comboBox_YourStages)
        self.__indexChanged_SelectYourStage()
        self.__cleanStagesWidgets()
        self.__cancelNewStage()
        self.sos = []
        if self.comboBox_SelectYourFilter.currentIndex() == 0:
            self.checkBox_SelectedFilterVisible.setDisabled(True)
            self.checkBox_SelectedFilterVisible.setChecked(False)
            self.label_SelectedFilterGain.setText("")
            self.label_SelectedFilterK.setText("")
        else:
            self.checkBox_SelectedFilterVisible.setDisabled(False)
            self.checkBox_SelectedFilterVisible.setChecked(True)
            i = self.comboBox_SelectYourFilter.currentIndex()-1
            self.__getAndOrderPolesAndZeros(i)
            self.label_SelectedFilterGain.setText(str(self.myFilters[i][1].get_Gain()))
            self.__refreshStagesGraphs()
            self.__updateStagesAvailable()
            #DO THINGS

    #

    def __hideAllStages(self):
        for i in self.sos:
            i[1] = False
        self.__indexChanged_SelectYourStage()
        self.__refreshStagesGraphs()

    def __showAllStages(self):
        for i in self.sos:
            i[1] = True
        self.__indexChanged_SelectYourStage()
        self.__refreshStagesGraphs()

    #

    def __indexChanged_SelectYourStage(self):
        if self.comboBox_YourStages.currentIndex() == 0:
            self.doubleSpinBox_StageGain.setDisabled(True)
            self.doubleSpinBox_StageGain.setValue(0)
            self.checkBox_SelectedStageVisible.setDisabled(True)
            self.checkBox_SelectedStageVisible.setChecked(False)
            self.label_StageK.setText("")
            self.label_SelectedStageOrder.setText("")
            self.label_SelectedStagefo.setText("")
            self.label_SelectedStageQ.setText("")
            self.pushButton_StageUpdateGain.setDisabled(True)
            self.pushButton_DeleteStage.setDisabled(True)
            self.label_NUMERATOR.setText("")
            self.label_DENOMINATOR.setText("")
        else:
            i = self.comboBox_YourStages.currentIndex() - 1
            sos = self.sos[i][0]
            K,order,f0,Q,gain = sos.getData()
            visible = self.sos[i][1]
            num = sos.getNumerator()
            den = sos.getDenominator()

            self.doubleSpinBox_StageGain.setDisabled(False)
            self.doubleSpinBox_StageGain.setValue(gain)
            self.checkBox_SelectedStageVisible.setDisabled(False)
            self.checkBox_SelectedStageVisible.setChecked(visible)
            self.label_StageK.setText(str(K))
            self.label_SelectedStageOrder.setText(str(order))
            self.label_SelectedStagefo.setText(str(f0))
            self.label_SelectedStageQ.setText(str(Q))
            self.pushButton_StageUpdateGain.setDisabled(False)
            self.pushButton_DeleteStage.setDisabled(False)
            self.label_NUMERATOR.setText(num)
            self.label_DENOMINATOR.setText(den)

    def __updateStageGain(self):
        i = self.comboBox_YourStages.currentIndex()
        if i != 0:
            sos = self.sos[i-1][0]
            sos.updateGain(self.doubleSpinBox_StageGain.value())
            self.__indexChanged_SelectYourStage()
            self.__refreshStagesGraphs()
        else:
            self.__error_message("This shouldn't be happening. It seems Alex forgot to disable this button...")

    def __changeStageVisibility(self):
        i = self.comboBox_YourStages.currentIndex()
        if i != 0:
            self.sos[i-1][1] = self.checkBox_SelectedStageVisible.isChecked()
            self.__refreshStagesGraphs()

    def __deleteCurrentStage(self):
        if self.comboBox_YourStages.currentIndex() != 0:
            i = self.comboBox_YourStages.currentIndex() -1
            self.comboBox_YourStages.removeItem(i+1)
            data = self.sos[i][2]
            self.sos.pop(i)

            #[self.polosIndex,
            # self.polosAreComplex,
            # self.cerosIndex,
            # self.cerosAreComples,
            # self.cerosRequired]

            polosIndex,polosAreComplex,cerosIndex,cerosAreComplex,cerosRequired = data

            #######################################################################

            if  polosAreComplex:
                #polosArray = self.ComplexPoles
                polosUsedArray = self.ComplexPolesUsed
                polosWidgetArray = self.ComplexPolesWidgets
                polesWidgetConstant = 5
                polesComboBoxes = [self.comboBox_SelectComplexPoles]
                #polosName = "C"
            else:
                #polosArray = self.RealPoles
                polosUsedArray = self.RealPolesUsed
                polosWidgetArray = self.RealPolesWidgets
                polesWidgetConstant = 3
                polesComboBoxes = [self.comboBox_SelectRealPole,
                                   self.comboBox_Select1stRealPole,
                                   self.comboBox_Select2ndRealPole]
                #polosName = "R"
            if cerosAreComplex:
                #cerosArray = self.ComplexZeros
                cerosUsedArray = self.ComplexZerosUsed
                cerosWidgetArray = self.ComplexZerosWidgets
                cerosWidgetConstant = 3
                cerosComboBoxes = [self.comboBox_SelectComplexZeros]
                #cerosName = "C"
            else:
                #cerosArray = self.RealZeros
                cerosUsedArray = self.RealZerosUsed
                cerosWidgetArray = self.RealZerosWidgets
                cerosWidgetConstant = 2
                cerosComboBoxes = [self.comboBox_SelectRealZero,
                                   self.comboBox_Select1stRealZero,
                                   self.comboBox_Select2ndRealZero]
                #cerosName = "R"

            for i in polosIndex:
                for combo in polesComboBoxes:
                    t = combo.itemText(i)
                    t=t.replace('USED - ','')
                    combo.setItemText(i,t)
                polosUsedArray[i-1] = False
                polosWidgetArray[polesWidgetConstant*i-1].setChecked(False)
            if cerosRequired:
                for i in cerosIndex:
                    for combo in cerosComboBoxes:
                        t = combo.itemText(i)
                        t=t.replace('USED - ', '')
                        combo.setItemText(i, t)
                    cerosUsedArray[i - 1] = False
                    cerosWidgetArray[cerosWidgetConstant * i - 1].setChecked(False)

            ###############################################################

            self.__refreshStagesGraphs()
            self.comboBox_YourStages.setCurrentIndex(0)
            self.__indexChanged_SelectYourStage()

    def __cleanThisComboBox(self,comboBox):
        n = comboBox.count()
        for i in reversed(range(n-1)):
            comboBox.removeItem(i+1)

    def __cleanStagesWidgets(self):
        self.ComplexPoles = []
        self.RealPoles = []
        self.ComplexZeros = []
        self.RealZeros = []
        self.ComplexPolesUsed = []
        self.RealPolesUsed = []
        self.ComplexZerosUsed = []
        self.RealZerosUsed = []
        self.__cleanThisGridLayout(self.gridLayout_ComplexPoles,self.ComplexPolesWidgets)
        self.__cleanThisGridLayout(self.gridLayout_RealPoles, self.RealPolesWidgets)
        self.__cleanThisGridLayout(self.gridLayout_ComplesZeros, self.ComplexZerosWidgets)
        self.__cleanThisGridLayout(self.gridLayout_RealZeros, self.RealZerosWidgets)
        self.__cleanThisComboBox(self.comboBox_SelectComplexPoles)
        self.__cleanThisComboBox(self.comboBox_SelectRealPole)
        self.__cleanThisComboBox(self.comboBox_Select1stRealPole)
        self.__cleanThisComboBox(self.comboBox_Select2ndRealPole)
        self.__cleanThisComboBox(self.comboBox_SelectComplexZeros)
        self.__cleanThisComboBox(self.comboBox_Select1stRealZero)
        self.__cleanThisComboBox(self.comboBox_Select2ndRealZero)
        self.__cleanThisComboBox(self.comboBox_SelectRealZero)

    def __cleanThisGridLayout(self,layout: QGridLayout,widgets):
        if DEBUG:
            print(layout.columnCount())
            print(layout.rowCount())
        #x = layout.columnCount()
        #y = layout.rowCount()
        for i in widgets:
            layout.removeWidget(i)
            i.deleteLater()
            del i
        widgets.clear()

    def __updateStagesAvailable(self):
        for i in range(len(self.ComplexPoles)):
            reZ_ ='{:.2f}'.format(np.real(self.ComplexPoles[i]))
            reZ = QLabel(reZ_)
            #reZ.setMaximumWidth(50)
            self.gridLayout_ComplexPoles.addWidget(reZ,i+1,0)
            self.ComplexPolesWidgets.append(reZ)

            imZ_ = '{:.2f}'.format(np.imag(self.ComplexPoles[i]))
            imZ = QLabel(imZ_)
            #imZ.setMaximumWidth(50)
            self.gridLayout_ComplexPoles.addWidget(imZ,i+1,1)
            self.ComplexPolesWidgets.append(imZ)

            foZ_ = '{:.2f}'.format(np.absolute(self.ComplexPoles[i])/(2*np.pi))
            foZ = QLabel(foZ_)
            #foZ.setMaximumWidth(50)
            self.gridLayout_ComplexPoles.addWidget(foZ,i+1,2)
            self.ComplexPolesWidgets.append(foZ)

            if np.real(self.ComplexPoles[i]) == 0:
                QZ_ = "inf"
            else:
                QZ_ = '{:.2f}'.format(np.abs(self.ComplexPoles[i])/(2*np.abs(np.real(self.ComplexPoles[i]))))
            QZ = QLabel(QZ_)
            self.gridLayout_ComplexPoles.addWidget(QZ,i+1,3)
            self.ComplexPolesWidgets.append(QZ)

            visible = QCheckBox()
            visible.setMaximumWidth(16)
            visible.setChecked(False)
            visible.setDisabled(True)
            self.gridLayout_ComplexPoles.addWidget(visible,i+1,4)
            self.ComplexPolesWidgets.append(visible)

            self.comboBox_SelectComplexPoles.addItem('Z={0}+-{1}i - f0={2} - Q={3}'.format(reZ_,imZ_,foZ_,QZ_))

        for i in range(len(self.RealPoles)):
            reZ_ = '{:.2f}'.format(np.real(self.RealPoles[i]))
            reZ = QLabel(reZ_)
            #reZ.setMaximumWidth(50)
            self.gridLayout_RealPoles.addWidget(reZ,i+1,0)
            self.RealPolesWidgets.append(reZ)

            foZ_ ='{:.2f}'.format(np.absolute(self.RealPoles[i])/(2*np.pi))
            foZ = QLabel(foZ_)
            #foZ.setMaximumWidth(50)
            self.gridLayout_RealPoles.addWidget(foZ,i+1,1)
            self.RealPolesWidgets.append(foZ)

            visible = QCheckBox()
            visible.setMaximumWidth(16)
            visible.setChecked(False)
            visible.setDisabled(True)
            self.gridLayout_RealPoles.addWidget(visible,i+1,2)
            self.RealPolesWidgets.append(visible)

            self.comboBox_SelectRealPole.addItem('Z={0} - f0={1}'.format(reZ_, foZ_))
            self.comboBox_Select1stRealPole.addItem('Z={0} - f0={1}'.format(reZ_, foZ_))
            self.comboBox_Select2ndRealPole.addItem('Z={0} - f0={1}'.format(reZ_, foZ_))

        for i in range(len(self.ComplexZeros)):
            reZ_ = '{:.2f}'.format(np.real(self.ComplexZeros[i]))
            reZ = QLabel(reZ_)
            #reZ.setMaximumWidth(50)
            self.gridLayout_ComplesZeros.addWidget(reZ,i+1,0)
            self.ComplexZerosWidgets.append(reZ)

            imZ_ = '{:.2f}'.format(np.imag(self.ComplexZeros[i]))
            imZ = QLabel(imZ_)
            #imZ.setMaximumWidth(50)
            self.gridLayout_ComplesZeros.addWidget(imZ,i+1,1)
            self.ComplexZerosWidgets.append(imZ)

            visible = QCheckBox()
            visible.setMaximumWidth(16)
            visible.setChecked(False)
            visible.setDisabled(True)
            self.gridLayout_ComplesZeros.addWidget(visible,i+1,2)
            self.ComplexZerosWidgets.append(visible)

            self.comboBox_SelectComplexZeros.addItem('Z={0}+-{1}i'.format(reZ_, imZ_))

        for i in range(len(self.RealZeros)):
            reZ_ = '{:.2f}'.format(np.real(self.RealZeros[i]))
            reZ = QLabel(reZ_)
            #reZ.setMaximumWidth(50)
            self.gridLayout_RealZeros.addWidget(reZ,i+1,0)
            self.RealZerosWidgets.append(reZ)

            visible = QCheckBox()
            visible.setMaximumWidth(16)
            visible.setChecked(False)
            visible.setDisabled(True)
            self.gridLayout_RealZeros.addWidget(visible,i+1,1)
            self.RealZerosWidgets.append(visible)

            self.comboBox_SelectRealZero.addItem('Z={0}'.format(reZ_))
            self.comboBox_Select1stRealZero.addItem('Z={0}'.format(reZ_))
            self.comboBox_Select2ndRealZero.addItem('Z={0}'.format(reZ_))

    def __getAndOrderPolesAndZeros(self,i):
        z, p, Gk = self.myFilters[i][1].get_zpGk()
        newZ = []
        newP = []
        for i in z:
            re = np.real(i)
            im = np.imag(i)
            if re != 0:
                newRe = round(re, SIG_FIGURE - int(math.floor(math.log10(abs(re)))) - 1)
            else:
                newRe = 0
            if im != 0:
                newIm = round(im, SIG_FIGURE - int(math.floor(math.log10(abs(im)))) - 1)
            else:
                newIm = 0
            newZ.append(newRe + 1j * newIm)
        for i in p:
            re = np.real(i)
            im = np.imag(i)
            if re != 0:
                newRe = round(re, SIG_FIGURE - int(math.floor(math.log10(abs(re)))) - 1)
            else:
                newRe = 0
            if im != 0:
                newIm = round(im, SIG_FIGURE - int(math.floor(math.log10(abs(im)))) - 1)
            else:
                newIm = 0
            newP.append(newRe + 1j * newIm)

        if DEBUG:
            print(z)
            print(newZ)
            print(p)
            print(newP)
        self.label_SelectedFilterK.setText(str(Gk))
        for i in newZ:
            if np.imag(i) == 0:
                self.RealZeros.append(i)
                self.RealZerosUsed.append(False)
            else:
                if (i in self.ComplexZeros) or (np.conjugate(i) in self.ComplexZeros):
                    print("Found z="+str(i))
                else:
                    self.ComplexZeros.append(i)
                    self.ComplexZerosUsed.append(False)

        for i in newP:
            if np.imag(i) == 0:
                self.RealPoles.append(i)
                self.RealPolesUsed.append(False)
            else:
                if i in self.ComplexPoles or np.conjugate(i) in self.ComplexPoles:
                    print("Found p="+str(i))
                else:
                    self.ComplexPoles.append(i)
                    self.ComplexPolesUsed.append(False)

    def __refreshStagesGraphs(self):
        subtotal_created_k = 1
        subtotal_created_gain = w_domain_blank
        subtotal_created_phase = w_domain_blank
        subtotal_visible_k = 1
        subtotal_visible_gain = w_domain_blank
        subtotal_visible_phase = w_domain_blank
        self.__cleanStagesGraphs()
        if self.checkBox_SelectedFilterVisible.isChecked() and self.comboBox_SelectYourFilter.currentIndex() > 0:
            i = self.comboBox_SelectYourFilter.currentIndex() - 1
            self.__addGraphicsForStages(self.myFilters[i][0]+" - Desired Filter",self.myFilters[i][2])
        for i in self.sos:
            Hs = i[0].getHs()
            K = i[0].getK()
            subtotal_created_k *= K
            subtotal_created_gain = np.add(subtotal_created_gain,Hs[1])
            subtotal_created_phase = np.add(subtotal_created_phase,Hs[2])
            if i[1]:
                subtotal_visible_k *= K
                subtotal_visible_gain = np.add(subtotal_visible_gain,Hs[1])
                subtotal_visible_phase = np.add(subtotal_visible_phase,Hs[2])
                self.__addGraphicsForStages(i[3],Hs)

        if self.checkBox_CratedStagesSubtotalVisible.isChecked():
            self.__addGraphicsForStages("SUBTOTAL (Created Stages)",
                                        [w_domain,subtotal_created_gain,subtotal_created_phase])
        self.label_CratedStagesSubtotalK.setText(str(subtotal_created_k))

        if self.checkBox_VisibleStagesSubtotalVisible.isChecked():
            self.__addGraphicsForStages("SUBTOTAL (Visible Stages)",
                                        [w_domain,subtotal_visible_gain,subtotal_visible_phase])
        self.label_VisibleStagesSubtotalK.setText(str(subtotal_visible_k))

    def __cleanStagesGraphs(self):
        self.axis_StagesGain.clear()
        self.axis_StagesGain.grid()
        self.canvas_StagesGain.draw()
        self.axis_StagesGain.set_title("Gain Plot")
        self.axis_StagesGain.set_xlabel("Frequency [Hz]")
        self.axis_StagesGain.set_ylabel("Gain [dB]")

        self.axis_StagesPhase.clear()
        self.axis_StagesPhase.grid()
        self.canvas_StagesPhase.draw()
        self.axis_StagesPhase.set_title("Phase Plot")
        self.axis_StagesPhase.set_xlabel("Frequency [Hz]")
        self.axis_StagesPhase.set_ylabel("Phase [deg]")

    def __addGraphicsForStages(self,name, transfunc):
        self.axis_StagesGain.semilogx(transfunc[0]/(2*np.pi),transfunc[1],label=name)
        self.axis_StagesGain.legend()
        self.canvas_StagesGain.draw()

        self.axis_StagesPhase.semilogx(transfunc[0]/(2*np.pi),transfunc[2],label=name)
        self.axis_StagesPhase.legend()
        self.canvas_StagesPhase.draw()

    def __clicked_visibleFilter(self):
        if self.comboBox_YourFilters.currentIndex() > 0:
            i = self.comboBox_YourFilters.currentIndex()-1
            self.myFilters[i][3] = self.checkBox_VisibleFilter.isChecked()
            self.__refreshFilterMakerGraphs()

    def __clicked_EraseFilter(self):
        if self.comboBox_YourFilters.currentIndex() > 0:
            i = self.comboBox_YourFilters.currentIndex()-1
            self.myFilters.pop(i)
            self.comboBox_YourFilters.removeItem(i+1)
            self.comboBox_SelectYourFilter.removeItem(i+1)
            self.__refreshFilterMakerGraphs()
            self.comboBox_YourFilters.setCurrentIndex(0)
            self.__indexChanged_yourFilters()
            #self.comboBox_SelectYourFilter.setCurrentIndex(0)

    def __init_graphs(self):
        self.figure_Magnitude = Figure()
        self.canvas_Magnitude = FigureCanvas(self.figure_Magnitude)
        self.index_Magnitude = self.stackedWidget_Magnitude.addWidget(self.canvas_Magnitude)
        self.stackedWidget_Magnitude.setCurrentIndex(self.index_Magnitude)
        self.toolbar_Magnitude = NavigationToolbar(self.canvas_Magnitude,self)
        self.horizontalLayout_Magnitude.addWidget(self.toolbar_Magnitude)
        self.axis_Magnitude = self.figure_Magnitude.add_subplot()

        self.figure_Attenuation = Figure()
        self.canvas_Attenuation = FigureCanvas(self.figure_Attenuation)
        self.index_Attenuation = self.stackedWidget_Attenuation.addWidget(self.canvas_Attenuation)
        self.stackedWidget_Attenuation.setCurrentIndex(self.index_Attenuation)
        self.toolbar_Attenuation = NavigationToolbar(self.canvas_Attenuation,self)
        self.horizontalLayout_Attenuation.addWidget(self.toolbar_Attenuation)
        self.axis_Attenuation = self.figure_Attenuation.add_subplot()

        self.figure_NormalizedAttenuation = Figure()
        self.canvas_NormalizedAttenuation = FigureCanvas(self.figure_NormalizedAttenuation)
        self.index_NormalizedAttenuation = self.stackedWidget_NormalizedAttenuation.addWidget(self.canvas_NormalizedAttenuation)
        self.stackedWidget_NormalizedAttenuation.setCurrentIndex(self.index_NormalizedAttenuation)
        self.toolbar_NormalizedAttenuation = NavigationToolbar(self.canvas_NormalizedAttenuation,self)
        self.horizontalLayout_NormalizedAttenuation.addWidget(self.toolbar_NormalizedAttenuation)
        self.axis_NormalizedAttenuation = self.figure_NormalizedAttenuation.add_subplot()

        self.figure_Phase = Figure()
        self.canvas_Phase = FigureCanvas(self.figure_Phase)
        self.index_Phase = self.stackedWidget_Phase.addWidget(self.canvas_Phase)
        self.stackedWidget_Phase.setCurrentIndex(self.index_Phase)
        self.toolbar_Phase = NavigationToolbar(self.canvas_Phase,self)
        self.horizontalLayout_Phase.addWidget(self.toolbar_Phase)
        self.axis_Phase = self.figure_Phase.add_subplot()

        self.figure_GroupDelay = Figure()
        self.canvas_GroupDelay = FigureCanvas(self.figure_GroupDelay)
        self.index_GroupDelay = self.stackedWidget_GroupDelay.addWidget(self.canvas_GroupDelay)
        self.stackedWidget_GroupDelay.setCurrentIndex(self.index_GroupDelay)
        self.toolbar_GroupDelay = NavigationToolbar(self.canvas_GroupDelay,self)
        self.horizontalLayout_GroupDelay.addWidget(self.toolbar_GroupDelay)
        self.axis_GroupDelay = self.figure_GroupDelay.add_subplot()

        self.figure_ZerosAndPoles = Figure()
        self.canvas_ZerosAndPoles = FigureCanvas(self.figure_ZerosAndPoles)
        self.index_ZerosAndPoles = self.stackedWidget_ZerosAndPoles.addWidget(self.canvas_ZerosAndPoles)
        self.stackedWidget_ZerosAndPoles.setCurrentIndex(self.index_ZerosAndPoles)
        self.toolbar_ZerosAndPoles = NavigationToolbar(self.canvas_ZerosAndPoles,self)
        self.horizontalLayout_ZerosAndPoles.addWidget(self.toolbar_ZerosAndPoles)
        self.axis_ZerosAndPoles = self.figure_ZerosAndPoles.add_subplot()

        self.figure_ImpulseResponse = Figure()
        self.canvas_ImpulseResponse = FigureCanvas(self.figure_ImpulseResponse)
        self.index_ImpulseResponse = self.stackedWidget_ImpulseResponse.addWidget(self.canvas_ImpulseResponse)
        self.stackedWidget_ImpulseResponse.setCurrentIndex(self.index_ImpulseResponse)
        self.toolbar_ImpulseResponse = NavigationToolbar(self.canvas_ImpulseResponse,self)
        self.horizontalLayout_ImpulseResponse.addWidget(self.toolbar_ImpulseResponse)
        self.axis_ImpulseResponse = self.figure_ImpulseResponse.add_subplot()

        self.figure_StepResponse = Figure()
        self.canvas_StepResponse = FigureCanvas(self.figure_StepResponse)
        self.index_StepResponse = self.stackedWidget_StepResponse.addWidget(self.canvas_StepResponse)
        self.stackedWidget_StepResponse.setCurrentIndex(self.index_StepResponse)
        self.toolbar_StepResponse = NavigationToolbar(self.canvas_StepResponse,self)
        self.horizontalLayout_StepResponse.addWidget(self.toolbar_StepResponse)
        self.axis_StepResponse = self.figure_StepResponse.add_subplot()

        self.figure_Q = Figure()
        self.canvas_Q = FigureCanvas(self.figure_Q)
        self.index_Q = self.stackedWidget_Q.addWidget(self.canvas_Q)
        self.stackedWidget_Q.setCurrentIndex(self.index_Q)
        self.toolbar_Q = NavigationToolbar(self.canvas_Q,self)
        self.horizontalLayout_Q.addWidget(self.toolbar_Q)
        self.axis_Q = self.figure_Q.add_subplot()

        self.figure_StagesGain = Figure()
        self.canvas_StagesGain = FigureCanvas(self.figure_StagesGain)
        self.index_StagesGain = self.stackedWidget_StagesGain.addWidget(self.canvas_StagesGain)
        self.stackedWidget_StagesGain.setCurrentIndex(self.index_StagesGain)
        self.toolbar_StagesGain = NavigationToolbar(self.canvas_StagesGain,self)
        self.horizontalLayout_StagesGain.addWidget(self.toolbar_StagesGain)
        self.axis_StagesGain = self.figure_StagesGain.add_subplot()

        self.figure_StagesPhase = Figure()
        self.canvas_StagesPhase = FigureCanvas(self.figure_StagesPhase)
        self.index_StagesPhase = self.stackedWidget_StagesPhase.addWidget(self.canvas_StagesPhase)
        self.stackedWidget_StagesPhase.setCurrentIndex(self.index_StagesPhase)
        self.toolbar_StagesPhase = NavigationToolbar(self.canvas_StagesPhase,self)
        self.horizontalLayout_StagesPhase.addWidget(self.toolbar_StagesPhase)
        self.axis_StagesPhase = self.figure_StagesPhase.add_subplot()

    #

    def __refreshFilterMakerGraphs(self):
        self.__cleanFilterMakerGraphs()
        self.auxGraphIndex = 1
        for i in self.myFilters:
            if i[3]:
                self.__addGraphicsForFilterMaker(i[0],i[1],i[2])
        self.auxGraphIndex = 0

    def __addGraphicsForFilterMaker(self,name,filter,transfunc):
        self.axis_Magnitude.semilogx(transfunc[0]/(2*np.pi),transfunc[1],label=name)
        self.axis_Magnitude.legend()
        self.canvas_Magnitude.draw()

        self.axis_Phase.semilogx(transfunc[0]/(2*np.pi),transfunc[2],label=name)
        self.axis_Phase.legend()
        self.canvas_Phase.draw()

        z,p,gk=filter.get_zpGk()
        zReal = []
        zImag = []
        for i in z:
            zReal.append(np.real(i))
            zImag.append(np.imag(i))
        pReal = []
        pImag = []
        for i in p:
            pReal.append(np.real(i))
            pImag.append(np.imag(i))

        temp = self.axis_ZerosAndPoles.scatter(pReal,pImag,marker="x",label=name+' - Poles')
        color = temp.get_facecolor()[0]
        if len(zReal) != 0:
            self.axis_ZerosAndPoles.scatter(zReal,zImag,marker="o",label=name+' - Zeros',color=color)
        self.axis_ZerosAndPoles.legend()
        self.canvas_ZerosAndPoles.draw()

        w,delay = filter.get_Group_Delay()
        self.axis_GroupDelay.semilogx(np.divide(w,2*np.pi),delay,label=name)
        self.axis_GroupDelay.legend()
        self.canvas_GroupDelay.draw()

        w_att,att = filter.get_Attenuation()
        self.axis_Attenuation.semilogx(np.divide(w_att,2*np.pi),att,label=name)
        self.axis_Attenuation.legend()
        self.canvas_Attenuation.draw()

        w_norm,norm = filter.get_Norm_Attenuation()
        self.axis_NormalizedAttenuation.semilogx(w_norm,norm,label=name)
        self.axis_NormalizedAttenuation.legend()
        self.canvas_NormalizedAttenuation.draw()

        #Hs = filter.get_ssTransferFunction()
        #impulseResponse = ss.impulse(Hs)
        #stepResponse = ss.step(Hs)

        impulse = filter.get_Impulse_Response()
        self.axis_ImpulseResponse.plot(impulse[0],impulse[1],label=name)
        self.axis_ImpulseResponse.legend()
        self.canvas_ImpulseResponse.draw()

        step = filter.get_Step_Response()
        self.axis_StepResponse.plot(step[0],step[1],label=name)
        self.axis_StepResponse.legend()
        self.canvas_StepResponse.draw()

        if DEBUG:
            print("_____________________________")
            print(filter.get_very_useful_data())
            print("------------------------------")
        make_rectangles = True
        fpm, fpM, Ap, fam, faM, Aa = filter.get_very_useful_data()
        if fam == None or fpm == None or Aa == None or Ap == None:
            if DEBUG:
                print("es retardo de grupo")
            make_rectangles = False
            ft,GD,tolerance,gain = filter.get_very_very_useful_data()
            overline = [[1e-1,1e9],
                        [GD*1e-6,GD*1e-6]]
            rectangle = [[ft,1e-1,1e-1,ft,ft],
                         [0,0,GD*(100-tolerance)/100*1e-6,GD*(100-tolerance)/100*1e-6,0]]
            self.__drawRectangleGD([overline,rectangle])
            self.axis_GroupDelay.legend()
            self.canvas_GroupDelay.draw()
        elif faM == None or fpM == None:
            #NO ES PASA BANDA NI RECHAZA BANDA
            #ES PASABAJOS O PASAALTOS
            if fam < fpm:
                #ES PASAALTOS
                rectangle_att=[[fam,fam,fam/1000,fam/1000,fam],
                               [Aa,Aa-100,Aa-100,Aa,Aa]]
                rectangle_pass=[[fpm,fpm,fpm*1000,fpm*1000,fpm],
                                [Ap,Ap+100,Ap+100,Ap,Ap]]
                #colorAtt = tempAtt.get_facecolor()[0]
                self.axis_Attenuation.semilogx(rectangle_att[0],rectangle_att[1],color='k')
                self.axis_Attenuation.semilogx(rectangle_pass[0],rectangle_pass[1],color='k')
                self.axis_Attenuation.legend()
                self.canvas_Attenuation.draw()

            else:
                #ES PASABAJOS
                rectangle_pass = [[fpm, fpm, fpm / 1000, fpm / 1000, fpm],
                                  [Ap, Ap + 100, Ap + 100, Ap, Ap]]
                rectangle_att = [[fam, fam, fam * 1000, fam * 1000, fam],
                                 [Aa, Aa - 100, Aa - 100, Aa, Aa]]
                #colorAtt = tempAtt.get_facecolor()[0]
                self.__drawRectangleAtt([rectangle_att,rectangle_pass])
                self.axis_Attenuation.legend()
                self.canvas_Attenuation.draw()
        else:
            #PASA BANDA O RECHAZA BANDA
            if fam<fpm:
                #PASA BANDA
                rectangle_att_izq = [[fam,fam,fam/1000,fam/1000,fam],
                                     [Aa,Aa-100,Aa-100,Aa,Aa]]
                rectangle_pass = [[fpm,fpm,fpM,fpM,fpm],
                                  [Ap,Ap+100,Ap+100,Ap,Ap]]
                rectangle_att_der = [[faM, faM, faM * 1000, faM * 1000, faM],
                                     [Aa, Aa - 100, Aa - 100, Aa, Aa]]
                self.__drawRectangleAtt([rectangle_att_izq,rectangle_pass,rectangle_att_der])
                self.axis_Attenuation.legend()
                self.canvas_Attenuation.draw()
            else:
                rectangle_pass_izq = [[fpm,fpm,fpm/1000,fpm/1000,fpm],
                                      [Ap, Ap + 100, Ap + 100, Ap, Ap]]
                rectangle_att = [[fam,fam,faM,faM,fam],
                                 [Aa, Aa - 100, Aa - 100, Aa, Aa]]
                rectangle_pass_der = [[fpM, fpM, fpM * 1000, fpM * 1000, fpM],
                                      [Ap, Ap + 100, Ap + 100, Ap, Ap]]
                self.__drawRectangleAtt([rectangle_pass_izq, rectangle_att, rectangle_pass_der])
                self.axis_Attenuation.legend()
                self.canvas_Attenuation.draw()

        if make_rectangles:
            wan = filter.get_wan()
            rectangle_pass = [[1, 1, 1 / 1000, 1 / 1000, 1],
                              [Ap, Ap + 100, Ap + 100, Ap, Ap]]
            rectangle_att = [[wan, wan, wan * 1000, wan * 1000, wan],
                             [Aa, Aa - 100, Aa - 100, Aa, Aa]]
            # colorAtt = tempAtt.get_facecolor()[0]
            self.__drawNormRectangleAtt([rectangle_att, rectangle_pass])
            self.axis_NormalizedAttenuation.legend()
            self.canvas_NormalizedAttenuation.draw()
        #self.axis_ImpulseResponse.plot(impulseResponse[0],impulseResponse[1],label=name)
        #self.axis_ImpulseResponse.legend()
        #self.canvas_ImpulseResponse.draw()

        #self.axis_StepResponse.plot(stepResponse[0],stepResponse[1],label=name)
        #self.axis_StepResponse.legend()
        #self.canvas_StepResponse.draw()

        q_y = filter.get_Qs()
        q_x = np.multiply([1]*len(q_y),self.auxGraphIndex)

        self.axis_Q.scatter(q_x,q_y,marker="o",label=name)
        self.axis_Q.legend()
        self.canvas_Q.draw()

        self.auxGraphIndex += 1

    def __drawRectangleAtt(self,rectangles):
        for rectangle in rectangles:
            self.axis_Attenuation.semilogx(rectangle[0], rectangle[1], color='k')

    def __drawNormRectangleAtt(self,rectangles):
        for rectangle in rectangles:
            self.axis_NormalizedAttenuation.semilogx(rectangle[0], rectangle[1], color='k')

    def __drawRectangleGD(self,rectangles):
        for rectangle in rectangles:
            self.axis_GroupDelay.semilogx(rectangle[0], rectangle[1], color='k')

    def __cleanFilterMakerGraphs(self):
        self.axis_Magnitude.clear()
        self.axis_Magnitude.grid()
        self.canvas_Magnitude.draw()
        self.axis_Magnitude.set_title("Magnitude Plot")
        self.axis_Magnitude.set_xlabel("Frequency [Hz]")
        self.axis_Magnitude.set_ylabel("Magnitude [dB]")

        self.axis_Attenuation.clear()
        self.axis_Attenuation.grid()
        self.canvas_Attenuation.draw()
        self.axis_Attenuation.set_title("Attenuation Plot")
        self.axis_Attenuation.set_xlabel("Frequency [Hz]")
        self.axis_Attenuation.set_ylabel("Attenuation [dB]")

        self.axis_NormalizedAttenuation.clear()
        self.axis_NormalizedAttenuation.grid()
        self.canvas_NormalizedAttenuation.draw()
        self.axis_NormalizedAttenuation.set_title("Normalized Attenuation Plot")
        self.axis_NormalizedAttenuation.set_xlabel("Normalized Frequency [dimensionless]")
        self.axis_NormalizedAttenuation.set_ylabel("Attenuation [dB]")

        self.axis_Phase.clear()
        self.axis_Phase.grid()
        self.canvas_Phase.draw()
        self.axis_Phase.set_title("Phase Plot")
        self.axis_Phase.set_xlabel("Frequency [Hz]")
        self.axis_Phase.set_ylabel("Phase [deg]")

        self.axis_GroupDelay.clear()
        self.axis_GroupDelay.grid()
        self.canvas_GroupDelay.draw()
        self.axis_GroupDelay.set_title("Group Delay Plot")
        self.axis_GroupDelay.set_xlabel("Frequency [Hz]")
        self.axis_GroupDelay.set_ylabel("Group Delay [seg]")

        self.axis_ZerosAndPoles.clear()
        self.axis_ZerosAndPoles.grid()
        self.canvas_ZerosAndPoles.draw()
        self.axis_ZerosAndPoles.set_title("Poles and Zeros Plot")
        self.axis_ZerosAndPoles.set_xlabel("Real part [1/seg]")
        self.axis_ZerosAndPoles.set_ylabel("Imaginary part [1/seg]")

        self.axis_ImpulseResponse.clear()
        self.axis_ImpulseResponse.grid()
        self.canvas_ImpulseResponse.draw()
        self.axis_ImpulseResponse.set_title("Impulse Response Plot")
        self.axis_ImpulseResponse.set_xlabel("Time [seg]")
        self.axis_ImpulseResponse.set_ylabel("Voltage [V]")

        self.axis_StepResponse.clear()
        self.axis_StepResponse.grid()
        self.canvas_StepResponse.draw()
        self.axis_StepResponse.set_title("Step Response Plot")
        self.axis_StepResponse.set_xlabel("Time [seg]")
        self.axis_StepResponse.set_ylabel("Voltage [V]")

        self.axis_Q.clear()
        self.axis_Q.grid()
        self.canvas_Q.draw()
        self.axis_Q.set_title("Quality Factor Plot")
        self.axis_Q.set_xlabel("Filter")
        self.axis_Q.set_ylabel("Q factor")
        ticks = 0
        for i in self.myFilters:
            if i[3]:
                ticks += 1
        self.axis_Q.set_xticks(ticks=range(1,ticks+1))

    ##################################################################################
    # CREACION DE STAGES
    ##################################################################################

    def __cancelNewStage(self):
        self.__showHideState_CreateNewStage()
        #self.tempPolos = []
        #self.tempZeros = []
        self.polosAreComplex = False
        self.polosIndex = []
        self.cerosRequired = False
        self.cerosAreComples = False
        self.cerosIndex = []

    def __crateNewStage(self):
        self.__showHideState_SelectOrder()

    def __clicked_1stOrderPole(self):
        self.__showHideState_SelectRealPole()
        self.polosAreComplex = False

    def __clicked_secondOrderPoles(self):
        self.__showHideState_SelectTypeOfPoles()

    def __clicked_complexPoles(self):
        self.__showHideState_SelectComplexPoles()
        self.polosAreComplex = True

    def __clicked_realPoles(self):
        self.__showHideState_SelectRealPoles()
        self.polosAreComplex = False

    ###

    def __selectRealPole(self):
        i = self.comboBox_SelectRealPole.currentIndex()
        if i != 0 and not ("USED" in self.comboBox_SelectRealPole.itemText(i)):
            self.polosIndex.append(i)
            self.__showHideState_1stOrderPoleReady()
        else:
            self.__error_message("INVALID POLE SELECTED")
            self.__cancelNewStage()

    def __selectRealPoles(self):
        i1 = self.comboBox_Select1stRealPole.currentIndex()
        i2 = self.comboBox_Select2ndRealPole.currentIndex()
        if i1 != 0 and i2 != 0 and i1 != i2 and not (
                "USED" in self.comboBox_Select1stRealPole.itemText(i1)) and not (
                "USED" in self.comboBox_Select2ndRealPole.itemText(i2)):
            self.polosIndex.append(i1)
            self.polosIndex.append(i2)
            self.__showHideState_2ndOrderPolesReady()
        else:
            self.__error_message("INVALID POLES SELECTED")
            self.__cancelNewStage()

    def __selectComplexPoles(self):
        i = self.comboBox_SelectComplexPoles.currentIndex()
        if i!= 0 and not ("USED" in self.comboBox_SelectComplexPoles.itemText(i)):
            self.polosIndex.append(i)
            self.__showHideState_2ndOrderPolesReady()
        else:
            self.__error_message("INVALID POLES SELECTED")
            self.__cancelNewStage()

    def __finish1stOrderPole(self):
        self.__createSOS()
        self.__cancelNewStage()

    def __clicked_addZero(self):
        self.__showHideState_SelectRealZero()
        self.cerosRequired = True
        self.cerosAreComples = False

    def __finish2ndOrderPole(self):
        self.__createSOS()
        self.__cancelNewStage()

    def __clicked_add1Zero(self):
        self.__showHideState_SelectRealZero()
        self.cerosRequired = True
        self.cerosAreComples = False

    def __clicked_add2Zeros(self):
        self.__showHideState_SelectTypeOfZeros()

    def __clicked_complexZeros(self):
        self.__showHideState_SelectComplexZeros()
        self.cerosRequired = True
        self.cerosAreComples = True

    def __clicked_realZeros(self):
        self.__showHideState_SelectRealZeros()
        self.cerosRequired = True
        self.cerosAreComples = False

    def __finishRealZero(self):
        i = self.comboBox_SelectRealZero.currentIndex()
        if i != 0 and not ("USED" in self.comboBox_SelectRealZero.itemText(i)):
            self.cerosIndex.append(i)
            self.__createSOS()
            self.__cancelNewStage()
        else:
            self.__error_message("INVALID ZERO SELECTED")
            self.__cancelNewStage()

    def __finishRealZeros(self):
        i1 = self.comboBox_Select1stRealZero.currentIndex()
        i2 = self.comboBox_Select2ndRealZero.currentIndex()
        if i1 != 0 and i2 != 0 and i1 != i2 and not (
                "USED" in self.comboBox_Select1stRealZero.itemText(i1)) and not (
                "USED" in self.comboBox_Select2ndRealZero.itemText(i2)):
            self.cerosIndex.append(i1)
            self.cerosIndex.append(i2)
            self.__createSOS()
            self.__cancelNewStage()
        else:
            self.__error_message("INVALID ZEROS SELECTED")
            self.__cancelNewStage()

    def __finishComplexZeros(self):
        i = self.comboBox_SelectComplexZeros.currentIndex()
        if i != 0 and not ("USED" in self.comboBox_SelectComplexZeros.itemText(i)):
            self.cerosIndex.append(i)
            self.__createSOS()
            self.__cancelNewStage()
        else:
            self.__error_message("INVALID ZEROS SELECTED")
            self.__cancelNewStage()
    #

    def __createSOS(self): #OJO!!! INDEX SUMADOS MAS 1
        if DEBUG:
            print(self.polosIndex)
            print(self.polosAreComplex)
            print(self.cerosIndex)
            print(self.cerosAreComples)
            print(self.cerosRequired)

        if self.polosAreComplex:
            polosArray = self.ComplexPoles
            polosUsedArray = self.ComplexPolesUsed
            polosWidgetArray = self.ComplexPolesWidgets
            polesWidgetConstant = 5
            polesComboBoxes = [self.comboBox_SelectComplexPoles]
            polosName = "C"
        else:
            polosArray = self.RealPoles
            polosUsedArray = self.RealPolesUsed
            polosWidgetArray = self.RealPolesWidgets
            polesWidgetConstant = 3
            polesComboBoxes = [self.comboBox_SelectRealPole,
                               self.comboBox_Select1stRealPole,
                               self.comboBox_Select2ndRealPole]
            polosName = "R"
        if self.cerosAreComples:
            cerosArray = self.ComplexZeros
            cerosUsedArray = self.ComplexZerosUsed
            cerosWidgetArray = self.ComplexZerosWidgets
            cerosWidgetConstant = 3
            cerosComboBoxes = [self.comboBox_SelectComplexZeros]
            cerosName = "C"
        else:
            cerosArray = self.RealZeros
            cerosUsedArray = self.RealZerosUsed
            cerosWidgetArray = self.RealZerosWidgets
            cerosWidgetConstant = 2
            cerosComboBoxes = [self.comboBox_SelectRealZero,
                               self.comboBox_Select1stRealZero,
                               self.comboBox_Select2ndRealZero]
            cerosName = "R"

        sosPolos = []
        for i in self.polosIndex:
            polosUsedArray[i-1] = True
            polosWidgetArray[polesWidgetConstant*i-1].setChecked(True)
            for combo in polesComboBoxes:
                text = combo.itemText(i)
                combo.setItemText(i,'USED - '+text)
            sosPolos.append(polosArray[i-1])
            if self.polosAreComplex:
                sosPolos.append(np.conjugate(polosArray[i-1]))
        sosCeros = []
        if self.cerosRequired:
            for i in self.cerosIndex:
                cerosUsedArray[i-1] = True
                cerosWidgetArray[cerosWidgetConstant*i-1].setChecked(True)
                for combo in cerosComboBoxes:
                    text = combo.itemText(i)
                    combo.setItemText(i,'USED - '+text)
                sosCeros.append(cerosArray[i-1])
                if self.cerosAreComples:
                    sosCeros.append(np.conjugate(cerosArray[i-1]))

        if self.cerosRequired:
            name = polosName + 'P:' + str(self.polosIndex) + ' ' + cerosName + 'Z:' + str(self.cerosIndex)
        else:
            name = polosName + 'P:' + str(self.polosIndex)

        newSos = SimpleHs(sosCeros,sosPolos)
        self.sos.append([newSos,True,
                         [self.polosIndex,
                          self.polosAreComplex,
                          self.cerosIndex,
                          self.cerosAreComples,
                          self.cerosRequired],name])
        self.__refreshStagesGraphs()
        self.comboBox_YourStages.addItem(name)

    #####################################################################33

    def __showHideState_CreateNewStage(self):
        self.__showAndHideButtons([1, 0,0,0, 0,0,0, 0,0, 0,0,0, 0,0, 0,0,0, 0,0,0,0, 0,0,0, 0,0, 0,0,0, 0,0])

    def __showHideState_SelectOrder(self):
        self.__showAndHideButtons([0, 1,1,1, 0,0,0, 0,0, 0,0,0, 0,0, 0,0,0, 0,0,0,0, 0,0,0, 0,0, 0,0,0, 0,0])

    def __showHideState_SelectTypeOfPoles(self):
        self.__showAndHideButtons([0, 0,0,0, 1,1,1, 0,0, 0,0,0, 0,0, 0,0,0, 0,0,0,0, 0,0,0, 0,0, 0,0,0, 0,0])

    def __showHideState_SelectRealPole(self):
        self.__showAndHideButtons([0, 0,0,0, 0,0,0, 1,1, 0,0,0, 0,0, 0,0,0, 0,0,0,0, 0,0,0, 0,0, 0,0,0, 0,0])

    def __showHideState_SelectRealPoles(self):
        self.__showAndHideButtons([0, 0,0,0, 0,0,0, 0,0, 1,1,1, 0,0, 0,0,0, 0,0,0,0, 0,0,0, 0,0, 0,0,0, 0,0])

    def __showHideState_SelectComplexPoles(self):
        self.__showAndHideButtons([0, 0,0,0, 0,0,0, 0,0, 0,0,0, 1,1, 0,0,0, 0,0,0,0, 0,0,0, 0,0, 0,0,0, 0,0])

    def __showHideState_1stOrderPoleReady(self):
        self.__showAndHideButtons([0, 0,0,0, 0,0,0, 0,0, 0,0,0, 0,0, 1,1,1, 0,0,0,0, 0,0,0, 0,0, 0,0,0, 0,0])

    def __showHideState_2ndOrderPolesReady(self):
        self.__showAndHideButtons([0, 0,0,0, 0,0,0, 0,0, 0,0,0, 0,0, 0,0,0, 1,1,1,1, 0,0,0, 0,0, 0,0,0, 0,0])

    def __showHideState_SelectTypeOfZeros(self):
        self.__showAndHideButtons([0, 0,0,0, 0,0,0, 0,0, 0,0,0, 0,0, 0,0,0, 0,0,0,0, 1,1,1, 0,0, 0,0,0, 0,0])

    def __showHideState_SelectRealZero(self):
        self.__showAndHideButtons([0, 0,0,0, 0,0,0, 0,0, 0,0,0, 0,0, 0,0,0, 0,0,0,0, 0,0,0, 1,1, 0,0,0, 0,0])

    def __showHideState_SelectRealZeros(self):
        self.__showAndHideButtons([0, 0,0,0, 0,0,0, 0,0, 0,0,0, 0,0, 0,0,0, 0,0,0,0, 0,0,0, 0,0, 1,1,1, 0,0])

    def __showHideState_SelectComplexZeros(self):
        self.__showAndHideButtons([0, 0,0,0, 0,0,0, 0,0, 0,0,0, 0,0, 0,0,0, 0,0,0,0, 0,0,0, 0,0, 0,0,0, 1,1])

    def __showAndHideButtons(self, arr):
        length = len(self.itemList)
        for i in range(length):
            self.__showOrHideSingle(self.itemList[i],arr[i])

    def __showAndHideParameters(self):
        filterT = self.comboBox_filterType.currentIndex()
        if filterT is filterType.LP.value or filterT is filterType.HP.value:
            self.label_5.show()
            self.label_6.show()
            self.label_7.show()
            self.label_8.show()
            self.doubleSpinBox_Aa.show()
            self.doubleSpinBox_Ap.show()
            self.doubleSpinBox_fa_minus.show()
            self.doubleSpinBox_fp_minus.show()

            self.label_9.hide()
            self.label_10.hide()
            self.doubleSpinBox_fa_plus.hide()
            self.doubleSpinBox_fp_plus.hide()

            self.label_47.hide()
            self.label_49.hide()
            self.label_51.hide()
            self.doubleSpinBox_ft.hide()
            self.doubleSpinBox_Tolerance.hide()
            self.doubleSpinBox_GroupDelay.hide()
        elif filterT is filterType.BP.value or filterT is filterType.BS.value:
            self.label_5.show()
            self.label_6.show()
            self.label_7.show()
            self.label_8.show()
            self.doubleSpinBox_Aa.show()
            self.doubleSpinBox_Ap.show()
            self.doubleSpinBox_fa_minus.show()
            self.doubleSpinBox_fp_minus.show()

            self.label_9.show()
            self.label_10.show()
            self.doubleSpinBox_fa_plus.show()
            self.doubleSpinBox_fp_plus.show()

            self.label_47.hide()
            self.label_49.hide()
            self.label_51.hide()
            self.label_47.hide()
            self.label_49.hide()
            self.label_51.hide()
            self.doubleSpinBox_ft.hide()
            self.doubleSpinBox_Tolerance.hide()
            self.doubleSpinBox_GroupDelay.hide()
        else:
            self.label_5.hide()
            self.label_6.hide()
            self.label_7.hide()
            self.label_8.hide()
            self.doubleSpinBox_Aa.hide()
            self.doubleSpinBox_Ap.hide()
            self.doubleSpinBox_fa_minus.hide()
            self.doubleSpinBox_fp_minus.hide()

            self.label_9.hide()
            self.label_10.hide()
            self.doubleSpinBox_fa_plus.hide()
            self.doubleSpinBox_fp_plus.hide()

            self.label_47.show()
            self.label_49.show()
            self.label_51.show()
            self.doubleSpinBox_ft.show()
            self.doubleSpinBox_Tolerance.show()
            self.doubleSpinBox_GroupDelay.show()

    def __showOrHideSingle(self, item, show):
        if show:
            item.show()
        else:
            item.hide()

    def __setConstants(self):
        self.itemList=[
            self.pushButton_CreateNewStage,

            self.label_33,
            self.pushButton_1stOrderPole,
            self.pushButton_SecondOrderPoles,

            self.label_34,
            self.pushButton_ComplexPoles,
            self.pushButton_RealPoles,

            self.comboBox_SelectRealPole,
            self.pushButton_SelectRealPole,

            self.comboBox_Select1stRealPole,
            self.comboBox_Select2ndRealPole,
            self.pushButton_SelectRealPoles,

            self.comboBox_SelectComplexPoles,
            self.pushButton_SelectComplexPoles,

            self.label_37,
            self.pushButton_FINISH1stOrder,
            self.pushButton_AddZero_1stOrder,

            self.label_38,
            self.pushButton_FINISH2ndOrder,
            self.pushButton_Add1Zero,
            self.pushButton_Add2Zeros,

            self.label_39,
            self.pushButton_ComplexConjZeros,
            self.pushButton_RealZeros,

            self.comboBox_SelectRealZero,
            self.pushButton_SelectRealZero,

            self.comboBox_Select1stRealZero,
            self.comboBox_Select2ndRealZero,
            self.pushButton_SelectRealZeros,

            self.comboBox_SelectComplexZeros,
            self.pushButton_SelectComplexZeros
        ]
        self.errorBox = QtWidgets.QMessageBox()
        self.myFilters = []
        self.myStages = []
        self.currentFilter = None
        self.currentStage = None

        # PREVIOS A COMENZAR ETAPAS
        self.ComplexPoles = []
        self.RealPoles = []
        self.ComplexZeros = []
        self.RealZeros = []
        self.ComplexPolesUsed = []
        self.RealPolesUsed = []
        self.ComplexZerosUsed = []
        self.RealZerosUsed = []
        self.ComplexPolesWidgets = []
        self.RealPolesWidgets = []
        self.ComplexZerosWidgets = []
        self.RealZerosWidgets = []

        #self.tempPolos = []
        #self.tempZeros = []
        # USADOS PARA LA CREACION DE NUEVAS ETAPAS
        self.polosAreComplex = False
        self.polosIndex = []
        self.cerosRequired = False
        self.cerosAreComples = False
        self.cerosIndex = []

        #ULTIMA ETAPA: SOS y FOS
        self.sos = []
        ########################################################
        self.testvar1 = 0
        self.testvar2 = 0
        ########################################################
        self.auxGraphIndex = 0

    def __error_message(self, description):
        self.errorBox.setWindowTitle("Error")
        self.errorBox.setIcon(self.errorBox.Information)
        self.errorBox.setText(description)
        self.errorBox.exec()

    def __test(self):
        print("Test")
        #self.comboBox_YourFilters.addItem("LOL"+str(self.testvar1))
        #self.comboBox_SelectYourFilter.addItem("LOL"+str(self.testvar1))
        #self.testvar1+=1
        #self.myFilters.append([str(self.testvar1),None,None,True])
        #mfl = myFilterTest()

        #for y in reversed(range(self.gridLayout_TEST.rowCount()-1)):
        #    for x in reversed(range(self.gridLayout_TEST.columnCount()-1)):
        #        temp = self.gridLayout_TEST.itemAt(x + 3*y)
        #        self.gridLayout_TEST.removeItem(temp)
        #        del temp
        #l = [1,2]
        #a,b = l
        #print(str(a))
        #print(str(b))
        #w = [1]*26
        #print(w)

        #a = test
        #b = test
        #a = np.add(a,b)
        #print(a)
        #print(b)
        #print(test)

        #plt.text(r'\mu')

    def __test2(self):
        #print("Test2")
        #if self.comboBox_YourFilters.count() >= 2 + 1:
        #    self.comboBox_YourFilters.removeItem(2)
        #    self.comboBox_SelectYourFilter.removeItem(2)
        #    self.myFilters.pop(2)
        #else:
        #    print("YOLO")

        #x = [0,1,0,2,3,4,5]
        #print(str(0 in x))

        for x in range(3):
            for y in range(3):
                button = QPushButton(str(str(3*x+y)))
                button.setMaximumWidth(30)
                self.gridLayout_TEST.addWidget(button, x, y)

        print("TEST2")
        print(str(sp.oo))
        print(self.gridLayout_ComplexPoles.rowCount())

    def __manageDebug(self):
        if DEBUG:
            print("DEBUG")
        else:
            self.pushButton_TEST.hide()
            self.pushButton_TEST_2.hide()
            self.checkBox_Test.hide()
            self.line_16.hide()

class myFilterTest(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.z = [-15000j,15000j,-5000j,5000j,0,-3000,6000]
        #self.p = [-200000,-150000+150000j,-150000-150000j,
        #    -50000+200000j,-50000-200000j,
        #     -20000, -15000 + 15000j, -15000 - 15000j,
        #     -5000 + 20000j, -5000 - 20000j]
        r1 = random()
        r2 = random()
        r3 = random()
        r4 = random()
        r5 = random()
        r6 = random()
        r7 = random()
        self.z = [0, -4000*r2, 5000j*r1, -5000j*r1]
        self.p = [-4000+7000j*r3,-4000-7000j*r3,-6000*r4, -2000*r5]
        self.num = np.polynomial.polynomial.polyfromroots(self.z).tolist()
        self.den = np.polynomial.polynomial.polyfromroots(self.p).tolist()
        self.num.reverse()
        self.den.reverse()
        print(self.num)
        print(self.den)
        self.k = 1*r6
        self.gain = 150*r7

        self.Hs=ss.TransferFunction(np.real(self.num)*self.k*10**(self.gain/20),np.real(self.den))
        print(self.Hs)
        print(self.Hs.zeros)
        print(self.Hs.poles)

    def calculate(self):
        self.sys = ss.ZerosPolesGain(self.z,self.p,self.k)
        self.w,self.mag,self.pha = ss.bode(self.sys,np.logspace(0,9,10000)*(2*np.pi))
        self.mag += self.gain
        return self.w,self.mag,self.pha

    def get_zpGk(self):
        return self.z,self.p, self.k * np.power(10,self.gain/20)

    def get_ssTransferFunction(self):
        return self.Hs

    def get_Gain(self):
        return self.gain

    def get_Group_Delay(self):
        return [1,1000],[0,10]
