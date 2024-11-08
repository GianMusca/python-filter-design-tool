from src.filterTool import FilterTool
from PyQt5.QtWidgets import QApplication
#from src.de_internet.latexManagement import testingLatexInMatPlotLib

#TESTINGLATEX = True

if __name__ == '__main__':
#    if TESTINGLATEX:
#        testingLatexInMatPlotLib()
#    else:
    app = QApplication([])
    window = FilterTool()
    window.show()
    app.exec()

