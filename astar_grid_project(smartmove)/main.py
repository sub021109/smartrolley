import sys
from PyQt5.QtWidgets import QApplication
from grid_map import GridMap

app = QApplication(sys.argv)
window = GridMap(rows=20, cols=20)
window.show()
sys.exit(app.exec_())
