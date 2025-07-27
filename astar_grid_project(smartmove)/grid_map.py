from PyQt5.QtWidgets import QWidget, QLabel
from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtCore import QPoint
from astar import astar

class GridMap(QWidget):
    def __init__(self, rows=20, cols=20, cell_size=25):
        super().__init__()
        self.rows, self.cols = rows, cols
        self.cell_size = cell_size
        self.start = None
        self.goal = None
        self.grid = [[0 for _ in range(cols)] for _ in range(rows)]
        self.path = []

        self.setFixedSize(cols*cell_size, rows*cell_size)
        self.setWindowTitle("A* Grid Map")

    def mousePressEvent(self, event):
        x = event.x() // self.cell_size
        y = event.y() // self.cell_size
        if self.start is None:
            self.start = (y, x)
        elif self.goal is None:
            self.goal = (y, x)
            self.path = astar(self.grid, self.start, self.goal)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        for y in range(self.rows):
            for x in range(self.cols):
                rect_color = QColor(255,255,255)
                if self.grid[y][x] == 1:
                    rect_color = QColor(0,0,0)
                elif self.grid[y][x] == -1:
                    rect_color = QColor(128,128,128)
                painter.fillRect(x*self.cell_size, y*self.cell_size, self.cell_size, self.cell_size, rect_color)
                painter.drawRect(x*self.cell_size, y*self.cell_size, self.cell_size, self.cell_size)

        if self.start:
            sx, sy = self.start[1]*self.cell_size, self.start[0]*self.cell_size
            painter.fillRect(sx, sy, self.cell_size, self.cell_size, QColor(0,255,0))
        if self.goal:
            gx, gy = self.goal[1]*self.cell_size, self.goal[0]*self.cell_size
            painter.fillRect(gx, gy, self.cell_size, self.cell_size, QColor(255,0,0))

        for y, x in self.path:
            painter.fillRect(x*self.cell_size, y*self.cell_size, self.cell_size, self.cell_size, QColor(100, 100, 255, 150))
