from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtUiTools import QUiLoader

class ColorPickerApp(QMainWindow):
    def __init__(self):
        super().__init__()

        loader = QUiLoader()
        self.ui = loader.load('colorPicker.ui', None)
        self.ui.show()
        
        self.red = 0
        self.green = 0
        self.blue = 0
        
        self.ui.HSlider_red.valueChanged.connect(self.update_red_label)
        self.ui.HSlider_green.valueChanged.connect(self.update_green_label)
        self.ui.HSlider_blue.valueChanged.connect(self.update_blue_label)

    def update_red_label(self, value):
        self.ui.red_label2.setText(str(value))
        self.red = int(self.ui.red_label2.text())
        self.apply_result(self.red, self.green, self.blue)

    def update_green_label(self, value):    
        self.ui.green_label2.setText(str(value))
        self.green = int(self.ui.green_label2.text())
        self.apply_result(self.red, self.green, self.blue)

    def update_blue_label(self, value):    
        self.ui.blue_label2.setText(str(value))
        self.blue = int(self.ui.blue_label2.text())
        self.apply_result(self.red, self.green, self.blue)

    def apply_result(self, r, g, b):
        self.ui.result_label.setStyleSheet(f"background-color: rgb({r},{g},{b}); color: rgb(255, 255, 255); border-radius: 15px;")
        self.ui.result_label.setText(f'rgb({r}, {g}, {b})')
    
app = QApplication()
window = ColorPickerApp()
app.exec_()