import sys, os

from PyQt5.QtWidgets import *
from PyQt5.QtCore import QDir, Qt
from Generator import Generator
from GuiLogger import  GuiLogger


class DSGView(QWidget):

    def __init__(self):
        self.selectedPathes = []
        self.logger = GuiLogger(self)
        super().__init__()
        self.generator = None
        self.directory = ""

        self.grid = QGridLayout()
        self.grid.setSpacing(10)

        self.path_label = QLabel('Arbeitsverzeichnis')
        self.set_path_btn = QPushButton("Pfad wählen")
        self.set_path_btn.clicked.connect(self.on_set_path)
        self.path = QLineEdit()

        self.grid.addWidget(self.path_label, 1, 0, 1, 2)
        self.grid.addWidget(self.path, 1, 2, 1, 5)
        self.grid.addWidget(self.set_path_btn, 1, 7)

        self.file_model = QFileSystemModel()
        self.tree_view = QTreeView()
        self.tree_view.setModel(self.file_model)
        self.tree_view.setStyleSheet('color: black')
        self.tree_view.setSelectionMode(QAbstractItemView.MultiSelection)
        self.tree_view.selectionModel().selectionChanged.connect(self.getItems)
        self.tree_view.doubleClicked.connect(self.on_double_click)
        self.tree_view.setVisible(False)
        self.grid.addWidget(self.tree_view, 2, 2, 6, 5)

        self.error = QTextEdit()
        self.error.setLineWrapColumnOrWidth(200)
        self.error.setMaximumHeight(50)
        self.error.setStyleSheet('font-size: 9px;')
        self.error.setLineWrapMode(QTextEdit.FixedColumnWidth)
        self.grid.addWidget(self.error, 8, 2, 1, 5)

        self.dataset_btn = QPushButton('Datenset erzeugen')
        self.dataset_btn.setEnabled(False)
        self.dataset_btn.clicked.connect(self.on_dataset)
        self.grid.addWidget(self.dataset_btn, 8, 7, 1, 1, Qt.AlignBottom)

        self.sep_box = QCheckBox("Seperate Klassen?",self)
        self.sep_box.setEnabled(False)
        self.grid.addWidget(self.sep_box,8, 7, 1, 1, Qt.AlignTop)

        self.delete_images_btn = QPushButton("Leere Bilder löschen")
        self.delete_images_btn.setEnabled(False)
        self.delete_images_btn.clicked.connect(self.on_delete_img)
        self.grid.addWidget(self.delete_images_btn, 3, 0, 1, 2)

        self.create_stat_btn = QPushButton('Statistik erzeugen')
        self.create_stat_btn.setEnabled(False)
        self.create_stat_btn.clicked.connect(self.on_create_stat)
        self.grid.addWidget(self.create_stat_btn, 4, 0, 1, 2)

        self.train_btn = QPushButton('Train.csv erstellen')
        self.train_btn.setEnabled(False)
        self.train_btn.clicked.connect(self.on_create_train)
        self.grid.addWidget(self.train_btn, 5, 0, 1, 2)


        self.train_lbl = QLabel("Train Split")
        self.grid.addWidget(self.train_lbl,6,0,1,1,Qt.AlignBottom)

        self.train_split = QLineEdit()
        self.train_split.setFixedWidth(40)
        self.train_split.setEnabled(False)
        self.grid.addWidget(self.train_split,7,0,1,1,Qt.AlignTop)

        self.test_lbl = QLabel("Test Split")
        self.grid.addWidget(self.test_lbl,6,1,1,1,Qt.AlignBottom)

        self.test_split = QLineEdit()
        self.test_split.setFixedWidth(40)
        self.test_split.setEnabled(False)
        self.grid.addWidget(self.test_split,7,1,1,1,Qt.AlignTop)

        self.setLayout(self.grid)
        self.setGeometry(150, 150, 900, 450)
        self.setWindowTitle('Data Set Generator')

        self.up_btn = QToolButton()
        self.up_btn.setArrowType(Qt.UpArrow)
        self.up_btn.setEnabled(False)
        self.up_btn.clicked.connect(self.on_navi_up)
        self.grid.addWidget(self.up_btn, 2, 1)

    def on_set_path(self):
        self.directory = str(QFileDialog.getExistingDirectory(self, "Arbeitsverzeichnis wählen"))

        self.up_btn.setEnabled(True)
        self.dataset_btn.setEnabled(True)
        self.delete_images_btn.setEnabled(True)
        self.tree_view.setVisible(True)
        self.create_stat_btn.setEnabled(True)
        self.train_btn.setEnabled(True)
        self.sep_box.setEnabled(True)
        self.test_split.setEnabled(True)
        self.train_split.setEnabled(True)
        self._update_path(self.directory)

    def on_dataset(self):
        if self.determineSplit():
            classes= []
            if self.sep_box.checkState():
                sep = True
            else:
                sep = False

            if self.selectedPathes:
                for p in self.selectedPathes:
                    if os.path.isdir(p):

                        text, ok = QInputDialog.getText(self, 'Klassenauswahl', 'Klassen mir Kommata getrennt angeben. Range = 1-10. Leer = Alle Klassen')
                        if ok:
                            if not text:
                                classes = [x for x in range(155)]
                            else:
                                inputs = text.split(",")
                                for i in inputs:
                                    if "-" in i:
                                        r = i.split("-")
                                        for x in range(int(r[0]), int(r[1]) + 1):
                                            classes.append(x)
                                    else:
                                        classes.append(int(i))

                        if not sep:
                            self.generator = Generator(p, self.logger, classes=classes)
                            self.generator.createDataSetZIP(split=self.split)
                        else:
                            for c in classes:
                                name = "Klasse_"+str(c)+".zip"
                                self.generator = Generator(p, self.logger, classes=c)
                                self.generator.createDataSetZIP(name=name,split=self.split)
            else:
                text, ok = QInputDialog.getText(self, 'Klassenauswahl', 'Klassen mir Kommata getrennt angeben. Range = 1-10. Leer = Alle Klassen')
                if ok:
                    if not text:
                        classes = [x for x in range(155)]
                    else:
                        inputs = text.split(",")
                        for i in inputs:
                            if "-" in i:
                                r = i.split("-")
                                for x in range(int(r[0]), int(r[1]) + 1):
                                    classes.append(x)
                            else:
                                classes.append(int(i))
                if not sep:
                    self.generator = Generator(self.directory, self.logger, classes=classes)
                    self.generator.createDataSetZIP(split=self.split)
                else:
                    for c in classes:
                        name = "Klasse_" + str(c) + ".zip"
                        self.generator = Generator(self.directory, self.logger, classes=[c])
                        print("test")
                        self.generator.createDataSetZIP(name=name,split=self.split)
        else:
            self.logger.log_err("Kein korrekter Train/Test Split")

    def on_delete_img(self):
        deleted = []

        if self.selectedPathes:
            for p in self.selectedPathes:
                if os.path.isdir(p):
                    self.generator = Generator(p,self.logger)
                    deleted += self.generator.deleteEmptyImages()
        else:
            self.generator = Generator(self.directory, self.logger)
            deleted+=self.generator.deleteEmptyImages()

        if deleted:
            msg = 'Gelöschte Bilder:\n' + '\n'.join(deleted)
            self.error.setText(msg)
            self.error.setStyleSheet('color: green')
        else:
            self.error.setText('Keine leeren Bilder gefunden!')
            self.error.setStyleSheet('color: green')

    def on_create_stat(self):
        choices = ['Graphische Statistik', 'CSV Statistik', 'Beides']
        item, okPressed = QInputDialog.getItem(self, 'Statistik auswählen!', 'Statistik:', choices, 0, False)

        if okPressed and item:
            if self.selectedPathes:
                for p in self.selectedPathes:
                    if os.path.isdir(p):
                        self.generator = Generator(p,self.logger)
                        if item != choices[1]:
                            self.generator.createPieChart()
                        if item != choices[0]:
                            self.generator.createCSVOverview()

            else:
                self.generator = Generator(self.directory,self.logger)
                if item != choices[1]:
                    self.generator.createPieChart()
                if item != choices[0]:
                    self.generator.createCSVOverview()

    def on_create_train(self):
        msg = ""
        if self.selectedPathes:
            for p in self.selectedPathes:
                if os.path.isdir(p):
                    self.generator = Generator(p,self.logger)
                    self.generator.createCSVLabelMap()
        else:
            self.generator = Generator(self.directory,self.logger)
            self.generator.createCSVLabelMap()


    def getItems(self):
        selected = self.tree_view.selectionModel().selectedIndexes()
        new = []
        for index in selected:
            path = self.sender().model().filePath(index)
            if path not in new:
                new.append(path)
        self.selectedPathes = new

    def on_double_click(self):
        for index in self.tree_view.selectionModel().selectedIndexes():
            self.directory = self.sender().model().filePath(index)
        self._update_path(self.directory)
        self.selectedPathes = []
        self.tree_view.clearSelection()
        self.logger.clear()

    def on_navi_up(self):
        self.directory = '/'.join(self.directory.split('/')[:-1])
        self._update_path(self.directory)

    def _update_path(self,path):
        self.path.setText(path)
        self.directory = path
        self.file_model.setRootPath(self.directory)
        self.tree_view.setRootIndex(self.file_model.index(self.directory))

    def determineSplit(self):
        train = self.train_split.text()
        test = self.test_split.text()
        train = int(train) if train else 0
        test = int(test) if test else 0

        if train+test == 0:
            self.split = None
            return True
        elif train+test == 100:
            self.split = train
            return True
        else:
            return False

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DSGView()
    ex.show()
    sys.exit(app.exec_())
