# -*- coding: utf-8 -*-
"""输入对话框示例"""
import sys
from PyQt5 import QtWidgets, QtCore, QtGui
import run

class Dialog(QtWidgets.QWidget):
    # 包含一个按钮和两个行编辑部件。
    # 单击按钮会弹出输入对话框，以获取用户输入的文本数据，该文本数据将会显示在第一个行编辑部件中。
    def __init__(self):
        super(Dialog, self).__init__()
        self.setWindowTitle("自动聊天机器人")
        self.setGeometry(200, 200, 1000, 500)
        self.setWindowIcon(QtGui.QIcon('figs/bot.png'))     # 设置图标
        self.ques, self.ans = None, None
        '--- in'
        self.inButton = QtWidgets.QPushButton('你要说的话', self)
        self.inButton.setFocusPolicy(QtCore.Qt.NoFocus)
        self.inButton.move(20, 20)
        self.inButton.clicked.connect(self.show_in_dialog)
        self.setFocus()

        self.inText = QtWidgets.QLineEdit(self)
        self.inText.move(300, 20)

        '--- out'
        self.outButton = QtWidgets.QPushButton('点击查看chatbot要说的话', self)
        self.outButton.setFocusPolicy(QtCore.Qt.NoFocus)
        self.outButton.move(20, 200)
        self.outButton.clicked.connect(self.show_out_dialog)

        self.outText = QtWidgets.QLineEdit(self)
        self.outText.move(300, 200)

    def show_in_dialog(self):
        text, ok = QtWidgets.QInputDialog.getText(self, "输入对话框", "你要说的话")
        # 用来显示一个输入对话框。
        # 第一个参数"输入对话框"是对话框的标题，第二个参数"请输入你的名字："将作为提示信息显示在对话框内。
        # 该对话框将返回用户输入的内容和一个布尔值，如果用户单击OK按钮确认输入，则返回的布尔值为true，否则返回的布尔值为false。
        if ok:
            self.inText.setText(text)
        ans = run.predict(text)
        self.ques, self.ans = text, ans

    def show_out_dialog(self):
        self.outText.setText(self.ans)



app = QtWidgets.QApplication(sys.argv)
dialog = Dialog()
dialog.show()
sys.exit(app.exec_())