import sys
import pandas as pd
import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from truot_ham_muc_tieu import phuong_phap_truot_ham_muc_tieu # Import hàm vẽ

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QStackedWidget, QFrame,
                             QTabWidget, QTableWidget, QTableWidgetItem, QComboBox, 
                             QSpinBox, QHeaderView, QAbstractItemView, QMessageBox, 
                             QTextEdit, QFileDialog, QSplitter)
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QPixmap, QColor, QBrush

from nhap_du_lieu import BaiToanQuyHoachTuyenTinh

# ==========================================================
# CUSTOM CLASS: Bắt tín hiệu print() đẩy lên Giao diện
# ==========================================================
class PrintRedirector(QObject):
    text_written = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.terminal = sys.stdout # Giữ lại terminal gốc để chống đứng IDE

    def write(self, text):
        self.terminal.write(text) # Vẫn in ra terminal đen
        self.text_written.emit(text) # Đẩy thêm lên giao diện
        
    def flush(self):
        self.terminal.flush()

# ==========================================================
# CUSTOM WIDGET: Bảng nhập liệu hỗ trợ nhấn Enter để nhảy ô
# ==========================================================
class BangNhapLieu(QTableWidget):
    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            row = self.currentRow()
            col = self.currentColumn()
            if col < self.columnCount() - 1:
                self.setCurrentCell(row, col + 1)
            elif row < self.rowCount() - 1:
                self.setCurrentCell(row + 1, 0)
        else:
            super().keyPressEvent(event)

# ==========================================================
# CLASS CHÍNH CỦA ỨNG DỤNG
# ==========================================================
class OptimizationSolverApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hệ Thống Giải Quy Hoạch Tuyến Tính")
        self.resize(1250, 750)
        
        # --- CÀI ĐẶT CONSOLE ẢO (Tránh lỗi chặn luồng) ---
        self.redirector = PrintRedirector()
        self.redirector.text_written.connect(self.in_ra_man_hinh_ket_qua)
        sys.stdout = self.redirector

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        self.man_hinh_chao_mung = self.tao_man_hinh_welcome()
        self.man_hinh_lam_viac = self.tao_man_hinh_workspace()
        
        self.stacked_widget.addWidget(self.man_hinh_chao_mung)
        self.stacked_widget.addWidget(self.man_hinh_lam_viac)
        self.stacked_widget.setCurrentIndex(0)

    def in_ra_man_hinh_ket_qua(self, text):
        self.txt_ket_qua.insertPlainText(text)
        scrollbar = self.txt_ket_qua.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    # --- MÀN HÌNH 1: WELCOME SCREEN ---
    # --- MÀN HÌNH 1: WELCOME SCREEN ---
    def tao_man_hinh_welcome(self):
        # Widget tổng chứa toàn bộ màn hình
        widget = QWidget()
        widget.setStyleSheet("background-color: #edf2f7;") # Nền tổng thể xám nhạt
        
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # --- TẠO KHUNG NỀN LÀM ĐIỂM NHẤN (CARD FRAME) ---
        frame_container = QFrame()
        frame_container.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border-radius: 15px;
                border: 1px solid #cbd5e0;
            }
            /* Tránh các Label bên trong bị dính viền của Frame */
            QLabel {
                border: none; 
                background: transparent;
            }
        """)
        
        # Thêm bóng đổ (Shadow) nhẹ cho khung thêm phần sang trọng
        from PyQt6.QtWidgets import QGraphicsDropShadowEffect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 40))
        shadow.setOffset(0, 5)
        frame_container.setGraphicsEffect(shadow)

        # Layout bên trong khung
        frame_layout = QVBoxLayout()
        frame_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        frame_layout.setContentsMargins(60, 50, 60, 50) # Căn lề trong (trên, phải, dưới, trái)
        frame_layout.setSpacing(15)

        # 1. Khu vực Logo
        logo_layout = QHBoxLayout()
        logo_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        lbl_logo_truong = QLabel()
        pixmap_truong = QPixmap("logo_hcmus.png")
        lbl_logo_truong.setPixmap(pixmap_truong.scaledToHeight(110, Qt.TransformationMode.SmoothTransformation))
        if pixmap_truong.isNull():
            lbl_logo_truong.setText("[Logo HCMUS]")
            lbl_logo_truong.setStyleSheet("border: 2px dashed gray; padding: 20px; color: gray;")
            
        lbl_logo_khoa = QLabel()
        pixmap_khoa = QPixmap("logo_toan_tin.png")
        lbl_logo_khoa.setPixmap(pixmap_khoa.scaledToHeight(110, Qt.TransformationMode.SmoothTransformation))
        if pixmap_khoa.isNull():
            lbl_logo_khoa.setText("[Logo Toán - Tin]")
            lbl_logo_khoa.setStyleSheet("border: 2px dashed gray; padding: 20px; color: gray;")

        logo_layout.addWidget(lbl_logo_truong)
        logo_layout.addSpacing(50)
        logo_layout.addWidget(lbl_logo_khoa)

        # 2. Tiêu đề phần mềm
        lbl_title = QLabel("PHẦN MỀM GIẢI QUY HOẠCH TUYẾN TÍNH")
        lbl_title.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_title.setStyleSheet("color: #1a365d; margin-top: 15px;")
        
        lbl_subtitle = QLabel("Trượt hàm mục tiêu - Bland - Hai Pha - Dantzig")
        lbl_subtitle.setFont(QFont("Arial", 14))
        lbl_subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_subtitle.setStyleSheet("color: #4a5568; font-style: italic;")
        
        # 3. Thông tin Học thuật
        lbl_course = QLabel("Môn học: Quy hoạch tuyến tính\nGiảng viên phụ trách: Thầy Nguyễn Lê Hoàng Anh")
        lbl_course.setFont(QFont("Arial", 12))
        lbl_course.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_course.setStyleSheet("color: #2b6cb0; margin-top: 15px; line-height: 1.5;")
        
        # 4. Thông tin Nhóm
        lbl_team = QLabel("Thực hiện bởi nhóm:\nMy - Vy - Đạt - Duyên - Hiếu")
        lbl_team.setFont(QFont("Arial", 13, QFont.Weight.Bold))
        lbl_team.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_team.setStyleSheet("color: #c53030; margin-top: 10px; margin-bottom: 25px;")
        
        # 5. Nút Bắt đầu
        btn_start = QPushButton("Bắt đầu tính toán")
        btn_start.setFixedSize(220, 50)
        btn_start.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        btn_start.setStyleSheet("""
            QPushButton {
                background-color: #3182ce; 
                color: white; 
                border-radius: 8px;
                border: none;
            }
            QPushButton:hover {
                background-color: #2b6cb0;
            }
        """)
        btn_start.clicked.connect(self.chuyen_sang_workspace)
        
        # Lắp ráp mọi thứ vào Khung
        frame_layout.addLayout(logo_layout)
        frame_layout.addWidget(lbl_title)
        frame_layout.addWidget(lbl_subtitle)
        frame_layout.addWidget(lbl_course)
        frame_layout.addWidget(lbl_team)
        frame_layout.addWidget(btn_start, alignment=Qt.AlignmentFlag.AlignCenter)
        
        frame_container.setLayout(frame_layout)
        main_layout.addWidget(frame_container, alignment=Qt.AlignmentFlag.AlignCenter)
        
        widget.setLayout(main_layout)
        return widget

    # --- MÀN HÌNH 2: WORKSPACE ---
    def tao_man_hinh_workspace(self):
        widget = QWidget()
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # PANEL TRÁI
        left_panel = QFrame()
        left_panel.setStyleSheet("background-color: #f7fafc; border-right: 1px solid #e2e8f0;")
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(15, 15, 15, 15)
        
        header_layout = QHBoxLayout()
        btn_back = QPushButton("<- Quay lại")
        btn_back.setStyleSheet("border: none; color: #3182ce; font-weight: bold;")
        btn_back.clicked.connect(self.chuyen_ve_welcome)
        lbl_input = QLabel("Nhập liệu Bài toán")
        lbl_input.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        header_layout.addWidget(btn_back)
        header_layout.addStretch()
        header_layout.addWidget(lbl_input)
        
        config_layout = QHBoxLayout()
        lbl_bien = QLabel("Số biến:")
        self.spin_bien = QSpinBox()
        self.spin_bien.setRange(2, 50)
        self.spin_bien.setValue(2)
        lbl_rb = QLabel("Số ràng buộc:")
        self.spin_rb = QSpinBox()
        self.spin_rb.setRange(1, 50)
        self.spin_rb.setValue(3)
        btn_tao_bang = QPushButton("Tạo Bảng")
        btn_tao_bang.setStyleSheet("background-color: #2b6cb0; color: white; padding: 5px 10px; border-radius: 4px;")
        btn_tao_bang.clicked.connect(self.cap_nhat_bang_nhap_lieu)

        config_layout.addWidget(lbl_bien)
        config_layout.addWidget(self.spin_bien)
        config_layout.addWidget(lbl_rb)
        config_layout.addWidget(self.spin_rb)
        config_layout.addStretch()
        config_layout.addWidget(btn_tao_bang)

        self.tab_nhap_lieu = QTabWidget()
        
        self.tab_nhap_tay = QWidget()
        tab1_layout = QVBoxLayout()
        self.bang_ma_tran = BangNhapLieu()
        self.bang_ma_tran.setAlternatingRowColors(True)
        tab1_layout.addWidget(self.bang_ma_tran)
        
        btn_giai = QPushButton("🚀 GIẢI BÀI TOÁN")
        btn_giai.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        btn_giai.setStyleSheet("background-color: #38a169; color: white; padding: 15px; border-radius: 5px;")
        btn_giai.clicked.connect(self.thu_thap_du_lieu_va_giai)
        tab1_layout.addWidget(btn_giai)
        self.tab_nhap_tay.setLayout(tab1_layout)
        
        self.tab_nhap_file = QWidget()
        tab2_layout = QVBoxLayout()
        lbl_file = QLabel("Chức năng tải file Excel đang được phát triển...")
        lbl_file.setStyleSheet("color: gray;")
        btn_upload = QPushButton("Duyệt tìm file .xlsx")
        btn_upload.setFixedSize(150, 40)
        tab2_layout.addWidget(lbl_file, alignment=Qt.AlignmentFlag.AlignCenter)
        tab2_layout.addWidget(btn_upload, alignment=Qt.AlignmentFlag.AlignCenter)
        self.tab_nhap_file.setLayout(tab2_layout)
        
        self.tab_nhap_lieu.addTab(self.tab_nhap_tay, "Nhập trực tiếp (≤ 10)")
        self.tab_nhap_lieu.addTab(self.tab_nhap_file, "Tải file Excel (> 10)")

        left_layout.addLayout(header_layout)
        left_layout.addWidget(QLabel("---"))
        left_layout.addLayout(config_layout)
        left_layout.addWidget(self.tab_nhap_lieu)
        left_panel.setLayout(left_layout)
        
        # PANEL PHẢI (KẾT QUẢ)
        right_panel = QFrame()
        right_panel.setStyleSheet("background-color: #ffffff;")
        right_layout = QVBoxLayout()
        lbl_output = QLabel("Khu vực Kết quả & Lộ trình thuật toán")
        lbl_output.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        right_layout.addWidget(lbl_output)
        
        self.tab_ket_qua = QTabWidget()
        
        self.tab_tong_quan = QWidget()
        tq_layout = QVBoxLayout()
        self.txt_ket_qua = QTextEdit()
        self.txt_ket_qua.setReadOnly(True)
        self.txt_ket_qua.setStyleSheet("font-family: 'Courier New', Consolas; font-size: 15px; background-color: #1e1e1e; color: #4ade80; padding: 10px; border-radius: 5px;")
        tq_layout.addWidget(self.txt_ket_qua)
        self.tab_tong_quan.setLayout(tq_layout)
        
        # --- TAB 2: LỘ TRÌNH ĐƠN HÌNH (MA TRẬN GAUSS-JORDAN) ---
        self.tab_lo_trinh = QWidget()
        lt_layout = QVBoxLayout()
        lbl_file = QLabel("Tải lên file Excel (.xlsx) theo đúng định dạng mẫu")
        lbl_file.setStyleSheet("color: #4a5568; font-weight: bold;")
        
        btn_upload = QPushButton("Duyệt tìm file .xlsx")
        btn_upload.setFixedSize(200, 45)
        btn_upload.setStyleSheet("background-color: #ed8936; color: white; font-weight: bold; font-size: 14px; border-radius: 5px;")
        
        # ---> THÊM DÒNG NÀY ĐỂ KÍCH HOẠT NÚT BẤM <---
        btn_upload.clicked.connect(self.tai_file_excel)
        
        tab2_layout.addWidget(lbl_file, alignment=Qt.AlignmentFlag.AlignCenter)
        tab2_layout.addWidget(btn_upload, alignment=Qt.AlignmentFlag.AlignCenter)
        self.tab_nhap_file.setLayout(tab2_layout)

        # 1. Thanh điều hướng (Toolbar)
        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("◀ Bước trước")
        self.btn_prev.clicked.connect(self.xem_buoc_truoc)
        self.btn_prev.setStyleSheet("background-color: #edf2f7; font-weight: bold; padding: 5px;")
        
        self.lbl_buoc = QLabel("Chưa có dữ liệu ma trận")
        self.lbl_buoc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_buoc.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.lbl_buoc.setStyleSheet("color: #2b6cb0;")
        
        self.btn_next = QPushButton("Bước sau ▶")
        self.btn_next.clicked.connect(self.xem_buoc_sau)
        self.btn_next.setStyleSheet("background-color: #edf2f7; font-weight: bold; padding: 5px;")
        
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.lbl_buoc, stretch=1)
        nav_layout.addWidget(self.btn_next)
        
        # 2. Bảng hiển thị ma trận
        self.bang_don_hinh = QTableWidget()
        self.bang_don_hinh.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers) # Khóa, chỉ cho phép đọc
        self.bang_don_hinh.setStyleSheet("""
            QTableWidget { background-color: #ffffff; gridline-color: #cbd5e0; font-size: 14px; }
            QHeaderView::section { background-color: #2d3748; color: white; font-weight: bold; padding: 5px; }
            QTableWidget::item { padding: 5px; }
        """)
        
        lt_layout.addLayout(nav_layout)
        lt_layout.addWidget(self.bang_don_hinh)
        self.tab_lo_trinh.setLayout(lt_layout)

        # Tab 3: Không Gian Hình Học
        self.tab_do_thi = QWidget()
        dt_layout = QVBoxLayout()
        # Tạo khung canvas trắng của matplotlib
        self.fig = Figure(figsize=(6, 5), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        dt_layout.addWidget(self.canvas)
        self.tab_do_thi.setLayout(dt_layout)

        self.tab_ket_qua.addTab(self.tab_tong_quan, "Báo cáo Tổng quan")
        self.tab_ket_qua.addTab(self.tab_lo_trinh, "Lộ trình Đơn hình")
        self.tab_ket_qua.addTab(self.tab_do_thi, "Đồ thị Hình học")

        right_layout.addWidget(self.tab_ket_qua)
        right_panel.setLayout(right_layout)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        
        # Đặt tỷ lệ kích thước ban đầu (chia đôi màn hình)
        splitter.setSizes([650, 650])
        
        main_layout.addWidget(splitter)
        widget.setLayout(main_layout)
        self.cap_nhat_bang_nhap_lieu()
        return widget

    def chuyen_sang_workspace(self):
        self.stacked_widget.setCurrentIndex(1)
        
    def chuyen_ve_welcome(self):
        self.stacked_widget.setCurrentIndex(0)

    def cap_nhat_bang_nhap_lieu(self):
        so_bien = self.spin_bien.value()
        so_rb = self.spin_rb.value()
        
        if so_bien > 10 or so_rb > 10:
            self.tab_nhap_lieu.setTabEnabled(0, False)
            self.tab_nhap_lieu.setCurrentIndex(1)
            return
        else:
            self.tab_nhap_lieu.setTabEnabled(0, True)
            self.tab_nhap_lieu.setCurrentIndex(0)

        self.bang_ma_tran.setRowCount(0)
        self.bang_ma_tran.setColumnCount(0)
        self.bang_ma_tran.clear()

        self.bang_ma_tran.setRowCount(so_rb + 2)
        self.bang_ma_tran.setColumnCount(so_bien + 3)

        hang_labels = ["Hàm f(x)"] + [f"Ràng buộc {i+1}" for i in range(so_rb)] + ["Dấu của biến"]
        self.bang_ma_tran.setVerticalHeaderLabels(hang_labels)
        cot_labels = ["Mục tiêu"] + [f"x{i+1}" for i in range(so_bien)] + ["Dấu (≤, ≥, =)", "Hệ số b"]
        self.bang_ma_tran.setHorizontalHeaderLabels(cot_labels)
        
        self.bang_ma_tran.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.bang_ma_tran.horizontalHeader().setDefaultSectionSize(60)
        self.bang_ma_tran.setColumnWidth(0, 90)
        self.bang_ma_tran.setColumnWidth(so_bien + 1, 100)
        self.bang_ma_tran.setColumnWidth(so_bien + 2, 80)
        self.bang_ma_tran.verticalHeader().setDefaultSectionSize(40) 

        self.bang_ma_tran.setStyleSheet("""
            QTableWidget { background-color: #ffffff; gridline-color: #cbd5e0; border: 1px solid #e2e8f0; font-size: 14px; }
            QHeaderView::section { background-color: #edf2f7; color: #2d3748; font-weight: bold; border: 1px solid #cbd5e0; padding: 5px; }
            QComboBox { padding: 4px; border: 1px solid #a0aec0; border-radius: 4px; background-color: #f7fafc; font-weight: bold; }
            QComboBox:hover { border-color: #3182ce; }
        """)

        combo_min_max = QComboBox()
        combo_min_max.addItems(["min", "max"])
        self.bang_ma_tran.setCellWidget(0, 0, combo_min_max)

        for i in range(1, so_rb + 1):
            combo_dau = QComboBox()
            combo_dau.addItems(["<=", ">=", "="])
            self.bang_ma_tran.setCellWidget(i, so_bien + 1, combo_dau)

        for j in range(so_bien):
            combo_dau_bien = QComboBox()
            combo_dau_bien.addItems([">=0", "<=0", "free"])
            self.bang_ma_tran.setCellWidget(so_rb + 1, j + 1, combo_dau_bien)

        brush_xam = QBrush(QColor("#e2e8f0"))
        def khoa_o(row, col):
            item = QTableWidgetItem()
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            item.setBackground(brush_xam)
            self.bang_ma_tran.setItem(row, col, item)

        khoa_o(0, so_bien + 1)
        khoa_o(0, so_bien + 2)
        khoa_o(so_rb + 1, 0)
        khoa_o(so_rb + 1, so_bien + 1)
        khoa_o(so_rb + 1, so_bien + 2)

    # ==========================================================
    # CÁC HÀM XỬ LÝ HIỂN THỊ MA TRẬN ĐƠN HÌNH
    # ==========================================================
    def xem_buoc_truoc(self):
        if hasattr(self, 'lich_su_bang_ui') and self.buoc_hien_tai > 0:
            self.buoc_hien_tai -= 1
            self.hien_thi_bang_don_hinh()

    def xem_buoc_sau(self):
        if hasattr(self, 'lich_su_bang_ui') and self.buoc_hien_tai < len(self.lich_su_bang_ui) - 1:
            self.buoc_hien_tai += 1
            self.hien_thi_bang_don_hinh()

    def hien_thi_bang_don_hinh(self):
        if not hasattr(self, 'lich_su_bang_ui') or len(self.lich_su_bang_ui) == 0:
            return
            
        du_lieu = self.lich_su_bang_ui[self.buoc_hien_tai]
        ma_tran = du_lieu["bang"]
        co_so = du_lieu["co_so"]
        
        so_dong, so_cot = ma_tran.shape
        self.bang_don_hinh.setRowCount(so_dong)
        self.bang_don_hinh.setColumnCount(so_cot)
        
        # Cập nhật Label chỉ số vòng lặp
        self.lbl_buoc.setText(f"Ma trận vòng lặp thứ: {self.buoc_hien_tai + 1} / {len(self.lich_su_bang_ui)}")
        
        # Nạp Tiêu đề Cột (X1, X2, ..., Vế phải)
        cot_labels = [f"X{j+1}" for j in range(so_cot - 1)] + ["Vế phải (b)"]
        self.bang_don_hinh.setHorizontalHeaderLabels(cot_labels)
        
        # Nạp Tiêu đề Dòng (Biến cơ sở, Hàm Z)
        dong_labels = [f"Cơ sở (X{c+1})" for c in co_so] + ["Hàm Z (Δ)"]
        self.bang_don_hinh.setVerticalHeaderLabels(dong_labels)
        
        # Đổ từng con số vào bảng
        for i in range(so_dong):
            for j in range(so_cot):
                val = ma_tran[i, j]
                # Khử sai số máy tính (-0.00000000001 thành 0)
                if abs(val) < 1e-9: val = 0.0
                
                # Format hiển thị: nếu là số nguyên in gọn, nếu số thập phân lấy 3 số lẻ
                if abs(val - round(val)) < 1e-9:
                    str_val = f"{int(round(val))}"
                else:
                    str_val = f"{val:.3f}"
                
                item = QTableWidgetItem(str_val)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                
                # Làm nổi bật cột Vế phải (b) và Dòng hàm Z
                if j == so_cot - 1:
                    item.setBackground(QBrush(QColor("#ebf8ff"))) # Nền xanh nhạt
                if i == so_dong - 1:
                    item.setBackground(QBrush(QColor("#fefcbf"))) # Nền vàng nhạt
                    item.setFont(QFont("Arial", 11, QFont.Weight.Bold))
                    
                self.bang_don_hinh.setItem(i, j, item)
                
        # Căn chỉnh độ rộng cột tự động
        self.bang_don_hinh.resizeColumnsToContents()

    def thu_thap_du_lieu_va_giai(self):
        try:
            so_bien = self.spin_bien.value()
            so_rb = self.spin_rb.value()
            loai_muc_tieu = self.bang_ma_tran.cellWidget(0, 0).currentText()
            
            he_so_f = []
            for j in range(1, so_bien + 1):
                item = self.bang_ma_tran.item(0, j)
                if item is None or item.text().strip() == "":
                    raise ValueError(f"Vui lòng nhập hệ số hàm f(x) cho biến x{j}")
                he_so_f.append(float(item.text()))
                
            cac_rang_buoc = []
            for i in range(1, so_rb + 1):
                he_so_vt = []
                for j in range(1, so_bien + 1):
                    item = self.bang_ma_tran.item(i, j)
                    if item is None or item.text().strip() == "":
                        raise ValueError(f"Thiếu hệ số tại Ràng buộc {i}, biến x{j}")
                    he_so_vt.append(float(item.text()))
                dau_rb = self.bang_ma_tran.cellWidget(i, so_bien + 1).currentText()
                item_b = self.bang_ma_tran.item(i, so_bien + 2)
                if item_b is None or item_b.text().strip() == "":
                    raise ValueError(f"Thiếu vế phải 'b' tại Ràng buộc {i}")
                gia_tri_b = float(item_b.text())
                cac_rang_buoc.append((he_so_vt, dau_rb, gia_tri_b))
                
            dau_cua_bien = []
            for j in range(1, so_bien + 1):
                dau = self.bang_ma_tran.cellWidget(so_rb + 1, j).currentText()
                dau_cua_bien.append(dau)
                
            bai_toan = BaiToanQuyHoachTuyenTinh()
            bai_toan.nap_du_lieu_tu_bien_ngoai(loai_muc_tieu, so_bien, he_so_f, cac_rang_buoc, dau_cua_bien)
            bai_toan.buoc3_chuan_hoa_bai_toan()
            
            self.txt_ket_qua.clear()
            self.tab_ket_qua.setCurrentIndex(0)
            print("🚀 HỆ THỐNG ĐANG XỬ LÝ...\n" + "="*50)
            bai_toan.giai_va_xuat_ket_qua()
            
            # ---> BỔ SUNG ĐOẠN NÀY ĐỂ BẬT BẢNG LỘ TRÌNH <---
            if hasattr(bai_toan, 'lich_su_bang_hien_tai') and len(bai_toan.lich_su_bang_hien_tai) > 0:
                self.lich_su_bang_ui = bai_toan.lich_su_bang_hien_tai
                self.buoc_hien_tai = 0
                self.hien_thi_bang_don_hinh()
                self.tab_ket_qua.setTabEnabled(1, True)
            else:
                self.tab_ket_qua.setTabEnabled(1, False)
            
            # ---> BỔ SUNG ĐOẠN NÀY ĐỂ KÍCH HOẠT VẼ ĐỒ THỊ <---
            if so_bien == 2:
                self.tab_ket_qua.setTabEnabled(2, True) # Mở khóa Tab đồ thị
                self.fig.clear() # Xóa đồ thị cũ
                ax_do_thi = self.fig.add_subplot(111)
                phuong_phap_truot_ham_muc_tieu(bai_toan, bai_toan.duong_bland, bai_toan.duong_dantzig, ax=ax_do_thi)
                self.fig.tight_layout()
                self.canvas.draw() # Render lại giao diện PyQt
            else:
                self.tab_ket_qua.setTabEnabled(2, False) # 3 biến trở lên thì khóa Tab này lại

        except ValueError as e:
            QMessageBox.warning(self, "Lỗi nhập liệu", str(e))
        except Exception as e:
            QMessageBox.critical(self, "Lỗi hệ thống", f"Đã xảy ra lỗi không xác định: {str(e)}")

    def tai_file_excel(self):
        # Mở hộp thoại chọn file
        duong_dan_file, _ = QFileDialog.getOpenFileName(
            self, "Chọn file dữ liệu Excel", "", "Excel Files (*.xlsx *.xls)"
        )
        
        if not duong_dan_file:
            return # Người dùng bấm Cancel không chọn file nữa
            
        try:
            # Đọc file Excel không lấy tiêu đề mặc định (header=None)
            df = pd.read_excel(duong_dan_file, header=None)
            
            # --- 1. BÓC TÁCH DỮ LIỆU ---
            loai_muc_tieu = str(df.iloc[0, 1]).strip().lower()
            
            # Đọc hệ số f(x) ở dòng 1 (bỏ cột đầu tiên, lấy tới khi gặp ô trống/NaN)
            he_so_f = []
            for val in df.iloc[1, 1:]:
                if pd.notna(val) and str(val).strip() != "":
                    he_so_f.append(float(val))
            so_bien = len(he_so_f)
            
            # Đọc dấu của biến ở dòng cuối cùng
            dong_cuoi_idx = len(df) - 1
            dau_cua_bien = []
            for val in df.iloc[dong_cuoi_idx, 1:so_bien+1]:
                dau_cua_bien.append(str(val).strip())
                
            # Đọc các ràng buộc (từ dòng 2 đến sát dòng cuối)
            cac_rang_buoc = []
            for i in range(2, dong_cuoi_idx):
                dong_data = df.iloc[i].tolist()
                he_so_vt = [float(x) for x in dong_data[1:1+so_bien]]
                dau_rb = str(dong_data[1+so_bien]).strip()
                gia_tri_vp = float(dong_data[1+so_bien+1])
                cac_rang_buoc.append((he_so_vt, dau_rb, gia_tri_vp))
                
            # --- 2. BƠM DỮ LIỆU VÀO BACKEND ĐỂ GIẢI ---
            bai_toan = BaiToanQuyHoachTuyenTinh()
            bai_toan.nap_du_lieu_tu_bien_ngoai(loai_muc_tieu, so_bien, he_so_f, cac_rang_buoc, dau_cua_bien)
            bai_toan.buoc3_chuan_hoa_bai_toan()
            
            # --- 3. HIỂN THỊ KẾT QUẢ LÊN GIAO DIỆN ---
            self.txt_ket_qua.clear()
            self.tab_ket_qua.setCurrentIndex(0)
            print(f"📂 Đã đọc thành công file: {duong_dan_file.split('/')[-1]}")
            print(f"📌 Bài toán kích thước lớn: {so_bien} biến, {len(cac_rang_buoc)} ràng buộc")
            print("🚀 HỆ THỐNG ĐANG XỬ LÝ...\n" + "="*50)
            bai_toan.giai_va_xuat_ket_qua()
            
            # Bật/Tắt Tab Ma trận Đơn hình
            if hasattr(bai_toan, 'lich_su_bang_hien_tai') and len(bai_toan.lich_su_bang_hien_tai) > 0:
                self.lich_su_bang_ui = bai_toan.lich_su_bang_hien_tai
                self.buoc_hien_tai = 0
                self.hien_thi_bang_don_hinh()
                self.tab_ket_qua.setTabEnabled(1, True)
            else:
                self.tab_ket_qua.setTabEnabled(1, False)
                
            # Bật/Tắt Tab Đồ thị Hình học
            if so_bien == 2:
                self.tab_ket_qua.setTabEnabled(2, True)
                self.fig.clear()
                ax_do_thi = self.fig.add_subplot(111)
                from truot_ham_muc_tieu import phuong_phap_truot_ham_muc_tieu
                phuong_phap_truot_ham_muc_tieu(bai_toan, bai_toan.duong_bland, bai_toan.duong_dantzig, ax=ax_do_thi)
                self.canvas.draw()
            else:
                self.tab_ket_qua.setTabEnabled(2, False)

            QMessageBox.information(self, "Thành công", "Đã nạp và giải xong dữ liệu từ file Excel!")

        except Exception as e:
            QMessageBox.critical(self, "Lỗi đọc file Excel", f"File không đúng định dạng. Chi tiết lỗi:\n{str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = OptimizationSolverApp()
    window.show()
    sys.exit(app.exec())