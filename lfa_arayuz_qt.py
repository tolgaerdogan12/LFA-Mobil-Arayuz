# Dosya Adı: lfa_arayuz_qt.py
# Sürüm: V15.0 (Sağ Tık > Görüntüyü Düzenle ve Yeniden Hesapla)

import sys
import os
import json
import csv
import glob
import datetime
import shutil
import threading
from functools import partial

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QComboBox, QTableWidget, 
    QTableWidgetItem, QHeaderView, QFileDialog, QMessageBox, 
    QSplitter, QFrame, QTabWidget, QDialog, QGraphicsScene, 
    QGraphicsView, QGraphicsRectItem, QGraphicsTextItem, QGroupBox,
    QFormLayout, QScrollArea, QAbstractItemView, QSizePolicy, QMenu
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRectF, QSize
from PyQt5.QtGui import QPixmap, QImage, QPen, QBrush, QColor, QFont, QCursor

import cv2
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import lfa_motoru

# --- ÇALIŞAN THREAD ---
class AnalizWorker(QThread):
    result_ready = pyqtSignal(int, dict); finished_all = pyqtSignal()
    def __init__(self, queue_data, study_name): super().__init__(); self.queue_data = queue_data; self.study_name = study_name; self.is_running = True
    def run(self):
        pending_ids = [k for k, v in self.queue_data.items() if v["status"] == "pending"]
        for iid in pending_ids:
            if not self.is_running: break
            data = self.queue_data[iid]
            res = lfa_motoru.motoru_calistir(data["path"], self.study_name, data["hid"], data.get("kaynak", ""), data.get("matris", ""), data.get("kons", "0"), data.get("notlar", ""))
            self.result_ready.emit(iid, res)
        self.finished_all.emit()
    def stop(self): self.is_running = False

# --- GÖRSEL EDİTÖR (GELİŞMİŞ: Hem Genel Hem Tekil Mod) ---
class QtVisualEditor(QDialog):
    def __init__(self, parent=None, image_path=None, initial_pos=None):
        super().__init__(parent)
        self.setWindowTitle("Manuel Müdahale & Yeniden Hesaplama" if image_path else "ROI Ayarları")
        self.resize(800, 900)
        self.setStyleSheet("background-color: #2b2b2b; color: white;")
        self.layout = QVBoxLayout(self)
        
        self.image_path = image_path
        self.is_correction_mode = image_path is not None
        
        top_layout = QHBoxLayout()
        if not self.is_correction_mode:
            btn_load = QPushButton("Fotoğraf Seç"); btn_load.clicked.connect(self.load_image); top_layout.addWidget(btn_load)
        
        btn_save = QPushButton("YENİDEN HESAPLA & KAYDET" if self.is_correction_mode else "Kaydet ve Çık")
        btn_save.clicked.connect(self.save_config)
        btn_save.setStyleSheet("background-color: #2ecc71; color: black; font-weight: bold;")
        top_layout.addWidget(btn_save)
        self.layout.addLayout(top_layout)
        
        self.scene = QGraphicsScene(); self.view = QGraphicsView(self.scene); self.view.setStyleSheet("background-color: #1e1e1e; border: 1px solid #444;")
        self.layout.addWidget(self.view)
        
        self.scale_factor = 0.5
        self.std_w = int(lfa_motoru.Config.STD_WIDTH * self.scale_factor)
        self.std_h = int(lfa_motoru.Config.STD_HEIGHT * self.scale_factor)
        self.roi_w = int(lfa_motoru.Config.ROI_TEST_STRIP[2] * self.scale_factor)
        self.tol = int(lfa_motoru.Config.POS_TOLERANCE * self.scale_factor)
        
        # Başlangıç pozisyonları (Varsayılan veya o satıra özel)
        self.c_pos = initial_pos["C"] if initial_pos else lfa_motoru.Config.LINE_LOCATIONS["C"]["pos"]
        self.t_pos = initial_pos["T"] if initial_pos else lfa_motoru.Config.LINE_LOCATIONS["T"]["pos"]
        
        self.info_lbl = QLabel(f"C: {self.c_pos} | T: {self.t_pos}")
        self.layout.addWidget(self.info_lbl)
        
        if self.is_correction_mode: self.load_specific_image()

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Resim Seç", "", "Images (*.jpg *.jpeg *.png)")
        if path: self.process_and_show(path)

    def load_specific_image(self):
        if self.image_path: self.process_and_show(self.image_path)

    def process_and_show(self, path):
        try:
            img = lfa_motoru.resim_oku_guvenli(path); pre = lfa_motoru.ImagePreprocessor()
            warped = pre.warp_image(img); strip = pre.extract_strip(warped)
            strip = cv2.cvtColor(strip, cv2.COLOR_BGR2RGB); strip = cv2.resize(strip, (self.std_w, self.std_h))
            h, w, ch = strip.shape; qimg = QImage(strip.data, w, h, ch * w, QImage.Format_RGB888)
            self.scene.clear(); self.scene.addPixmap(QPixmap.fromImage(qimg)); self.draw_boxes()
        except Exception as e: QMessageBox.critical(self, "Hata", str(e))

    def draw_boxes(self):
        cx = self.std_w / 2; hw = self.roi_w / 2
        c_y = self.c_pos * self.scale_factor; self.c_rect = DraggableRect(cx - hw, c_y - self.tol, self.roi_w, self.tol * 2, "C", self); self.scene.addItem(self.c_rect)
        t_y = self.t_pos * self.scale_factor; self.t_rect = DraggableRect(cx - hw, t_y - self.tol, self.roi_w, self.tol * 2, "T", self); self.scene.addItem(self.t_rect)

    def update_pos(self, tag, new_y_center):
        real_pos = int(new_y_center / self.scale_factor)
        if tag == "C": self.c_pos = real_pos
        else: self.t_pos = real_pos
        self.info_lbl.setText(f"C: {self.c_pos} | T: {self.t_pos}")

    def save_config(self):
        if self.is_correction_mode:
            # Tekil moddaysa sadece bu değerleri döndür
            self.accept()
        else:
            # Genel moddaysa Config'i güncelle
            lfa_motoru.Config.LINE_LOCATIONS["C"]["pos"] = self.c_pos
            lfa_motoru.Config.LINE_LOCATIONS["T"]["pos"] = self.t_pos
            lfa_motoru.Config.save_config(); self.accept()

    def get_corrected_positions(self):
        return self.c_pos, self.t_pos

class DraggableRect(QGraphicsRectItem):
    def __init__(self, x, y, w, h, tag, parent_dialog):
        super().__init__(x, y, w, h); self.tag = tag; self.parent_dialog = parent_dialog
        color = Qt.green if tag == "C" else Qt.red; self.setPen(QPen(color, 2)); self.setBrush(QBrush(QColor(0, 255, 0, 50) if tag=="C" else QColor(255, 0, 0, 50)))
        self.setFlag(QGraphicsRectItem.ItemIsMovable, True); self.setFlag(QGraphicsRectItem.ItemSendsGeometryChanges, True)
    def mouseReleaseEvent(self, event): super().mouseReleaseEvent(event); self.parent_dialog.update_pos(self.tag, self.sceneBoundingRect().center().y())

# --- ANA PENCERE ---
class LFAMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LFA Analyzer Pro V15.0 - Precision Surgery")
        self.resize(1600, 950)
        self.queue_data = {}; self.next_id = 0; self.worker = None
        self.init_ui(); self.refresh_study_list()
        
        # Stil (Dark Mode)
        self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; color: white; }
            QWidget { color: white; font-family: 'Segoe UI', sans-serif; font-size: 14px; }
            QGroupBox { border: 1px solid #555; border-radius: 5px; margin-top: 10px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 5px; color: #3B8ED0; }
            QLineEdit, QComboBox { background-color: #333; border: 1px solid #555; padding: 5px; border-radius: 3px; color: white; }
            QTableWidget { background-color: #1e1e1e; gridline-color: #444; selection-background-color: #3B8ED0; border: none; }
            QHeaderView::section { background-color: #333; padding: 4px; border: 1px solid #444; color: white; font-weight: bold; }
            QTabWidget::pane { border: 1px solid #444; background-color: #2b2b2b; } 
            QTabBar::tab { background: #333; color: #aaa; padding: 8px 12px; margin-right: 2px; }
            QTabBar::tab:selected { background: #3B8ED0; color: white; font-weight: bold; }
            QPushButton { background-color: #444; border: none; padding: 8px; border-radius: 4px; color: white; }
            QPushButton:hover { background-color: #555; }
            QPushButton#btnStart { background-color: #e74c3c; font-weight: bold; font-size: 16px; }
            QPushButton#btnStart:hover { background-color: #c0392b; }
            QScrollArea { border: none; background-color: #2b2b2b; }
            QWidget#debugContent { background-color: #2b2b2b; }
            QLabel#lblStatus { color: #aaa; font-style: italic; }
            QLabel#infoTag { color: #3B8ED0; font-weight: bold; font-size: 12px; background-color: transparent; }
        """)

    def init_ui(self):
        central = QWidget(); self.setCentralWidget(central); main_layout = QHBoxLayout(central)
        
        # SOL PANEL
        left_panel = QWidget(); left_panel.setFixedWidth(350); left_layout = QVBoxLayout(left_panel); left_layout.setSpacing(15)
        grp_def = QGroupBox("Yeni Kayıt Varsayılanları"); frm_def = QFormLayout()
        self.txt_study = QLineEdit(f"Exp_{datetime.datetime.now().strftime('%Y%m%d')}")
        self.txt_source = QLineEdit(); self.txt_source.setPlaceholderText("Örn: iPhone 13")
        self.txt_matrix = QLineEdit(); self.txt_matrix.setPlaceholderText("Örn: Tam Kan")
        self.txt_conc = QLineEdit(); self.txt_conc.setPlaceholderText("0")
        frm_def.addRow("Çalışma Adı:", self.txt_study); frm_def.addRow("Kaynak:", self.txt_source)
        frm_def.addRow("Matris:", self.txt_matrix); frm_def.addRow("Kons.:", self.txt_conc)
        h_load = QHBoxLayout(); self.combo_studies = QComboBox()
        btn_load = QPushButton("Yükle"); btn_load.clicked.connect(self.load_study)
        btn_refresh = QPushButton("R"); btn_refresh.setFixedWidth(30); btn_refresh.clicked.connect(self.refresh_study_list)
        h_load.addWidget(self.combo_studies); h_load.addWidget(btn_load); h_load.addWidget(btn_refresh)
        frm_def.addRow("Eski Çalışma:", h_load); grp_def.setLayout(frm_def); left_layout.addWidget(grp_def)
        
        btn_add = QPushButton("➕ Fotoğraf Ekle"); btn_add.clicked.connect(self.add_files)
        self.btn_start = QPushButton("▶ ANALİZİ BAŞLAT"); self.btn_start.setObjectName("btnStart"); self.btn_start.clicked.connect(self.start_analysis)
        btn_export = QPushButton("📊 Tidy Excel Export"); btn_export.clicked.connect(self.export_excel)
        btn_pdf = QPushButton("📄 PDF Rapor"); btn_pdf.clicked.connect(self.create_pdf)
        btn_settings = QPushButton("⚙️ Genel Ayarlar (ROI)"); btn_settings.clicked.connect(self.open_settings)
        left_layout.addWidget(btn_add); left_layout.addWidget(self.btn_start); left_layout.addWidget(btn_export)
        left_layout.addWidget(btn_pdf); left_layout.addWidget(btn_settings)
        
        note = QLabel("İPUCU: Tabloya sağ tıklayarak\nhatalı satırları manuel düzeltebilirsiniz."); note.setStyleSheet("color: #aaa; font-size: 11px;"); left_layout.addWidget(note)
        left_layout.addStretch(); self.lbl_status = QLabel("Hazır"); self.lbl_status.setObjectName("lblStatus"); left_layout.addWidget(self.lbl_status); main_layout.addWidget(left_panel)
        
        # SAĞ PANEL
        right_panel = QWidget(); right_layout = QVBoxLayout(right_panel)
        self.table = QTableWidget(); self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels(["D", "Hasta ID", "Kaynak", "Matris", "Kons.", "Not", "C", "T", "Oran"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.cellChanged.connect(self.on_cell_changed)
        self.table.itemSelectionChanged.connect(self.on_table_selection)
        
        # SAĞ TIK MENÜSÜ EKLENİYOR
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_context_menu)
        
        right_layout.addWidget(self.table, stretch=1)
        self.tabs = QTabWidget(); self.tabs.setFixedHeight(350)
        self.tab_graph = QWidget(); l_graph = QVBoxLayout(self.tab_graph)
        self.fig = Figure(figsize=(5, 3), dpi=100, facecolor='#2b2b2b'); self.ax = self.fig.add_subplot(111); self.ax.set_facecolor('#2b2b2b')
        self.ax.tick_params(colors='white'); self.ax.xaxis.label.set_color('white'); self.ax.yaxis.label.set_color('white'); self.ax.title.set_color('white')
        for spine in self.ax.spines.values(): spine.set_edgecolor('#555')
        self.graph_view = FigureCanvas(self.fig); l_graph.addWidget(self.graph_view); self.tabs.addTab(self.tab_graph, "📈 Sinyal Grafiği")
        
        self.tab_debug = QWidget(); self.scroll_debug = QScrollArea(); self.scroll_debug.setWidgetResizable(True)
        self.debug_content = QWidget(); self.debug_content.setObjectName("debugContent"); self.debug_layout = QHBoxLayout(self.debug_content); self.debug_layout.setAlignment(Qt.AlignLeft)
        self.scroll_debug.setWidget(self.debug_content); l_debug = QVBoxLayout(self.tab_debug); l_debug.addWidget(self.scroll_debug); self.tabs.addTab(self.tab_debug, "🔧 Teknik Görseller")
        right_layout.addWidget(self.tabs); main_layout.addWidget(right_panel)

    # --- YENİ: SAĞ TIK MENÜSÜ ---
    def show_context_menu(self, pos):
        item = self.table.itemAt(pos)
        if not item: return
        menu = QMenu(self)
        menu.setStyleSheet("QMenu { background-color: #333; color: white; } QMenu::item:selected { background-color: #3B8ED0; }")
        
        action_edit = menu.addAction("✏️ Görüntüyü Düzenle & Yeniden Hesapla")
        action = menu.exec_(self.table.viewport().mapToGlobal(pos))
        
        if action == action_edit:
            self.open_manual_correction(item.row())

    def open_manual_correction(self, row):
        row_id = self.table.item(row, 0).data(Qt.UserRole)
        data = self.queue_data.get(row_id)
        
        if not data or not data["path"]: return
        
        # Mevcut C/T pozisyonlarını tahmin etmeye çalış veya varsayılanı al
        # Şimdilik varsayılanı gönderiyoruz, ileride veritabanından o anki pozisyonu çekebiliriz
        dlg = QtVisualEditor(self, image_path=data["path"])
        if dlg.exec_() == QDialog.Accepted:
            new_c, new_t = dlg.get_corrected_positions()
            
            # Yeniden Hesapla (Motoru Çağır)
            self.lbl_status.setText("Yeniden hesaplanıyor...")
            QApplication.processEvents()
            
            # Motorun tekil analiz fonksiyonunu çağır
            res = lfa_motoru.tekil_yeniden_analiz(
                image_path=data["path"],
                output_dir=data.get("res_dir", lfa_motoru.Config.BASE_OUTPUT_DIR),
                c_pos=new_c,
                t_pos=new_t
            )
            
            if res["success"]:
                # Veriyi Güncelle
                data["c"] = res["c_val"]
                data["t"] = res["t_val"]
                data["oran"] = res["ratio"]
                data["graph"] = res["graph_path"]
                
                # Tabloyu Güncelle
                self.table.blockSignals(True)
                self.table.item(row, 6).setText(str(int(res["c_val"])))
                self.table.item(row, 7).setText(str(int(res["t_val"])))
                self.table.item(row, 8).setText(str(res["ratio"]))
                self.table.blockSignals(False)
                
                self.show_images(data) # Görselleri yenile
                QMessageBox.information(self, "Başarılı", "Analiz güncellendi!")
                self.lbl_status.setText("Hazır")
            else:
                QMessageBox.critical(self, "Hata", f"Yeniden hesaplama başarısız: {res.get('error')}")

    # --- STANDART FONKSİYONLAR ---
    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Fotoğraf Seç", "", "Images (*.jpg *.jpeg *.png)")
        if not files: return
        src = self.txt_source.text() or "Bilinmiyor"; mat = self.txt_matrix.text() or "Bilinmiyor"; conc = self.txt_conc.text() or "0"
        self.table.blockSignals(True)
        for p in files:
            fname = os.path.basename(p); hid = "ID_" + os.path.splitext(fname)[0]; row_id = self.next_id
            self.queue_data[row_id] = {"path": p, "hid": hid, "kaynak": src, "matris": mat, "kons": conc, "notlar": "", "status": "pending"}
            self.next_id += 1
            r = self.table.rowCount(); self.table.insertRow(r)
            self.add_table_item(r, 0, "⏳", row_id, False); self.add_table_item(r, 1, hid, row_id, True)
            self.add_table_item(r, 2, src, row_id, True); self.add_table_item(r, 3, mat, row_id, True)
            self.add_table_item(r, 4, conc, row_id, True); self.add_table_item(r, 5, "", row_id, True)
            self.add_table_item(r, 6, "-", row_id, False); self.add_table_item(r, 7, "-", row_id, False); self.add_table_item(r, 8, "-", row_id, False)
        self.table.blockSignals(False); self.lbl_status.setText(f"{len(files)} dosya eklendi.")

    def add_table_item(self, row, col, text, row_id, editable=True):
        item = QTableWidgetItem(str(text)); item.setData(Qt.UserRole, row_id)
        if not editable: item.setFlags(item.flags() ^ Qt.ItemIsEditable) 
        self.table.setItem(row, col, item)

    def on_cell_changed(self, row, col):
        item = self.table.item(row, col)
        if not item: return
        row_id = item.data(Qt.UserRole); val = item.text()
        if row_id in self.queue_data:
            d = self.queue_data[row_id]
            if col == 1: d["hid"] = val
            elif col == 2: d["kaynak"] = val
            elif col == 3: d["matris"] = val
            elif col == 4: d["kons"] = val
            elif col == 5: d["notlar"] = val

    def on_table_selection(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows: return
        row = rows[0].row(); row_id = self.table.item(row, 0).data(Qt.UserRole); data = self.queue_data.get(row_id)
        if data and data.get("status") == "done": self.show_images(data)
        else: self.clear_images()

    def clear_images(self):
        self.ax.clear(); self.graph_view.draw()
        for i in reversed(range(self.debug_layout.count())): self.debug_layout.itemAt(i).widget().setParent(None)

    def show_images(self, data):
        self.ax.clear()
        if data.get("graph") and os.path.exists(data["graph"]):
            img = matplotlib.image.imread(data["graph"]); self.ax.imshow(img); self.ax.axis('off'); self.graph_view.draw()
        for i in reversed(range(self.debug_layout.count())): self.debug_layout.itemAt(i).widget().setParent(None)
        if data.get("res_dir"):
            files = sorted(glob.glob(os.path.join(data["res_dir"], "*.jpg")))
            for f in files:
                container = QWidget(); vbox = QVBoxLayout(container)
                lbl_img = QLabel(); pix = QPixmap(f)
                if not pix.isNull(): lbl_img.setPixmap(pix.scaled(200, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                lbl_txt = QLabel(os.path.basename(f)); lbl_txt.setObjectName("infoTag"); lbl_txt.setAlignment(Qt.AlignCenter)
                vbox.addWidget(lbl_img); vbox.addWidget(lbl_txt); self.debug_layout.addWidget(container)

    def start_analysis(self):
        self.worker = AnalizWorker(self.queue_data, self.txt_study.text())
        self.worker.result_ready.connect(self.on_result); self.worker.finished_all.connect(self.on_finished)
        self.btn_start.setEnabled(False); self.btn_start.setText("⏳ İŞLENİYOR..."); self.worker.start()

    def on_result(self, row_id, res):
        d = self.queue_data[row_id]
        if res["success"]:
            d.update({"status": "done", "c": res["c_val"], "t": res["t_val"], "oran": res["ratio"], "res_dir": res["output_dir"], "graph": res["graph_path"]})
            self.table.blockSignals(True)
            for r in range(self.table.rowCount()):
                if self.table.item(r, 0).data(Qt.UserRole) == row_id:
                    self.table.item(r, 0).setText("✅"); self.table.item(r, 6).setText(str(int(res["c_val"]))); self.table.item(r, 7).setText(str(int(res["t_val"]))); self.table.item(r, 8).setText(str(res["ratio"])); break
            self.table.blockSignals(False)
        else:
            for r in range(self.table.rowCount()):
                if self.table.item(r, 0).data(Qt.UserRole) == row_id: self.table.item(r, 0).setText("❌"); break

    def on_finished(self):
        self.btn_start.setEnabled(True); self.btn_start.setText("▶ ANALİZİ BAŞLAT"); self.lbl_status.setText("Analiz tamamlandı."); self.refresh_study_list()

    def load_study(self):
        st = self.combo_studies.currentText()
        if not st: return
        db = lfa_motoru.DatabaseManager(lfa_motoru.Config.DB_NAME); recs = db.calisma_verilerini_getir(st)
        self.table.setRowCount(0); self.queue_data.clear(); self.next_id = 0; self.table.blockSignals(True)
        for r in recs:
            row_id = self.next_id; self.next_id += 1
            self.queue_data[row_id] = {"status": "done", "hid": r[2], "dosya": r[3], "kaynak": r[4], "matris": r[5], "kons": r[6], "notlar": r[7], "c": r[9], "t": r[10], "oran": r[11], "res_dir": r[12], "graph": r[13]}
            rr = self.table.rowCount(); self.table.insertRow(rr)
            self.add_table_item(rr, 0, "✅", row_id, False); self.add_table_item(rr, 1, r[2], row_id, True); self.add_table_item(rr, 2, r[4], row_id, True); self.add_table_item(rr, 3, r[5], row_id, True)
            self.add_table_item(rr, 4, r[6], row_id, True); self.add_table_item(rr, 5, r[7], row_id, True); self.add_table_item(rr, 6, int(r[9]), row_id, False); self.add_table_item(rr, 7, int(r[10]), row_id, False); self.add_table_item(rr, 8, r[11], row_id, False)
        self.table.blockSignals(False); self.txt_study.setText(st)

    def refresh_study_list(self):
        db = lfa_motoru.DatabaseManager(lfa_motoru.Config.DB_NAME); self.combo_studies.clear(); self.combo_studies.addItems(db.calisma_listesi_getir())

    def export_excel(self):
        path, _ = QFileDialog.getSaveFileName(self, "Excel Kaydet", "", "CSV Files (*.csv)")
        if not path: return
        try:
            read_counters = {}
            with open(path, 'w', newline='', encoding='utf-8-sig') as f:
                w = csv.writer(f); w.writerow(["Matrix", "Concentration", "Phone", "Read", "C_line", "T_line"])
                for r in range(self.table.rowCount()):
                    rid = self.table.item(r, 0).data(Qt.UserRole); d = self.queue_data.get(rid)
                    if not d: continue
                    mat = d.get("matris", "Unknown"); conc = d.get("kons", "0"); phone = d.get("kaynak", "Unknown")
                    key = (mat, conc, phone); read_counters[key] = read_counters.get(key, 0) + 1
                    w.writerow([mat, conc, phone, read_counters[key], d.get("c", 0), d.get("t", 0)])
            QMessageBox.information(self, "Başarılı", "Tidy Dataset kaydedildi.")
        except Exception as e: QMessageBox.critical(self, "Hata", str(e))

    def create_pdf(self):
        data = [v for v in self.queue_data.values() if v.get("status") == "done"]
        if not data: return
        path, _ = QFileDialog.getSaveFileName(self, "PDF Kaydet", f"{self.txt_study.text()}_Rapor.pdf", "PDF Files (*.pdf)")
        if path:
            success, msg = lfa_motoru.toplu_pdf_olustur(data, path)
            if success: os.startfile(path)
            else: QMessageBox.critical(self, "Hata", msg)

    def open_settings(self): dlg = QtVisualEditor(self); dlg.exec_()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LFAMainWindow()
    window.show()
    sys.exit(app.exec_())