import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import collections
import subprocess
import pandas as pd
from ultralytics import YOLO
import cvzone
import threading
from PIL import Image, ImageTk
import customtkinter as ctk
import plyer
import time
import os
import numpy as np
from twilio.rest import Client
from datetime import datetime


# ──────────────────────────────────────────────
#  Color Palette & Constants
# ──────────────────────────────────────────────
COLORS = {
    "bg_dark":       "#0f1117",
    "sidebar":       "#161b22",
    "card":          "#1c2128",
    "card_hover":    "#252c35",
    "accent":        "#2f81f7",
    "accent_hover":  "#1a6ddb",
    "success":       "#2ea043",
    "success_hover": "#238636",
    "danger":        "#da3633",
    "danger_hover":  "#b62324",
    "warning":       "#d29922",
    "text_primary":  "#e6edf3",
    "text_secondary":"#8b949e",
    "border":        "#30363d",
    "input_bg":      "#0d1117",
}

SIDEBAR_WIDTH = 220
MEDIA_EXTENSIONS_IMAGE = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
MEDIA_EXTENSIONS_VIDEO = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm")


class AccidentDetectionApp:
    """Professional Real-Time & Media Accident Detection Application."""

    # ──────────────────────────────────────────
    #  Initialisation
    # ──────────────────────────────────────────
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("AcciVision \u2014 Intelligent Accident Detection")
        self.root.geometry("1280x780")
        self.root.minsize(1100, 700)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # ── State variables ──
        self.processing = False
        self.cap = None
        self._feed_thread = None   # tracks the active camera thread
        self._session_id = 0       # incremented on every Start; guards stale-thread cleanup
        self.model = YOLO("best.pt")
        self.confidence_threshold = 0.55
        self.last_notification_time = 0
        self.notification_cooldown = 10
        self.accident_frames_threshold = 8   # votes needed inside rolling window
        self.accident_frames_count = 0
        self._vote_window = collections.deque(maxlen=15)  # rolling majority-vote buffer
        self.save_evidence = True
        self.detected_accidents = set()

        # Rolling pre-buffer for live clip saving (~3 s at 30 fps)
        self.live_frame_buffer = collections.deque(maxlen=90)

        # Camera locations
        self.camera_locations = {0: "Main Entrance", 1: "Parking Lot"}
        self.current_camera_location = "Main Entrance"
        # External camera source: integer index ("1","2"…) or a URL
        # (RTSP, HTTP-MJPEG, etc.)
        self.external_cam_source = "1"

        # Twilio / WhatsApp
        self.twilio_enabled = False
        self.account_sid = ""
        self.auth_token = ""
        self.twilio_phone = ""
        self.recipient_phone = ""

        # Analytics
        self.detection_log = []
        self.total_accidents = 0
        self.session_start = datetime.now()

        # Media upload state
        self.media_processing = False
        self.uploaded_media_path = None

        # Class list
        try:
            with open("coco1.txt", "r") as f:
                self.class_list = f.read().strip().split("\n")
        except FileNotFoundError:
            messagebox.showerror("Error", "Class list file 'coco1.txt' not found!")
            self.class_list = []

        os.makedirs("accident_evidence", exist_ok=True)

        # ── Build UI ──
        self._current_page = None
        self._page_frames = {}
        self._nav_buttons = {}
        self._build_ui()
        self._show_page("dashboard")

    # ══════════════════════════════════════════
    #  UI CONSTRUCTION
    # ══════════════════════════════════════════

    def _build_ui(self):
        """Master layout: sidebar + content area."""
        self.container = ctk.CTkFrame(self.root, fg_color=COLORS["bg_dark"])
        self.container.pack(fill="both", expand=True)

        self._build_sidebar()

        # Content wrapper (right side)
        self.content_area = ctk.CTkFrame(self.container, fg_color=COLORS["bg_dark"], corner_radius=0)
        self.content_area.pack(side="left", fill="both", expand=True)

        # ── Build all pages ──
        self._build_dashboard_page()
        self._build_live_detection_page()
        self._build_media_upload_page()
        self._build_settings_page()
        self._build_history_page()

    # ─────────── Sidebar ───────────
    def _build_sidebar(self):
        sidebar = ctk.CTkFrame(self.container, width=SIDEBAR_WIDTH, fg_color=COLORS["sidebar"], corner_radius=0)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        # Brand
        brand_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        brand_frame.pack(fill="x", padx=16, pady=(24, 8))

        ctk.CTkLabel(
            brand_frame, text="AcciVision",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=COLORS["accent"]
        ).pack(anchor="w")

        ctk.CTkLabel(
            brand_frame, text="Accident Detection System",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_secondary"]
        ).pack(anchor="w")

        # Divider
        ctk.CTkFrame(sidebar, height=1, fg_color=COLORS["border"]).pack(fill="x", padx=16, pady=16)

        nav_items = [
            ("dashboard",      "\U0001F4CA  Dashboard"),
            ("live",           "\U0001F4F9  Live Detection"),
            ("media",          "\U0001F4C1  Upload Media"),
            ("settings",       "\u2699\uFE0F  Settings"),
            ("history",        "\U0001F4CB  History & Logs"),
        ]

        for key, label in nav_items:
            btn = ctk.CTkButton(
                sidebar, text=label, anchor="w",
                font=ctk.CTkFont(size=13),
                fg_color="transparent",
                text_color=COLORS["text_secondary"],
                hover_color=COLORS["card_hover"],
                height=38, corner_radius=8,
                command=lambda k=key: self._show_page(k),
            )
            btn.pack(fill="x", padx=12, pady=2)
            self._nav_buttons[key] = btn

        # Bottom spacer and version
        spacer = ctk.CTkFrame(sidebar, fg_color="transparent")
        spacer.pack(fill="both", expand=True)

        ctk.CTkLabel(
            sidebar, text="v2.0.0  \u2022  YOLOv8",
            font=ctk.CTkFont(size=10),
            text_color=COLORS["text_secondary"],
        ).pack(side="bottom", pady=12)

    # ─────────── Page switching ───────────
    def _show_page(self, page_key):
        if self._current_page == page_key:
            return
        for frame in self._page_frames.values():
            frame.pack_forget()
        self._page_frames[page_key].pack(fill="both", expand=True)
        self._current_page = page_key
        for key, btn in self._nav_buttons.items():
            if key == page_key:
                btn.configure(fg_color=COLORS["accent"], text_color="#ffffff")
            else:
                btn.configure(fg_color="transparent", text_color=COLORS["text_secondary"])

    # ─────────────────────────────────
    #  Helper: Card factory
    # ─────────────────────────────────
    def _card(self, parent, **kw):
        return ctk.CTkFrame(parent, fg_color=COLORS["card"], corner_radius=12,
                            border_width=1, border_color=COLORS["border"], **kw)

    def _stat_card(self, parent, title, value_var, accent=COLORS["accent"]):
        card = self._card(parent)
        card.pack(side="left", fill="both", expand=True, padx=6, pady=6)

        ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=11),
                     text_color=COLORS["text_secondary"]).pack(anchor="w", padx=16, pady=(14, 0))
        ctk.CTkLabel(card, textvariable=value_var,
                     font=ctk.CTkFont(size=28, weight="bold"),
                     text_color=accent).pack(anchor="w", padx=16, pady=(2, 14))
        return card

    # ══════════════════════════════════════════
    #  PAGE: Dashboard
    # ══════════════════════════════════════════
    def _build_dashboard_page(self):
        page = ctk.CTkFrame(self.content_area, fg_color=COLORS["bg_dark"])
        self._page_frames["dashboard"] = page

        # Header
        header = ctk.CTkFrame(page, fg_color="transparent")
        header.pack(fill="x", padx=28, pady=(24, 4))

        ctk.CTkLabel(header, text="Dashboard",
                     font=ctk.CTkFont(size=24, weight="bold"),
                     text_color=COLORS["text_primary"]).pack(anchor="w")
        ctk.CTkLabel(header, text="Overview of detection activity",
                     font=ctk.CTkFont(size=12),
                     text_color=COLORS["text_secondary"]).pack(anchor="w")

        # Stat row
        stats_row = ctk.CTkFrame(page, fg_color="transparent")
        stats_row.pack(fill="x", padx=22, pady=10)

        self._var_total = tk.StringVar(value="0")
        self._var_session = tk.StringVar(value="00:00:00")
        self._var_confidence = tk.StringVar(value=f"{self.confidence_threshold:.0%}")
        self._var_status = tk.StringVar(value="Idle")

        self._stat_card(stats_row, "Total Accidents", self._var_total, COLORS["danger"])
        self._stat_card(stats_row, "Session Uptime", self._var_session, COLORS["accent"])
        self._stat_card(stats_row, "Confidence", self._var_confidence, COLORS["warning"])
        self._stat_card(stats_row, "System Status", self._var_status, COLORS["success"])

        # Quick actions
        actions_card = self._card(page)
        actions_card.pack(fill="x", padx=28, pady=10)

        ctk.CTkLabel(actions_card, text="Quick Actions",
                     font=ctk.CTkFont(size=14, weight="bold"),
                     text_color=COLORS["text_primary"]).pack(anchor="w", padx=20, pady=(16, 8))

        btns = ctk.CTkFrame(actions_card, fg_color="transparent")
        btns.pack(fill="x", padx=20, pady=(0, 16))

        ctk.CTkButton(btns, text="\u25B6  Start Live Detection", fg_color=COLORS["success"],
                      hover_color=COLORS["success_hover"], corner_radius=8, height=38,
                      command=lambda: (self._show_page("live"), self._start_live_detection())
                      ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(btns, text="\U0001F4C1  Upload Media", fg_color=COLORS["accent"],
                      hover_color=COLORS["accent_hover"], corner_radius=8, height=38,
                      command=lambda: self._show_page("media")).pack(side="left", padx=(0, 8))
        ctk.CTkButton(btns, text="\U0001F4E5  Export Logs", fg_color=COLORS["card_hover"],
                      hover_color=COLORS["border"], corner_radius=8, height=38,
                      command=self.export_logs).pack(side="left", padx=(0, 8))

        # Recent activity
        recent_card = self._card(page)
        recent_card.pack(fill="both", expand=True, padx=28, pady=(4, 24))

        ctk.CTkLabel(recent_card, text="Recent Detections",
                     font=ctk.CTkFont(size=14, weight="bold"),
                     text_color=COLORS["text_primary"]).pack(anchor="w", padx=20, pady=(16, 4))

        self.recent_list_frame = ctk.CTkScrollableFrame(recent_card, fg_color="transparent")
        self.recent_list_frame.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        self._recent_placeholder = ctk.CTkLabel(
            self.recent_list_frame,
            text="No detections yet. Start a live session or upload media.",
            text_color=COLORS["text_secondary"], font=ctk.CTkFont(size=12))
        self._recent_placeholder.pack(pady=30)

        # Tick timer
        self._tick_dashboard()

    def _tick_dashboard(self):
        elapsed = datetime.now() - self.session_start
        h, rem = divmod(int(elapsed.total_seconds()), 3600)
        m, s = divmod(rem, 60)
        self._var_session.set(f"{h:02d}:{m:02d}:{s:02d}")
        self._var_total.set(str(self.total_accidents))
        self._var_confidence.set(f"{self.confidence_threshold:.0%}")
        self._var_status.set("Detecting" if self.processing or self.media_processing else "Idle")
        self.root.after(1000, self._tick_dashboard)

    def _refresh_recent_list(self):
        for w in self.recent_list_frame.winfo_children():
            w.destroy()

        if not self.detection_log:
            ctk.CTkLabel(self.recent_list_frame,
                         text="No detections yet. Start a live session or upload media.",
                         text_color=COLORS["text_secondary"],
                         font=ctk.CTkFont(size=12)).pack(pady=30)
            return

        for i, entry in enumerate(reversed(self.detection_log[-20:])):
            row = ctk.CTkFrame(self.recent_list_frame, fg_color=COLORS["card"], corner_radius=8, height=40)
            row.pack(fill="x", pady=2)

            ctk.CTkLabel(row, text=f"#{len(self.detection_log) - i}",
                         font=ctk.CTkFont(size=11, weight="bold"),
                         text_color=COLORS["danger"], width=40).pack(side="left", padx=(12, 6), pady=8)
            ctk.CTkLabel(row, text=entry["timestamp"],
                         font=ctk.CTkFont(size=11),
                         text_color=COLORS["text_secondary"]).pack(side="left", padx=4, pady=8)
            ctk.CTkLabel(row, text=f"\U0001F4CD {entry['location']}",
                         font=ctk.CTkFont(size=11),
                         text_color=COLORS["text_primary"]).pack(side="left", padx=4, pady=8)
            ctk.CTkLabel(row, text=f"Conf: {entry['confidence']:.0%}",
                         font=ctk.CTkFont(size=11),
                         text_color=COLORS["warning"]).pack(side="right", padx=12, pady=8)

    # ══════════════════════════════════════════
    #  PAGE: Live Detection
    # ══════════════════════════════════════════
    def _build_live_detection_page(self):
        page = ctk.CTkFrame(self.content_area, fg_color=COLORS["bg_dark"])
        self._page_frames["live"] = page

        # Header row
        top = ctk.CTkFrame(page, fg_color="transparent")
        top.pack(fill="x", padx=28, pady=(24, 8))

        ctk.CTkLabel(top, text="Live Camera Detection",
                     font=ctk.CTkFont(size=24, weight="bold"),
                     text_color=COLORS["text_primary"]).pack(side="left")

        self.live_status_label = ctk.CTkLabel(top, text="\u25CF Offline",
                                              font=ctk.CTkFont(size=12, weight="bold"),
                                              text_color=COLORS["text_secondary"])
        self.live_status_label.pack(side="right", padx=12)

        # Controls bar
        ctrl = self._card(page)
        ctrl.pack(fill="x", padx=28, pady=4)

        inner = ctk.CTkFrame(ctrl, fg_color="transparent")
        inner.pack(fill="x", padx=16, pady=12)

        ctk.CTkLabel(inner, text="Camera:", font=ctk.CTkFont(size=12),
                     text_color=COLORS["text_secondary"]).pack(side="left", padx=(0, 6))

        self.camera_var = tk.IntVar(value=0)
        self.camera_combo = ctk.CTkOptionMenu(
            inner, values=["0 \u2014 Webcam", "1 \u2014 External Camera"],
            command=self._on_camera_combo_change,
            width=180, height=32,
            fg_color=COLORS["input_bg"],
            button_color=COLORS["accent"],
            dropdown_fg_color=COLORS["card"],
        )
        self.camera_combo.pack(side="left", padx=(0, 16))

        ctk.CTkLabel(inner, text="Confidence:", font=ctk.CTkFont(size=12),
                     text_color=COLORS["text_secondary"]).pack(side="left", padx=(0, 6))

        self.live_conf_slider = ctk.CTkSlider(inner, from_=0.1, to=1.0, number_of_steps=18,
                                              command=self._update_confidence, width=160,
                                              progress_color=COLORS["accent"],
                                              button_color=COLORS["accent"])
        self.live_conf_slider.set(self.confidence_threshold)
        self.live_conf_slider.pack(side="left", padx=(0, 4))

        self.live_conf_val = ctk.CTkLabel(inner, text=f"{self.confidence_threshold:.0%}",
                                          font=ctk.CTkFont(size=12, weight="bold"),
                                          text_color=COLORS["accent"])
        self.live_conf_val.pack(side="left", padx=(0, 20))

        self.live_toggle_btn = ctk.CTkButton(
            inner, text="\u25B6  Start", width=120, height=32,
            fg_color=COLORS["success"], hover_color=COLORS["success_hover"],
            corner_radius=8, command=self.toggle_processing)
        self.live_toggle_btn.pack(side="right")

        # Video display
        vid_card = self._card(page)
        vid_card.pack(fill="both", expand=True, padx=28, pady=(8, 24))

        self.video_label = ctk.CTkLabel(vid_card,
                                        text="No Camera Feed\n\nClick \u25B6 Start to begin",
                                        font=ctk.CTkFont(size=14),
                                        text_color=COLORS["text_secondary"])
        self.video_label.pack(expand=True, fill="both", padx=4, pady=4)

    # ══════════════════════════════════════════
    #  PAGE: Media Upload
    # ══════════════════════════════════════════
    def _build_media_upload_page(self):
        page = ctk.CTkFrame(self.content_area, fg_color=COLORS["bg_dark"])
        self._page_frames["media"] = page

        # Header
        top = ctk.CTkFrame(page, fg_color="transparent")
        top.pack(fill="x", padx=28, pady=(24, 8))

        ctk.CTkLabel(top, text="Upload Media for Detection",
                     font=ctk.CTkFont(size=24, weight="bold"),
                     text_color=COLORS["text_primary"]).pack(side="left")

        self.media_status_label = ctk.CTkLabel(top, text="",
                                               font=ctk.CTkFont(size=12),
                                               text_color=COLORS["text_secondary"])
        self.media_status_label.pack(side="right", padx=12)

        # Upload zone
        upload_card = self._card(page)
        upload_card.pack(fill="x", padx=28, pady=6)

        drop_zone = ctk.CTkFrame(upload_card, fg_color=COLORS["input_bg"], corner_radius=10,
                                 border_width=2, border_color=COLORS["border"], height=110)
        drop_zone.pack(fill="x", padx=20, pady=20)
        drop_zone.pack_propagate(False)

        dz_inner = ctk.CTkFrame(drop_zone, fg_color="transparent")
        dz_inner.place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(dz_inner, text="\U0001F4C2",
                     font=ctk.CTkFont(size=28)).pack()
        ctk.CTkLabel(dz_inner, text="Click to browse image / video files",
                     font=ctk.CTkFont(size=13),
                     text_color=COLORS["text_secondary"]).pack(pady=(2, 0))

        supported = "Supported: JPG, PNG, BMP, TIFF, MP4, AVI, MOV, MKV, WMV"
        ctk.CTkLabel(dz_inner, text=supported,
                     font=ctk.CTkFont(size=10),
                     text_color=COLORS["text_secondary"]).pack()

        # Make the drop zone clickable
        for widget in (drop_zone, dz_inner, *dz_inner.winfo_children()):
            widget.bind("<Button-1>", lambda e: self._browse_media())

        # File info & controls
        info_row = ctk.CTkFrame(upload_card, fg_color="transparent")
        info_row.pack(fill="x", padx=20, pady=(0, 6))

        self.media_file_label = ctk.CTkLabel(info_row, text="No file selected",
                                             font=ctk.CTkFont(size=12),
                                             text_color=COLORS["text_secondary"])
        self.media_file_label.pack(side="left")

        ctrl_row = ctk.CTkFrame(upload_card, fg_color="transparent")
        ctrl_row.pack(fill="x", padx=20, pady=(0, 16))

        ctk.CTkButton(ctrl_row, text="\U0001F4C2  Browse Files", width=140, height=34,
                      fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
                      corner_radius=8, command=self._browse_media).pack(side="left", padx=(0, 8))

        self.media_analyze_btn = ctk.CTkButton(
            ctrl_row, text="\U0001F50D  Analyze", width=140, height=34,
            fg_color=COLORS["success"], hover_color=COLORS["success_hover"],
            corner_radius=8, state="disabled", command=self._analyze_media)
        self.media_analyze_btn.pack(side="left", padx=(0, 8))

        self.media_stop_btn = ctk.CTkButton(
            ctrl_row, text="\u23F9  Stop", width=100, height=34,
            fg_color=COLORS["danger"], hover_color=COLORS["danger_hover"],
            corner_radius=8, state="disabled", command=self._stop_media_analysis)
        self.media_stop_btn.pack(side="left")

        # Confidence slider for media
        conf_row = ctk.CTkFrame(upload_card, fg_color="transparent")
        conf_row.pack(fill="x", padx=20, pady=(0, 16))

        ctk.CTkLabel(conf_row, text="Confidence:", font=ctk.CTkFont(size=12),
                     text_color=COLORS["text_secondary"]).pack(side="left", padx=(0, 8))

        self.media_conf_slider = ctk.CTkSlider(conf_row, from_=0.1, to=1.0, number_of_steps=18,
                                               command=self._update_confidence, width=200,
                                               progress_color=COLORS["accent"],
                                               button_color=COLORS["accent"])
        self.media_conf_slider.set(self.confidence_threshold)
        self.media_conf_slider.pack(side="left", padx=(0, 6))

        self.media_conf_val = ctk.CTkLabel(conf_row, text=f"{self.confidence_threshold:.0%}",
                                           font=ctk.CTkFont(size=12, weight="bold"),
                                           text_color=COLORS["accent"])
        self.media_conf_val.pack(side="left")

        # Preview / Result display
        preview_card = self._card(page)
        preview_card.pack(fill="both", expand=True, padx=28, pady=(6, 24))

        self.media_preview_label = ctk.CTkLabel(
            preview_card, text="Upload an image or video to preview detection results",
            font=ctk.CTkFont(size=13), text_color=COLORS["text_secondary"])
        self.media_preview_label.pack(expand=True, fill="both", padx=4, pady=4)

    # ══════════════════════════════════════════
    #  PAGE: Settings
    # ══════════════════════════════════════════
    def _build_settings_page(self):
        page = ctk.CTkScrollableFrame(self.content_area, fg_color=COLORS["bg_dark"])
        self._page_frames["settings"] = page

        # Header
        ctk.CTkLabel(page, text="Settings",
                     font=ctk.CTkFont(size=24, weight="bold"),
                     text_color=COLORS["text_primary"]).pack(anchor="w", padx=28, pady=(24, 4))
        ctk.CTkLabel(page, text="Configure detection, cameras, and notifications",
                     font=ctk.CTkFont(size=12),
                     text_color=COLORS["text_secondary"]).pack(anchor="w", padx=28, pady=(0, 12))

        # ── Detection Settings Card ──
        det = self._card(page)
        det.pack(fill="x", padx=28, pady=6)

        ctk.CTkLabel(det, text="Detection", font=ctk.CTkFont(size=15, weight="bold"),
                     text_color=COLORS["text_primary"]).pack(anchor="w", padx=20, pady=(16, 8))

        row1 = ctk.CTkFrame(det, fg_color="transparent")
        row1.pack(fill="x", padx=20, pady=4)
        ctk.CTkLabel(row1, text="Confidence Threshold", font=ctk.CTkFont(size=12),
                     text_color=COLORS["text_secondary"]).pack(side="left")
        self.settings_conf_val = ctk.CTkLabel(row1, text=f"{self.confidence_threshold:.0%}",
                                              font=ctk.CTkFont(size=12, weight="bold"),
                                              text_color=COLORS["accent"])
        self.settings_conf_val.pack(side="right", padx=8)
        self.settings_conf_slider = ctk.CTkSlider(det, from_=0.1, to=1.0, number_of_steps=18,
                                                  command=self._update_confidence, width=400,
                                                  progress_color=COLORS["accent"],
                                                  button_color=COLORS["accent"])
        self.settings_conf_slider.set(self.confidence_threshold)
        self.settings_conf_slider.pack(anchor="w", padx=20, pady=(0, 6))

        row2 = ctk.CTkFrame(det, fg_color="transparent")
        row2.pack(fill="x", padx=20, pady=4)
        ctk.CTkLabel(row2, text="Consecutive frames to confirm accident",
                     font=ctk.CTkFont(size=12),
                     text_color=COLORS["text_secondary"]).pack(side="left")
        self.frames_thresh_entry = ctk.CTkEntry(row2, width=60, fg_color=COLORS["input_bg"],
                                                border_color=COLORS["border"])
        self.frames_thresh_entry.pack(side="right", padx=8)
        self.frames_thresh_entry.insert(0, str(self.accident_frames_threshold))

        self.save_evidence_var = tk.BooleanVar(value=self.save_evidence)
        ctk.CTkCheckBox(det, text="Save accident evidence images",
                        variable=self.save_evidence_var,
                        command=self._toggle_save_evidence,
                        font=ctk.CTkFont(size=12),
                        border_color=COLORS["border"],
                        fg_color=COLORS["accent"]).pack(anchor="w", padx=20, pady=(6, 16))

        # ── Camera Locations Card ──
        cam = self._card(page)
        cam.pack(fill="x", padx=28, pady=6)

        ctk.CTkLabel(cam, text="Camera Locations", font=ctk.CTkFont(size=15, weight="bold"),
                     text_color=COLORS["text_primary"]).pack(anchor="w", padx=20, pady=(16, 8))

        for cam_id, default_name in self.camera_locations.items():
            row = ctk.CTkFrame(cam, fg_color="transparent")
            row.pack(fill="x", padx=20, pady=3)
            label_text = "Webcam" if cam_id == 0 else "External Camera"
            ctk.CTkLabel(row, text=f"{label_text} Location:", font=ctk.CTkFont(size=12),
                         text_color=COLORS["text_secondary"]).pack(side="left")
            entry = ctk.CTkEntry(row, width=220, fg_color=COLORS["input_bg"], border_color=COLORS["border"])
            entry.pack(side="right", padx=8)
            entry.insert(0, default_name)
            if cam_id == 0:
                self.cam0_entry = entry
            else:
                self.cam1_entry = entry

        # ── External camera source row ──
        src_row = ctk.CTkFrame(cam, fg_color="transparent")
        src_row.pack(fill="x", padx=20, pady=3)
        ctk.CTkLabel(src_row, text="External Camera Source:", font=ctk.CTkFont(size=12),
                     text_color=COLORS["text_secondary"]).pack(side="left")
        self.ext_cam_src_entry = ctk.CTkEntry(
            src_row, width=300,
            placeholder_text="Index (1) or URL (rtsp://… / http://…)",
            fg_color=COLORS["input_bg"], border_color=COLORS["border"])
        self.ext_cam_src_entry.pack(side="right", padx=8)
        self.ext_cam_src_entry.insert(0, self.external_cam_source)

        ctk.CTkButton(cam, text="Save Camera Settings", width=180, height=34,
                      fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
                      corner_radius=8, command=self._save_location_settings).pack(anchor="w", padx=20, pady=(10, 16))

        # ── WhatsApp / Twilio Card ──
        wa = self._card(page)
        wa.pack(fill="x", padx=28, pady=6)

        ctk.CTkLabel(wa, text="WhatsApp Notifications", font=ctk.CTkFont(size=15, weight="bold"),
                     text_color=COLORS["text_primary"]).pack(anchor="w", padx=20, pady=(16, 8))

        self.twilio_enabled_var = tk.BooleanVar(value=self.twilio_enabled)
        ctk.CTkCheckBox(wa, text="Enable WhatsApp alerts via Twilio",
                        variable=self.twilio_enabled_var,
                        command=self._toggle_twilio,
                        font=ctk.CTkFont(size=12),
                        border_color=COLORS["border"],
                        fg_color=COLORS["accent"]).pack(anchor="w", padx=20, pady=(0, 8))

        fields = [
            ("Account SID",    "Enter your Twilio Account SID", False),
            ("Auth Token",     "Enter your Twilio Auth Token",  True),
            ("Twilio Phone",   "+1XXXXXXXXXX",                  False),
            ("Recipient Phone","+1XXXXXXXXXX",                  False),
        ]
        self._twilio_entries = {}
        for label_text, placeholder, secret in fields:
            row = ctk.CTkFrame(wa, fg_color="transparent")
            row.pack(fill="x", padx=20, pady=3)
            ctk.CTkLabel(row, text=f"{label_text}:", font=ctk.CTkFont(size=12),
                         text_color=COLORS["text_secondary"], width=130, anchor="w").pack(side="left")
            entry = ctk.CTkEntry(row, placeholder_text=placeholder, width=320,
                                 fg_color=COLORS["input_bg"], border_color=COLORS["border"],
                                 show="\u2022" if secret else "")
            entry.pack(side="left", padx=8)
            self._twilio_entries[label_text] = entry

        btn_row = ctk.CTkFrame(wa, fg_color="transparent")
        btn_row.pack(fill="x", padx=20, pady=(10, 16))
        ctk.CTkButton(btn_row, text="Save Settings", width=140, height=34,
                      fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
                      corner_radius=8, command=self._save_twilio_settings).pack(side="left", padx=(0, 8))
        ctk.CTkButton(btn_row, text="\U0001F514 Test WhatsApp", width=160, height=34,
                      fg_color=COLORS["success"], hover_color=COLORS["success_hover"],
                      corner_radius=8, command=self._test_twilio_whatsapp).pack(side="left")

        # ── Save All ──
        ctk.CTkButton(page, text="\U0001F4BE  Save All Settings", width=200, height=40,
                      fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
                      corner_radius=8, font=ctk.CTkFont(size=13, weight="bold"),
                      command=self._save_all_settings).pack(anchor="w", padx=28, pady=(12, 28))

    # ══════════════════════════════════════════
    #  PAGE: History & Logs
    # ══════════════════════════════════════════
    def _build_history_page(self):
        page = ctk.CTkFrame(self.content_area, fg_color=COLORS["bg_dark"])
        self._page_frames["history"] = page

        top = ctk.CTkFrame(page, fg_color="transparent")
        top.pack(fill="x", padx=28, pady=(24, 8))

        ctk.CTkLabel(top, text="Detection History",
                     font=ctk.CTkFont(size=24, weight="bold"),
                     text_color=COLORS["text_primary"]).pack(side="left")

        ctk.CTkButton(top, text="\U0001F4E5  Export CSV", width=130, height=34,
                      fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
                      corner_radius=8, command=self.export_logs).pack(side="right")
        ctk.CTkButton(top, text="\U0001F504  Refresh", width=100, height=34,
                      fg_color=COLORS["card_hover"], hover_color=COLORS["border"],
                      corner_radius=8, command=self._populate_history).pack(side="right", padx=6)

        # Table header
        table_header = self._card(page)
        table_header.pack(fill="x", padx=28, pady=(6, 0))

        hdr = ctk.CTkFrame(table_header, fg_color="transparent")
        hdr.pack(fill="x", padx=16, pady=10)
        for col, w in [("#", 40), ("Timestamp", 170), ("Location", 160),
                       ("Confidence", 100), ("Evidence", 200)]:
            ctk.CTkLabel(hdr, text=col, font=ctk.CTkFont(size=11, weight="bold"),
                         text_color=COLORS["text_secondary"], width=w, anchor="w").pack(side="left", padx=4)

        # Scrollable list
        self.history_scroll = ctk.CTkScrollableFrame(page, fg_color=COLORS["bg_dark"])
        self.history_scroll.pack(fill="both", expand=True, padx=28, pady=(0, 24))

        self._history_placeholder = ctk.CTkLabel(
            self.history_scroll, text="No detections recorded yet.",
            text_color=COLORS["text_secondary"], font=ctk.CTkFont(size=12))
        self._history_placeholder.pack(pady=40)

    def _populate_history(self):
        for w in self.history_scroll.winfo_children():
            w.destroy()

        if not self.detection_log:
            ctk.CTkLabel(self.history_scroll, text="No detections recorded yet.",
                         text_color=COLORS["text_secondary"],
                         font=ctk.CTkFont(size=12)).pack(pady=40)
            return

        for i, entry in enumerate(reversed(self.detection_log)):
            row_frame = ctk.CTkFrame(self.history_scroll, fg_color=COLORS["card"], corner_radius=8)
            row_frame.pack(fill="x", pady=2)

            inner = ctk.CTkFrame(row_frame, fg_color="transparent")
            inner.pack(fill="x", padx=16, pady=8)

            idx = len(self.detection_log) - i
            ctk.CTkLabel(inner, text=str(idx), width=40, anchor="w",
                         font=ctk.CTkFont(size=11, weight="bold"),
                         text_color=COLORS["danger"]).pack(side="left", padx=4)
            ctk.CTkLabel(inner, text=entry["timestamp"], width=170, anchor="w",
                         font=ctk.CTkFont(size=11),
                         text_color=COLORS["text_primary"]).pack(side="left", padx=4)
            ctk.CTkLabel(inner, text=entry["location"], width=160, anchor="w",
                         font=ctk.CTkFont(size=11),
                         text_color=COLORS["text_primary"]).pack(side="left", padx=4)
            ctk.CTkLabel(inner, text=f"{entry['confidence']:.0%}", width=100, anchor="w",
                         font=ctk.CTkFont(size=11, weight="bold"),
                         text_color=COLORS["warning"]).pack(side="left", padx=4)

            evidence_text = os.path.basename(entry.get("evidence", "N/A"))
            ctk.CTkLabel(inner, text=evidence_text, width=140, anchor="w",
                         font=ctk.CTkFont(size=11),
                         text_color=COLORS["text_secondary"]).pack(side="left", padx=4)

            if entry.get("evidence") and os.path.exists(entry["evidence"]):
                ctk.CTkButton(inner, text="View", width=60, height=26,
                              fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
                              corner_radius=6, font=ctk.CTkFont(size=11),
                              command=lambda p=entry["evidence"]: self._view_evidence(p)).pack(side="right")

    # ══════════════════════════════════════════
    #  CORE LOGIC — Settings callbacks
    # ══════════════════════════════════════════
    def _update_confidence(self, value):
        self.confidence_threshold = float(value)
        text = f"{self.confidence_threshold:.0%}"
        for lbl in (self.live_conf_val, self.media_conf_val, self.settings_conf_val):
            try:
                lbl.configure(text=text)
            except Exception:
                pass
        for sl in (self.live_conf_slider, self.media_conf_slider, self.settings_conf_slider):
            try:
                sl.set(self.confidence_threshold)
            except Exception:
                pass

    def _on_camera_combo_change(self, choice):
        cam_id = int(choice.split(" ")[0])
        self.camera_var.set(cam_id)
        self.current_camera_location = self.camera_locations.get(cam_id, f"Camera {cam_id}")

    def _toggle_save_evidence(self):
        self.save_evidence = self.save_evidence_var.get()

    def _toggle_twilio(self):
        self.twilio_enabled = self.twilio_enabled_var.get()

    def _save_location_settings(self):
        self.camera_locations[0] = self.cam0_entry.get().strip()
        self.camera_locations[1] = self.cam1_entry.get().strip()
        src = self.ext_cam_src_entry.get().strip()
        self.external_cam_source = src if src else "1"
        cam_id = self.camera_var.get()
        self.current_camera_location = self.camera_locations.get(cam_id, f"Camera {cam_id}")
        messagebox.showinfo("Saved", "Camera settings saved.")

    def _save_twilio_settings(self):
        self.account_sid = self._twilio_entries["Account SID"].get().strip()
        self.auth_token = self._twilio_entries["Auth Token"].get().strip()
        self.twilio_phone = self._twilio_entries["Twilio Phone"].get().strip()
        self.recipient_phone = self._twilio_entries["Recipient Phone"].get().strip()

        if self.twilio_enabled:
            if not all([self.account_sid, self.auth_token, self.twilio_phone, self.recipient_phone]):
                messagebox.showerror("Error", "Please fill in all WhatsApp fields!")
                self.twilio_enabled_var.set(False)
                self.twilio_enabled = False
                return
            if not (self.twilio_phone.startswith("+") and self.recipient_phone.startswith("+")):
                messagebox.showwarning("Warning",
                                       "Phone numbers should be in E.164 format, e.g. +1XXXXXXXXXX")

        messagebox.showinfo("Saved", "Notification settings saved.")

    def _test_twilio_whatsapp(self):
        if not self.twilio_enabled:
            messagebox.showinfo("Info", "WhatsApp notifications are disabled. Enable them first.")
            return
        if not all([self.account_sid, self.auth_token]):
            messagebox.showerror("Error", "Fill in all WhatsApp settings first!")
            return
        try:
            ok = self._send_whatsapp_message("Test", "This is a test message from AcciVision.")
            if ok:
                messagebox.showinfo("Success", "Test WhatsApp message sent!")
            else:
                messagebox.showerror("Error", "Failed to send. Check your settings.")
        except Exception as e:
            messagebox.showerror("Error", f"WhatsApp error: {e}")

    def _save_all_settings(self):
        try:
            self.accident_frames_threshold = int(self.frames_thresh_entry.get())
        except ValueError:
            pass
        self._save_location_settings()
        self._save_twilio_settings()

    # ══════════════════════════════════════════
    #  CORE LOGIC — Live detection
    # ══════════════════════════════════════════
    def _start_live_detection(self):
        if not self.processing:
            self.toggle_processing()

    def toggle_processing(self):
        if not self.processing:
            # Increment session counter BEFORE starting so the new thread owns this id.
            self._session_id += 1
            self.processing = True
            self.live_toggle_btn.configure(text="\u23F9  Stop", fg_color=COLORS["danger"],
                                           hover_color=COLORS["danger_hover"])
            self.live_status_label.configure(text="\u23F3 Starting\u2026", text_color=COLORS["warning"])

            prev_thread = self._feed_thread

            def _deferred_start():
                # Wait for the old thread to finish — it exits fast because Stop
                # already released its cap, making cap.read() return immediately.
                if prev_thread is not None and prev_thread.is_alive():
                    prev_thread.join(timeout=5.0)
                # Bail if the user pressed Stop while we were waiting.
                if not self.processing:
                    return
                t = threading.Thread(target=self._process_camera_feed, daemon=True)
                self._feed_thread = t
                t.start()

            threading.Thread(target=_deferred_start, daemon=True).start()
        else:
            self.processing = False
            # Release cap immediately so the worker's blocking cap.read() returns
            # right away, letting the thread exit in milliseconds.
            cap = self.cap
            if cap is not None:
                self.cap = None
                try:
                    cap.release()
                except Exception:
                    pass
            self.live_toggle_btn.configure(text="\u25B6  Start", fg_color=COLORS["success"],
                                           hover_color=COLORS["success_hover"])
            self.live_status_label.configure(text="\u25CF Offline", text_color=COLORS["text_secondary"])

    def _process_camera_feed(self):
        camera_id = self.camera_var.get()
        self.current_camera_location = self.camera_locations.get(camera_id, f"Camera {camera_id}")

        # Resolve the actual OpenCV source.
        # Webcam always uses device index 0.
        # External camera uses self.external_cam_source which can be:
        #   - a numeric string ("1", "2") → converted to int device index
        #   - a URL string (rtsp://…, http://…) → passed as-is
        if camera_id == 0:
            capture_source = 0
        else:
            src = self.external_cam_source.strip()
            if src.lstrip("-").isdigit():
                capture_source = int(src)
            else:
                capture_source = src if src else 1

        # Snapshot session id — all state-mutating code below is guarded by this
        # so a stale thread from a previous session can never clobber a new one.
        my_session = self._session_id

        cap = cv2.VideoCapture(capture_source)

        # Only register this cap if our session is still the active one.
        if my_session == self._session_id:
            self.cap = cap

        if cap.isOpened() and my_session == self._session_id:
            self.root.after(0, lambda: self.live_status_label.configure(
                text="\u25CF Live", text_color=COLORS["success"]))

        if not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            if my_session != self._session_id:
                return  # newer session already running; silently exit
            src_display = capture_source if isinstance(capture_source, int) else f'"{capture_source}"'
            self.root.after(0, lambda: messagebox.showerror(
                "Error", f"Failed to open camera source {src_display}!\n\n"
                         "Check Settings \u2192 External Camera Source."))
            self.processing = False
            self.root.after(0, lambda: self.live_toggle_btn.configure(
                text="\u25B6  Start", fg_color=COLORS["success"], hover_color=COLORS["success_hover"]))
            self.root.after(0, lambda: self.live_status_label.configure(
                text="\u25CF Offline", text_color=COLORS["text_secondary"]))
            return

        fw, fh = 800, 450

        while self.processing:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (fw, fh))
            results = self.model.predict(
                frame, conf=self.confidence_threshold,
                iou=0.45, agnostic_nms=True, verbose=False
            )
            frame_has_accident = False
            best_acc_conf = 0.0

            if results and len(results[0].boxes) > 0:
                px = pd.DataFrame(results[0].boxes.data.cpu().numpy()).astype("float")
                for _, row in px.iterrows():
                    x1, y1, x2, y2 = map(int, row[:4])
                    confidence = float(row[4])
                    class_id = int(row[5])
                    class_name = self.class_list[class_id] if class_id < len(self.class_list) else "Unknown"

                    if "accident" in class_name.lower():
                        frame_has_accident = True
                        best_acc_conf = max(best_acc_conf, confidence)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f"{class_name}: {confidence:.2f}",
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{class_name}: {confidence:.2f}",
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Rolling majority-vote over the last N frames — suppresses single-frame
            # hallucinations caused by motion blur, reflections or partial occlusion.
            self._vote_window.append(1 if frame_has_accident else 0)
            votes = sum(self._vote_window)
            if frame_has_accident:
                self.accident_frames_count += 1
            else:
                self.accident_frames_count = 0

            # Confirm only when enough recent frames agree
            if votes >= self.accident_frames_threshold and frame_has_accident:
                # Time-bucket dedup: at most one alert per 10-second window
                accident_id = str(int(time.time()) // 10)
                if accident_id not in self.detected_accidents:
                    self.detected_accidents.add(accident_id)
                    self.total_accidents += 1
                    clip_frames = list(self.live_frame_buffer)
                    threading.Thread(
                        target=self._save_accident_clip,
                        args=(clip_frames, best_acc_conf),
                        daemon=True
                    ).start()
                    self._send_notification("\u26A0\uFE0F ACCIDENT DETECTED!",
                                            f"Accident detected ({best_acc_conf:.0%} confidence)")
                    self.root.after(0, self._refresh_recent_list)

            # HUD overlays
            cv2.putText(frame, f"Location: {self.current_camera_location}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, time.strftime("%Y-%m-%d %H:%M:%S"), (fw - 220, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Add annotated frame to rolling pre-buffer
            self.live_frame_buffer.append(frame.copy())

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_rgb = Image.fromarray(rgb)
            img = ctk.CTkImage(light_image=pil_rgb, size=(fw, fh))
            # Schedule UI update on the main thread (thread-safe).
            self.root.after(0, lambda i=img: (
                self.video_label.configure(image=i, text=""),
                setattr(self.video_label, "image", i),
            ))

        # Release the local cap (no-op if Stop already released it).
        try:
            cap.release()
        except Exception:
            pass
        if self.cap is cap:
            self.cap = None

        # ── CRITICAL: only touch shared UI/state if we are still the active session.
        # If my_session != self._session_id a new Start has already taken over.
        if my_session != self._session_id:
            return

        if not self.processing:
            # Normal Stop — button is already "Start", just fix the status label.
            self.root.after(0, lambda: self.live_status_label.configure(
                text="\u25CF Offline", text_color=COLORS["text_secondary"]))
        else:
            # Camera dropped unexpectedly while still marked as running.
            self.processing = False
            self.root.after(0, lambda: (
                self.live_toggle_btn.configure(
                    text="\u25B6  Start", fg_color=COLORS["success"],
                    hover_color=COLORS["success_hover"]),
                self.live_status_label.configure(
                    text="\u25CF Offline", text_color=COLORS["text_secondary"]),
                messagebox.showwarning("Camera Lost", "Camera feed ended unexpectedly."),
            ))

        # Blank the video display only when no new session has registered a cap.
        if self.cap is None:
            self.root.after(0, lambda: (
                self.video_label.configure(
                    image=None, text="No Camera Feed\n\nClick \u25B6 Start to begin"),
                setattr(self.video_label, "image", None),
            ))

    # ══════════════════════════════════════════
    #  CORE LOGIC — Media upload & analysis
    # ══════════════════════════════════════════
    def _browse_media(self):
        filetypes = [
            ("All supported",
             "*.jpg *.jpeg *.png *.bmp *.tiff *.webp *.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
            ("Images", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
            ("Videos", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
        ]
        path = filedialog.askopenfilename(title="Select Image or Video", filetypes=filetypes)
        if path:
            self.uploaded_media_path = path
            fname = os.path.basename(path)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            ext = os.path.splitext(path)[1].lower()
            kind = "Image" if ext in MEDIA_EXTENSIONS_IMAGE else "Video"
            self.media_file_label.configure(
                text=f"{kind}: {fname}  ({size_mb:.1f} MB)",
                text_color=COLORS["text_primary"])
            self.media_analyze_btn.configure(state="normal")
            self.media_status_label.configure(text="Ready to analyze", text_color=COLORS["success"])

            # Show quick preview for images
            if ext in MEDIA_EXTENSIONS_IMAGE:
                self._show_image_preview(path)

    def _show_image_preview(self, path):
        try:
            img = cv2.imread(path)
            if img is None:
                return
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            max_w, max_h = 780, 440
            scale = min(max_w / w, max_h / h, 1.0)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h))
            photo = ctk.CTkImage(light_image=Image.fromarray(img), size=(new_w, new_h))
            self.media_preview_label.configure(image=photo, text="")
            self.media_preview_label.image = photo
        except Exception:
            pass

    def _analyze_media(self):
        if not self.uploaded_media_path:
            return
        ext = os.path.splitext(self.uploaded_media_path)[1].lower()
        if ext in MEDIA_EXTENSIONS_IMAGE:
            threading.Thread(target=self._analyze_image, daemon=True).start()
        elif ext in MEDIA_EXTENSIONS_VIDEO:
            threading.Thread(target=self._analyze_video, daemon=True).start()
        else:
            messagebox.showerror("Error", "Unsupported file format.")

    def _analyze_image(self):
        self.media_processing = True
        self.media_analyze_btn.configure(state="disabled")
        self.media_status_label.configure(text="Analyzing image\u2026", text_color=COLORS["warning"])

        try:
            img = cv2.imread(self.uploaded_media_path)
            if img is None:
                self.root.after(0, lambda: messagebox.showerror("Error", "Could not read image."))
                return

            results = self.model.predict(
                img, conf=self.confidence_threshold,
                iou=0.45, agnostic_nms=True, verbose=False
            )
            accident_found = False
            best_img_conf = 0.0

            if results and len(results[0].boxes) > 0:
                px = pd.DataFrame(results[0].boxes.data.cpu().numpy()).astype("float")
                for _, row in px.iterrows():
                    x1, y1, x2, y2 = map(int, row[:4])
                    confidence = float(row[4])
                    class_id = int(row[5])
                    class_name = self.class_list[class_id] if class_id < len(self.class_list) else "Unknown"

                    if "accident" in class_name.lower():
                        accident_found = True
                        best_img_conf = max(best_img_conf, confidence)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        label = f"ACCIDENT {confidence:.0%}"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 8, y1), (0, 0, 255), -1)
                        cv2.putText(img, label, (x1 + 4, y1 - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    else:
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img, f"{class_name}: {confidence:.2f}",
                                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if accident_found:
                self.total_accidents += 1
                self._save_accident_frame(img, best_img_conf)
                self._send_notification("\u26A0\uFE0F ACCIDENT DETECTED IN IMAGE",
                                        "Accident found in uploaded image")
                self.root.after(0, self._refresh_recent_list)
                status_text = "\u26A0 Accident detected!"
                status_color = COLORS["danger"]
            else:
                status_text = "\u2713 No accident detected"
                status_color = COLORS["success"]

            # Show result
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            max_w, max_h = 780, 440
            scale = min(max_w / w, max_h / h, 1.0)
            _disp_w, _disp_h = int(w * scale), int(h * scale)
            rgb = cv2.resize(rgb, (_disp_w, _disp_h))
            photo = ctk.CTkImage(light_image=Image.fromarray(rgb), size=(_disp_w, _disp_h))
            self.root.after(0, lambda: self.media_preview_label.configure(image=photo, text=""))
            self.media_preview_label.image = photo
            self.root.after(0, lambda: self.media_status_label.configure(
                text=status_text, text_color=status_color))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Analysis failed: {e}"))
        finally:
            self.media_processing = False
            self.root.after(0, lambda: self.media_analyze_btn.configure(state="normal"))

    def _analyze_video(self):
        self.media_processing = True
        self.media_analyze_btn.configure(state="disabled")
        self.media_stop_btn.configure(state="normal")
        self.media_status_label.configure(text="Processing video\u2026", text_color=COLORS["warning"])

        try:
            cap = cv2.VideoCapture(self.uploaded_media_path)
            if not cap.isOpened():
                self.root.after(0, lambda: messagebox.showerror("Error", "Could not open video."))
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            frame_count = 0
            accident_count_in_video = 0
            local_accident_frames = 0
            clip_buffer = collections.deque(maxlen=90)  # rolling pre-buffer for video clips

            while self.media_processing:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Process every 2nd frame for speed
                if frame_count % 2 != 0:
                    continue

                results = self.model.predict(
                    frame, conf=self.confidence_threshold,
                    iou=0.45, agnostic_nms=True, verbose=False
                )
                accident_in_frame = False
                best_vid_conf = 0.0

                if results and len(results[0].boxes) > 0:
                    px = pd.DataFrame(results[0].boxes.data.cpu().numpy()).astype("float")
                    for _, row in px.iterrows():
                        x1, y1, x2, y2 = map(int, row[:4])
                        confidence = float(row[4])
                        class_id = int(row[5])
                        class_name = (self.class_list[class_id]
                                      if class_id < len(self.class_list) else "Unknown")

                        if "accident" in class_name.lower():
                            accident_in_frame = True
                            best_vid_conf = max(best_vid_conf, confidence)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            label = f"ACCIDENT {confidence:.0%}"
                            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                            cv2.rectangle(frame, (x1, y1 - th - 10),
                                          (x1 + tw + 8, y1), (0, 0, 255), -1)
                            cv2.putText(frame, label, (x1 + 4, y1 - 6),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        else:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"{class_name}: {confidence:.2f}",
                                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Always add the annotated frame to the rolling clip buffer
                clip_buffer.append(frame.copy())

                if accident_in_frame:
                    local_accident_frames += 1
                    if local_accident_frames >= self.accident_frames_threshold:
                        accident_count_in_video += 1
                        self.total_accidents += 1
                        saved_clip = list(clip_buffer)
                        clip_fps = float(fps) if fps > 0 else 20.0
                        threading.Thread(
                            target=self._save_accident_clip,
                            args=(saved_clip, best_vid_conf, clip_fps),
                            daemon=True
                        ).start()
                        self._send_notification("\u26A0\uFE0F ACCIDENT IN VIDEO",
                                                f"Accident detected at frame {frame_count}")
                        self.root.after(0, self._refresh_recent_list)
                        local_accident_frames = 0
                else:
                    local_accident_frames = 0

                # Progress HUD
                progress = frame_count / total_frames if total_frames > 0 else 0
                ts = frame_count / fps
                cv2.putText(frame, f"Frame {frame_count}/{total_frames} ({progress:.0%})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Time: {ts:.1f}s", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                if accident_count_in_video > 0:
                    cv2.putText(frame, f"Accidents found: {accident_count_in_video}",
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Display
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                max_w, max_h = 780, 440
                scale = min(max_w / w, max_h / h, 1.0)
                _vdisp_w, _vdisp_h = int(w * scale), int(h * scale)
                rgb = cv2.resize(rgb, (_vdisp_w, _vdisp_h))
                photo = ctk.CTkImage(light_image=Image.fromarray(rgb), size=(_vdisp_w, _vdisp_h))
                self.media_preview_label.configure(image=photo, text="")
                self.media_preview_label.image = photo

                status = f"Processing\u2026 {progress:.0%}  |  Accidents: {accident_count_in_video}"
                self.media_status_label.configure(text=status, text_color=COLORS["warning"])
                self.root.update_idletasks()

            cap.release()

            # Final status
            if accident_count_in_video > 0:
                final = f"Done \u2014 {accident_count_in_video} accident(s) detected"
                color = COLORS["danger"]
            else:
                final = "Done \u2014 No accidents detected"
                color = COLORS["success"]
            self.root.after(0, lambda: self.media_status_label.configure(text=final, text_color=color))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Video analysis failed: {e}"))
        finally:
            self.media_processing = False
            self.root.after(0, lambda: self.media_analyze_btn.configure(state="normal"))
            self.root.after(0, lambda: self.media_stop_btn.configure(state="disabled"))

    def _stop_media_analysis(self):
        self.media_processing = False

    # ══════════════════════════════════════════
    #  CORE LOGIC — Shared helpers
    # ══════════════════════════════════════════
    def _generate_accident_id(self, frame):
        small = cv2.resize(frame, (32, 32))
        return hash(small.tobytes())

    def _save_accident_clip(self, frames, actual_conf=None, fps=20):
        """Save a list of annotated frames as an MP4 clip (live & video analysis evidence)."""
        if not self.save_evidence or not frames:
            return None
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filepath = f"accident_evidence/accident_{timestamp}.mp4"
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(filepath, fourcc, float(fps), (w, h))
        for f in frames:
            writer.write(f)
        writer.release()

        self.detection_log.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "evidence": filepath,
            "confidence": actual_conf if actual_conf is not None else self.confidence_threshold,
            "location": self.current_camera_location,
            "type": "clip",
        })
        return filepath

    def _save_accident_frame(self, frame, actual_conf=None):
        """Save a single annotated frame as JPEG (uploaded image analysis only)."""
        if not self.save_evidence:
            return None
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filepath = f"accident_evidence/accident_{timestamp}.jpg"
        cv2.imwrite(filepath, frame)

        self.detection_log.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "evidence": filepath,
            "confidence": actual_conf if actual_conf is not None else self.confidence_threshold,
            "location": self.current_camera_location,
            "type": "image",
        })
        return filepath

    def _view_evidence(self, filepath):
        if not os.path.exists(filepath):
            messagebox.showerror("Error", f"File not found: {filepath}")
            return
        if filepath.lower().endswith(".mp4"):
            self._view_video_evidence(filepath)
        else:
            self._view_image_evidence(filepath)

    def _view_image_evidence(self, filepath):
        try:
            win = ctk.CTkToplevel(self.root)
            win.title("Accident Evidence \u2014 Image")
            win.geometry("860x620")

            img = cv2.imread(filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            scale = min(840 / w, 580 / h, 1.0)
            _ew, _eh = int(w * scale), int(h * scale)
            img = cv2.resize(img, (_ew, _eh))
            photo = ctk.CTkImage(light_image=Image.fromarray(img), size=(_ew, _eh))

            lbl = ctk.CTkLabel(win, image=photo, text="")
            lbl.image = photo
            lbl.pack(expand=True, fill="both", padx=10, pady=10)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image: {e}")

    def _view_video_evidence(self, filepath):
        """Play back a saved accident clip inside the app."""
        win = ctk.CTkToplevel(self.root)
        win.title("Accident Evidence \u2014 Video Clip")
        win.geometry("880x650")
        win.resizable(False, False)

        info_bar = ctk.CTkFrame(win, fg_color=COLORS["card"], corner_radius=0)
        info_bar.pack(fill="x")
        ctk.CTkLabel(info_bar, text=f"\U0001F3AC  {os.path.basename(filepath)}",
                     font=ctk.CTkFont(size=12),
                     text_color=COLORS["text_secondary"]).pack(side="left", padx=12, pady=8)
        ctk.CTkButton(info_bar, text="\U0001F4E4  Open in Player", width=130, height=28,
                      fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
                      corner_radius=6, font=ctk.CTkFont(size=11),
                      command=lambda: self._open_with_system_player(filepath)).pack(side="right", padx=8, pady=6)

        display = ctk.CTkLabel(win, text="", fg_color=COLORS["bg_dark"])
        display.pack(fill="both", expand=True, padx=8, pady=8)

        ctrl = ctk.CTkFrame(win, fg_color=COLORS["sidebar"], corner_radius=0, height=46)
        ctrl.pack(fill="x", side="bottom")
        ctrl.pack_propagate(False)

        playing = [True]
        stop_flag = [False]

        def play_loop():
            cap = cv2.VideoCapture(filepath)
            fps_v = cap.get(cv2.CAP_PROP_FPS) or 20
            delay = max(1, int(1000 / fps_v))
            while not stop_flag[0]:
                if playing[0]:
                    ret, frame = cap.read()
                    if not ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h2, w2 = rgb.shape[:2]
                    scale = min(860 / w2, 580 / h2, 1.0)
                    _pvw, _pvh = int(w2 * scale), int(h2 * scale)
                    rgb = cv2.resize(rgb, (_pvw, _pvh))
                    photo = ctk.CTkImage(light_image=Image.fromarray(rgb), size=(_pvw, _pvh))
                    try:
                        display.configure(image=photo)
                        display.image = photo
                    except Exception:
                        break
                win.after(delay)
            cap.release()

        def on_close():
            stop_flag[0] = True
            win.destroy()

        def toggle_play():
            playing[0] = not playing[0]
            play_pause_btn.configure(text="\u23F8 Pause" if playing[0] else "\u25B6 Play")

        play_pause_btn = ctk.CTkButton(ctrl, text="\u23F8 Pause", width=90, height=30,
                                       fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
                                       corner_radius=6, command=toggle_play)
        play_pause_btn.pack(side="left", padx=10, pady=8)
        ctk.CTkButton(ctrl, text="\u23F9 Close", width=80, height=30,
                      fg_color=COLORS["danger"], hover_color=COLORS["danger_hover"],
                      corner_radius=6, command=on_close).pack(side="left", padx=4, pady=8)

        win.protocol("WM_DELETE_WINDOW", on_close)
        threading.Thread(target=play_loop, daemon=True).start()

    def _open_with_system_player(self, filepath):
        try:
            os.startfile(filepath)
        except Exception:
            subprocess.Popen(["explorer", os.path.abspath(filepath)])

    # ── Twilio / WhatsApp ──
    def _send_whatsapp_message(self, title, message):
        if not self.twilio_enabled:
            return False
        if not all([self.account_sid, self.auth_token, self.twilio_phone, self.recipient_phone]):
            return False
        try:
            client = Client(self.account_sid, self.auth_token)
            full_msg = (f"{title}\n{message}\n"
                        f"Location: {self.current_camera_location}\n"
                        f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            client.messages.create(
                body=full_msg,
                from_=f"whatsapp:{self.twilio_phone}",
                to=f"whatsapp:{self.recipient_phone}",
            )
            return True
        except Exception as e:
            print(f"WhatsApp error: {e}")
            return False

    def _send_notification(self, title, message):
        now = time.time()
        if now - self.last_notification_time >= self.notification_cooldown:
            if self.twilio_enabled:
                self._send_whatsapp_message(title, f"{message} at {self.current_camera_location}")
            self.last_notification_time = now

    # ── Export ──
    def export_logs(self):
        if not self.detection_log:
            messagebox.showinfo("Export", "No detections recorded yet.")
            return
        try:
            ts = time.strftime("%Y%m%d-%H%M%S")
            filename = f"accident_logs_{ts}.csv"
            pd.DataFrame(self.detection_log).to_csv(filename, index=False)
            messagebox.showinfo("Export", f"Logs exported to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {e}")

    # ── Run / Close ──
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.root.mainloop()

    def _on_closing(self):
        self.processing = False
        self.media_processing = False
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    app = AccidentDetectionApp()
    app.run()
