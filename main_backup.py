import tkinter as tk
from tkinter import ttk, messagebox
import cv2
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

class AccidentDetectionApp:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Real-Time Accident Detection")
        self.root.geometry("1000x800")

        # Set theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Variables
        self.processing = False
        self.cap = None
        self.model = YOLO('best.pt')
        self.confidence_threshold = 0.5  # Detection confidence threshold
        self.last_notification_time = 0
        self.notification_cooldown = 10  # Seconds between notifications
        self.accident_frames_threshold = 5  # Number of consecutive frames to confirm accident
        self.accident_frames_count = 0
        self.save_evidence = True  # Save evidence of detected accidents
        self.detected_accidents = set()  # Track unique accidents

        # Camera location settings
        self.camera_locations = {
            0: "Main Entrance",
            1: "Parking Lot"
        }
        self.current_camera_location = "Main Entrance"  # Default location
        # Twilio configuration
        self.twilio_enabled = False
        self.account_sid = ""
        self.auth_token = ""
        self.twilio_phone = ""  # Your Twilio phone number
        self.recipient_phone = ""  # Recipient's phone number
        self.use_whatsapp = True  # Flag to enable WhatsApp messaging
        self.whatsapp_template_sid = ""  # Your WhatsApp template SID


        # Analytics variables
        self.detection_log = []
        self.total_accidents = 0
        self.current_accident_id = None

        # Load class list
        try:
            with open("coco1.txt", "r") as my_file:
                self.class_list = my_file.read().split("\n")
        except FileNotFoundError:
            messagebox.showerror("Error", "Class list file 'coco1.txt' not found!")
            self.class_list = []

        # Create evidence directory if it doesn't exist
        if not os.path.exists("accident_evidence"):
            os.makedirs("accident_evidence")

        self.create_widgets()

    def create_widgets(self):
        """Creates the UI components."""
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Title
        title_label = ctk.CTkLabel(
            self.main_frame,
            text="Real-Time Accident Detection",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=("#1f538d", "#2d7cd6")
        )
        title_label.pack(pady=10)

        # Tab view for settings and notifications
        self.tabview = ctk.CTkTabview(self.main_frame)
        self.tabview.pack(fill="x", pady=10)

        # Create tabs
        self.tabview.add("Detection Settings")
        self.tabview.add("Notification Settings")
        self.tabview.add("Camera Locations")

        # Settings frame
        settings_frame = self.tabview.tab("Detection Settings")

        # Configuration in settings tab
        conf_label = ctk.CTkLabel(settings_frame, text="Detection Confidence:")
        conf_label.grid(row=0, column=0, padx=10, pady=5)

        self.conf_slider = ctk.CTkSlider(
            settings_frame,
            from_=0,
            to=1,
            number_of_steps=20,
            command=self.update_confidence
        )
        self.conf_slider.set(self.confidence_threshold)
        self.conf_slider.grid(row=0, column=1, padx=10, pady=5)

        self.conf_value_label = ctk.CTkLabel(settings_frame, text=f"{self.confidence_threshold:.2f}")
        self.conf_value_label.grid(row=0, column=2, padx=10, pady=5)

        # Camera selection
        camera_label = ctk.CTkLabel(settings_frame, text="Camera Source:")
        camera_label.grid(row=1, column=0, padx=10, pady=5)

        self.camera_var = tk.IntVar(value=0)
        camera_option1 = ctk.CTkRadioButton(settings_frame, text="Webcam", variable=self.camera_var, value=0, command=self.update_camera_location)
        camera_option1.grid(row=1, column=1, padx=10, pady=5, sticky="w")

        camera_option2 = ctk.CTkRadioButton(settings_frame, text="External Camera", variable=self.camera_var, value=1, command=self.update_camera_location)
        camera_option2.grid(row=1, column=2, padx=10, pady=5, sticky="w")

        # Save evidence checkbox
        self.save_evidence_var = tk.BooleanVar(value=self.save_evidence)
        save_evidence_cb = ctk.CTkCheckBox(
            settings_frame,
            text="Save accident evidence",
            variable=self.save_evidence_var,
            command=self.toggle_save_evidence
        )
        save_evidence_cb.grid(row=2, column=0, padx=10, pady=5, sticky="w", columnspan=2)

        # Add WhatsApp button to Detection Settings
        test_whatsapp_btn = ctk.CTkButton(
            settings_frame,
            text="🔔 Test WhatsApp Notification",
            command=self.test_twilio_whatsapp,
            fg_color="#28a745",
            hover_color="#218838"
        )
        test_whatsapp_btn.grid(row=3, column=0, padx=10, pady=15, columnspan=3)

        # Camera location settings
        location_frame = self.tabview.tab("Camera Locations")

        location_label = ctk.CTkLabel(
            location_frame,
            text="Configure Camera Locations",
            font=ctk.CTkFont(weight="bold")
        )
        location_label.grid(row=0, column=0, padx=10, pady=10, columnspan=2)

        # Camera 0 location
        cam0_label = ctk.CTkLabel(location_frame, text="Webcam Location:")
        cam0_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")

        self.cam0_entry = ctk.CTkEntry(location_frame, width=200)
        self.cam0_entry.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        self.cam0_entry.insert(0, self.camera_locations[0])

        # Camera 1 location
        cam1_label = ctk.CTkLabel(location_frame, text="External Camera Location:")
        cam1_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")

        self.cam1_entry = ctk.CTkEntry(location_frame, width=200)
        self.cam1_entry.grid(row=2, column=1, padx=10, pady=5, sticky="w")
        self.cam1_entry.insert(0, self.camera_locations[1])

        # Save location settings button
        save_loc_btn = ctk.CTkButton(
            location_frame,
            text="Save Location Settings",
            command=self.save_location_settings,
            fg_color="#007bff",
            hover_color="#0069d9"
        )
        save_loc_btn.grid(row=3, column=0, padx=10, pady=15, columnspan=2)

        # WhatsApp notification settings
        notification_frame = self.tabview.tab("Notification Settings")

        # Enable Twilio checkbox
        self.twilio_enabled_var = tk.BooleanVar(value=self.twilio_enabled)
        twilio_cb = ctk.CTkCheckBox(
            notification_frame,
            text="Enable WhatsApp Notifications",
            variable=self.twilio_enabled_var,
            command=self.toggle_twilio
        )
        twilio_cb.grid(row=0, column=0, padx=10, pady=5, sticky="w", columnspan=2)

        # Twilio Account SID
        sid_label = ctk.CTkLabel(notification_frame, text="Twilio Account SID:")
        sid_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")

        self.sid_entry = ctk.CTkEntry(notification_frame, width=300, placeholder_text="Enter your Twilio Account SID")
        self.sid_entry.grid(row=2, column=1, padx=10, pady=5, sticky="w")
        self.sid_entry.insert(0, self.account_sid)

        # Twilio Auth Token
        token_label = ctk.CTkLabel(notification_frame, text="Twilio Auth Token:")
        token_label.grid(row=3, column=0, padx=10, pady=5, sticky="w")

        self.token_entry = ctk.CTkEntry(notification_frame, width=300, placeholder_text="Enter your Twilio Auth Token", show="*")
        self.token_entry.grid(row=3, column=1, padx=10, pady=5, sticky="w")
        self.token_entry.insert(0, self.auth_token)

        # Twilio Phone Number
        twilio_phone_label = ctk.CTkLabel(notification_frame, text="Twilio Phone Number:")
        twilio_phone_label.grid(row=4, column=0, padx=10, pady=5, sticky="w")

        self.twilio_phone_entry = ctk.CTkEntry(notification_frame, width=300, placeholder_text="+1XXXXXXXXXX")
        self.twilio_phone_entry.grid(row=4, column=1, padx=10, pady=5, sticky="w")
        self.twilio_phone_entry.insert(0, self.twilio_phone)

        # Recipient Phone Number
        recipient_label = ctk.CTkLabel(notification_frame, text="Recipient's Phone Number:")
        recipient_label.grid(row=5, column=0, padx=10, pady=5, sticky="w")

        self.recipient_entry = ctk.CTkEntry(notification_frame, width=300, placeholder_text="+1XXXXXXXXXX")
        self.recipient_entry.grid(row=5, column=1, padx=10, pady=5, sticky="w")
        self.recipient_entry.insert(0, self.recipient_phone)

        # Save Twilio settings button
        save_twilio_btn = ctk.CTkButton(
            notification_frame,
            text="Save Notification Settings",
            command=self.save_twilio_settings,
            fg_color="#007bff",
            hover_color="#0069d9"
        )
        save_twilio_btn.grid(row=7, column=0, padx=10, pady=15, columnspan=2)

        # Control buttons
        button_frame = ctk.CTkFrame(self.main_frame)
        button_frame.pack(fill="x", pady=10)

        self.process_btn = ctk.CTkButton(
            button_frame,
            text="▶️ Start Detection",
            command=self.toggle_processing,
            fg_color="#28a745",
            hover_color="#218838"
        )
        # Make the Start Detection button fill the entire width
        self.process_btn.pack(side="left", padx=10, expand=True, fill="x")

        # Status label
        status_frame = ctk.CTkFrame(self.main_frame)
        status_frame.pack(fill="x", pady=5)

        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ctk.CTkLabel(
            status_frame,
            textvariable=self.status_var,
            font=ctk.CTkFont(weight="bold")
        )
        self.status_label.pack(side="left", padx=10)

        # Video display
        self.video_frame = ctk.CTkFrame(self.main_frame)
        self.video_frame.pack(fill="both", expand=True, pady=10)

        self.video_label = ctk.CTkLabel(self.video_frame, text="No Camera Feed")
        self.video_label.pack(expand=True, pady=20)

        # Add logs tab and statistics
        self.create_log_tab()

    def update_camera_location(self):
        """Updates the current camera location when camera changes."""
        camera_id = self.camera_var.get()
        self.current_camera_location = self.camera_locations.get(camera_id, f"Camera {camera_id}")
        print(f"Camera changed to {self.current_camera_location}")

    def save_location_settings(self):
        """Saves the camera location settings."""
        self.camera_locations[0] = self.cam0_entry.get().strip()
        self.camera_locations[1] = self.cam1_entry.get().strip()

        # Update current location
        self.update_camera_location()

        messagebox.showinfo("Success", "Camera location settings saved successfully!")

    def toggle_twilio(self):
        """Toggles Twilio notifications."""
        self.twilio_enabled = self.twilio_enabled_var.get()

    def save_twilio_settings(self):
        """Saves the Twilio configuration settings."""
        self.account_sid = self.sid_entry.get().strip()
        self.auth_token = self.token_entry.get().strip()
        self.twilio_phone = self.twilio_phone_entry.get().strip()
        self.recipient_phone = self.recipient_entry.get().strip()

        # Basic validation
        if self.twilio_enabled:
            if not all([self.account_sid, self.auth_token, self.twilio_phone, self.recipient_phone]):
                messagebox.showerror("Error", "Please fill in all WhatsApp settings fields!")
                self.twilio_enabled_var.set(False)
                self.twilio_enabled = False
                return

            # Simple validation for phone numbers
            if not (self.twilio_phone.startswith('+') and self.recipient_phone.startswith('+')):
                messagebox.showwarning("Warning", "Phone numbers should be in E.164 format (e.g., +1XXXXXXXXXX)")

        messagebox.showinfo("Success", "WhatsApp notification settings saved successfully!")

    def test_twilio_whatsapp(self):
        """Tests the WhatsApp notification."""
        if not self.twilio_enabled:
            messagebox.showinfo("Info", "WhatsApp notifications are disabled. Please enable them first.")
            return

        if not all([self.account_sid, self.auth_token]):
            messagebox.showerror("Error", "Please fill in all WhatsApp settings!")
            return

        try:
            # Send a test message
            result = self.send_whatsapp_message("Test", "This is a test message from the Accident Detection System.")

            if result:
                messagebox.showinfo("Success", "Test WhatsApp message sent successfully!")
                self.status_var.set("WhatsApp test successful!")

                # Reset the status after 3 seconds
                self.root.after(3000, lambda: self.status_var.set("Ready"))
            else:
                messagebox.showerror("Error", "Failed to send WhatsApp message. Check your settings.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to send WhatsApp message: {str(e)}")
            self.status_var.set(f"WhatsApp test failed: {str(e)[:50]}...")

    def create_log_tab(self):
        """Creates the log and statistics tab."""
        log_frame = ctk.CTkFrame(self.main_frame)
        log_frame.pack(fill="x", pady=10)

        log_label = ctk.CTkLabel(
            log_frame,
            text="Detection History",
            font=ctk.CTkFont(weight="bold")
        )
        log_label.pack(pady=5)

        # Create button to view logs
        view_logs_btn = ctk.CTkButton(
            log_frame,
            text="View Detection History",
            command=self.show_logs,
            fg_color="#6c757d",
            hover_color="#5a6268"
        )
        view_logs_btn.pack(pady=5)

    def show_logs(self):
        """Shows the detection logs in a new window."""
        if not self.detection_log:
            messagebox.showinfo("Logs", "No accidents have been detected yet.")
            return

        log_window = ctk.CTkToplevel(self.root)
        log_window.title("Accident Detection Logs")
        log_window.geometry("600x400")

        # Create scrollable frame
        scroll_frame = ctk.CTkScrollableFrame(log_window)
        scroll_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Add header
        header = ctk.CTkLabel(
            scroll_frame,
            text="Accident Detection History",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        header.pack(pady=10)

        # Add log entries
        for i, entry in enumerate(self.detection_log):
            entry_frame = ctk.CTkFrame(scroll_frame)
            entry_frame.pack(fill="x", pady=5)

            time_label = ctk.CTkLabel(
                entry_frame,
                text=f"Incident #{i+1} - {entry['timestamp']} at {entry['location']}",
                font=ctk.CTkFont(weight="bold")
            )
            time_label.pack(anchor="w", padx=10, pady=5)

            details = ctk.CTkLabel(
                entry_frame,
                text=f"Evidence: {entry['evidence']}\nConfidence: {entry['confidence']:.2f}"
            )
            details.pack(anchor="w", padx=10)

            # Add button to view image
            view_btn = ctk.CTkButton(
                entry_frame,
                text="View Image",
                command=lambda file=entry['evidence']: self.view_evidence(file),
                width=100
            )
            view_btn.pack(anchor="e", padx=10, pady=5)

    def view_evidence(self, filepath):
        """Opens a window to view the evidence image."""
        if not os.path.exists(filepath):
            messagebox.showerror("Error", f"File not found: {filepath}")
            return

        try:
            # Open image in a new window
            img_window = ctk.CTkToplevel(self.root)
            img_window.title("Accident Evidence")
            img_window.geometry("800x600")

            img = cv2.imread(filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(img)

            img_label = ctk.CTkLabel(img_window, image=img)
            img_label.image = img  # Keep a reference
            img_label.pack(expand=True, fill="both", padx=10, pady=10)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image: {e}")

    def update_confidence(self, value):
        """Updates the confidence threshold."""
        self.confidence_threshold = float(value)
        self.conf_value_label.configure(text=f"{self.confidence_threshold:.2f}")

    def toggle_save_evidence(self):
        """Toggles saving evidence of detected accidents."""
        self.save_evidence = self.save_evidence_var.get()

    def toggle_processing(self):
        """Toggles real-time accident detection."""
        if not self.processing:
            self.processing = True
            self.process_btn.configure(text="⏹️ Stop Detection", fg_color="#dc3545", hover_color="#c82333")
            self.status_var.set("Detecting...")
            threading.Thread(target=self.process_camera_feed, daemon=True).start()
        else:
            self.processing = False
            self.process_btn.configure(text="▶️ Start Detection", fg_color="#28a745", hover_color="#218838")
            self.status_var.set("Stopped")

    def send_whatsapp_message(self, title, message):
        """Sends a WhatsApp message using Twilio."""
        if not self.twilio_enabled:
            return False

        if not all([self.account_sid, self.auth_token, self.twilio_phone, self.recipient_phone]):
            self.status_var.set("WhatsApp configuration incomplete")
            return False

        try:
            # Create Twilio client
            client = Client(self.account_sid, self.auth_token)

            # Format phone numbers for WhatsApp
            from_whatsapp = f"whatsapp:{self.twilio_phone}"
            to_whatsapp = f"whatsapp:{self.recipient_phone}"

            # Create message content with location information
            full_message = f"{title}\n{message}\nLocation: {self.current_camera_location}\nTime: {time.strftime('%Y-%m-%d %H:%M:%S')}"

            # Send direct WhatsApp message (not using templates)
            message = client.messages.create(
                body=full_message,
                from_=from_whatsapp,
                to=to_whatsapp
            )

            print(f"WhatsApp message sent with SID: {message.sid}")
            return True

        except Exception as e:
            print(f"WhatsApp message error: {e}")
            self.status_var.set(f"Message failed: {str(e)[:50]}...")
            return False

    def send_notification(self, title, message):
        """Sends a WhatsApp notification directly."""
        current_time = time.time()

        # Check if enough time has passed since the last notification
        if current_time - self.last_notification_time >= self.notification_cooldown:
            try:
                # Send message via WhatsApp if enabled
                if self.twilio_enabled:
                    # Add location to the message
                    full_message = f"{message} at {self.current_camera_location}"
                    msg_sent = self.send_whatsapp_message(title, full_message)
                    if msg_sent:
                        print("WhatsApp notification sent successfully")
                        self.status_var.set("WhatsApp notification sent")
                        # Reset status after 3 seconds
                        self.root.after(3000, lambda: self.status_var.set("Detecting..."))
                    else:
                        print("Failed to send WhatsApp notification")
                        self.status_var.set("Failed to send WhatsApp notification")

                self.last_notification_time = current_time
                return True
            except Exception as e:
                print(f"Notification error: {e}")
                # Add visual feedback if notification fails
                self.status_var.set(f"Notification failed: {str(e)[:50]}...")
                return False
        return False

    def generate_accident_id(self, frame):
        """Generate a unique identifier for an accident scene."""
        # Use a simplified hash of the frame to identify similar accidents
        small_frame = cv2.resize(frame, (32, 32))  # Resize to small dimensions for faster processing
        return hash(small_frame.tobytes())

    def save_accident_frame(self, frame):
        """Saves the frame as evidence and returns the filepath."""
        if self.save_evidence:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filepath = f"accident_evidence/accident_{timestamp}.jpg"
            cv2.imwrite(filepath, frame)

            # Log the detection with location
            self.detection_log.append({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "evidence": filepath,
                "confidence": self.confidence_threshold,
                "location": self.current_camera_location
            })
            return filepath
        return None

    def process_camera_feed(self):
        """Processes the live camera feed for accident detection."""
        camera_id = self.camera_var.get()
        # Update the current camera location
        self.update_camera_location()

        self.cap = cv2.VideoCapture(camera_id)

        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Failed to open camera (ID: {camera_id})!")
            self.processing = False
            self.process_btn.configure(text="▶️ Start Detection", fg_color="#28a745", hover_color="#218838")
            self.status_var.set("Camera error")
            return

        frame_width = 800
        frame_height = 450

        # Frame processing optimization
        skip_frames = 0
        max_skip_frames = 2
        frame_count = 0

        while self.processing:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1

            # Skip frames for performance if needed
            if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                continue

            # Process frame with YOLO model
            frame = cv2.resize(frame, (frame_width, frame_height))
            results = self.model.predict(frame, conf=self.confidence_threshold)

            accident_detected = False

            if results and len(results[0].boxes) > 0:
                px = pd.DataFrame(results[0].boxes.data).astype("float")

                for _, row in px.iterrows():
                    x1, y1, x2, y2 = map(int, row[:4])
                    confidence = row[4]
                    class_id = int(row[5])

                    if class_id < len(self.class_list):
                        class_name = self.class_list[class_id]
                    else:
                        class_name = "Unknown"

                    # Apply additional filtering for accident detection
                    if 'accident' in class_name.lower() and confidence >= self.confidence_threshold:
                        # Mark as accident
                        accident_detected = True

                        # Draw bounding box for the accident
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                        # Add confidence label
                        conf_label = f"{class_name}: {confidence:.2f}"
                        cv2.putText(frame, conf_label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                        # Increment accident frames counter
                        self.accident_frames_count += 1

                        # If accident has been detected for enough consecutive frames, trigger alert
                        if self.accident_frames_count >= self.accident_frames_threshold:
                            # Generate an ID for this accident
                            accident_id = self.generate_accident_id(frame)

                            # Only count and notify for unique accidents
                            if accident_id not in self.detected_accidents:
                                # Save the accident ID
                                self.detected_accidents.add(accident_id)
                                self.current_accident_id = accident_id

                                # Increment total accidents - for internal tracking only
                                self.total_accidents += 1

                                # Save evidence
                                evidence_path = self.save_accident_frame(frame)

                                # Send Twilio WhatsApp notification directly
                                notification_title = "⚠️ ACCIDENT DETECTED!"
                                notification_message = f"Accident detected with {confidence:.2f} confidence."
                                self.send_notification(notification_title, notification_message)

                    # Draw all other detections with different color
                    elif confidence >= self.confidence_threshold:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{class_name}: {confidence:.2f}"
                        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Reset accident frames counter if no accident detected in this frame
            if not accident_detected:
                self.accident_frames_count = 0

            # Add current camera location to frame
            cv2.putText(frame, f"Location: {self.current_camera_location}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Add timestamp to frame
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (frame_width - 200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Convert frame for display in tkinter
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            img = ImageTk.PhotoImage(image=img)

            # Update the video label
            self.video_label.configure(image=img)
            self.video_label.image = img

            # Process UI events
            self.root.update_idletasks()
            self.root.update()

        # Release the camera when processing stops
        if self.cap is not None:
            self.cap.release()

        # Clear the video label when stopped
        self.video_label.configure(image=None)
        self.video_label.image = None
        self.video_label.configure(text="No Camera Feed")

    def export_logs(self):
        """Exports the detection logs to a CSV file."""
        if not self.detection_log:
            messagebox.showinfo("Export", "No accidents have been detected yet. Nothing to export.")
            return

        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"accident_logs_{timestamp}.csv"

            # Convert log to DataFrame
            log_df = pd.DataFrame(self.detection_log)
            log_df.to_csv(filename, index=False)

            messagebox.showinfo("Export", f"Logs exported successfully to {filename}")

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export logs: {str(e)}")

    def run(self):
        """Runs the application."""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def on_closing(self):
        """Handles application closure."""
        if self.processing:
            # Stop processing before closing
            self.processing = False
            if self.cap is not None:
                self.cap.release()

        # Clean up resources and exit
        self.root.destroy()

if __name__ == "__main__":
    app = AccidentDetectionApp()
    app.run()