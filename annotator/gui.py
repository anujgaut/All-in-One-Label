# annotator/gui.py
import os
import glob
import copy
import json
import time
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from PIL import Image, ImageTk

try:
    import cv2
except ImportError:
    cv2 = None

from .models import Annotation
from .utils import point_in_polygon


# Constants for minimum canvas size.
CANVAS_MIN_WIDTH = 800
CANVAS_MIN_HEIGHT = 600

class ImageVideoAnnotator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("YOLO / Mask R-CNN Annotator")
        self.geometry("1400x900")
        self.protocol("WM_DELETE_WINDOW", self.on_exit)

        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self.configure_styles()

        # State Variables
        self.image_list = []
        self.current_image_index = -1
        self.image_path = None
        self.image_obj = None
        self.display_image = None
        self.initial_scale = 1.0
        self.zoom_factor = 1.0
        self.pan_offset = [0, 0]
        self.labels = []
        self.annotations = []
        self.undo_stack = []
        self.redo_stack = []
        self.selected_annotation = None
        self.annotation_mode = "bbox"  # or "polygon"
        self.temp_polygon_points = []
        self.drawing = False
        self.start_point = None

        self.edit_mode = False
        self.move_mode = False
        self.resize_mode = False
        self.resize_handle = None

        self.image_status = {}
        self.selected_class = None

        self.create_header()
        self.create_menu()
        self.create_widgets()
        self.bind_events()

        self.auto_save_interval = 60000  # 1 minute
        self.after(self.auto_save_interval, self.auto_save_project)

        self.current_theme = "light"
        self.apply_theme()

    def configure_styles(self):
        self.style.configure("Header.TLabel", font=("Helvetica", 20, "bold"), foreground="#2e6da4")
        self.style.configure("TButton", font=("Helvetica", 10), padding=5)
        self.style.configure("Class.TButton", font=("Helvetica", 10), padding=5,
                             relief="solid", borderwidth=1)
        self.style.configure("SelectedClass.TButton", font=("Helvetica", 10, "bold"),
                             padding=5, relief="solid", borderwidth=1,
                             background="#007acc", foreground="white")
        self.style.map("SelectedClass.TButton", background=[("active", "#005b99")])

    def point_in_polygon(self, x, y, poly_points):
        return point_in_polygon(x, y, poly_points)

    def create_header(self):
        header_frame = ttk.Frame(self, padding=10)
        header_frame.pack(side=tk.TOP, fill=tk.X)
        header_label = ttk.Label(header_frame,
                                 text="YOLO / Mask R-CNN Annotator",
                                 style="Header.TLabel")
        header_label.pack(side=tk.TOP)

    def create_menu(self):
        self.menu_bar = tk.Menu(self)
        # File Menu
        file_menu = tk.Menu(self.menu_bar, tearoff=0)
        file_menu.add_command(label="Load Image Folder", command=self.load_folder)
        if cv2:
            file_menu.add_command(label="Load Video", command=self.load_video)
        file_menu.add_separator()
        file_menu.add_command(label="Save Project", command=self.save_project)
        file_menu.add_command(label="Load Project", command=self.load_project)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_exit)
        self.menu_bar.add_cascade(label="File", menu=file_menu)
        # Edit Menu
        edit_menu = tk.Menu(self.menu_bar, tearoff=0)
        edit_menu.add_command(label="Undo (Z)", command=self.undo)
        edit_menu.add_command(label="Redo (Y)", command=self.redo)
        edit_menu.add_command(label="Delete Selected (D)", command=self.delete_selected_annotation)
        edit_menu.add_command(label="Manage Classes", command=self.manage_labels)
        self.menu_bar.add_cascade(label="Edit", menu=edit_menu)
        # View Menu
        view_menu = tk.Menu(self.menu_bar, tearoff=0)
        theme_menu = tk.Menu(view_menu, tearoff=0)
        theme_menu.add_command(label="Light Mode", command=lambda: self.set_theme("light"))
        theme_menu.add_command(label="Dark Mode", command=lambda: self.set_theme("dark"))
        view_menu.add_cascade(label="Themes", menu=theme_menu)
        self.menu_bar.add_cascade(label="View", menu=view_menu)
        # Project Menu
        project_menu = tk.Menu(self.menu_bar, tearoff=0)
        project_menu.add_command(label="Next Image (N)", command=self.next_image)
        project_menu.add_command(label="Previous Image", command=self.previous_image)
        self.menu_bar.add_cascade(label="Project", menu=project_menu)
        # Export Menu
        export_menu = tk.Menu(self.menu_bar, tearoff=0)
        export_menu.add_command(label="Export YOLO Format", command=self.export_yolo)
        export_menu.add_command(label="Export Pascal VOC", command=self.export_voc)
        export_menu.add_command(label="Export COCO JSON", command=self.export_coco)
        export_menu.add_command(label="Export CSV", command=self.export_csv)
        self.menu_bar.add_cascade(label="Export", menu=export_menu)
        # Tools Menu
        tools_menu = tk.Menu(self.menu_bar, tearoff=0)
        tools_menu.add_command(label="AI Pre-label (YOLOv11)", command=self.ai_prelabel)
        tools_menu.add_command(label="Train Custom Model", command=self.train_custom_model)
        tools_menu.add_command(label="Test Model", command=self.test_model)  # New test model option
        tools_menu.add_command(label="Quality Check", command=self.quality_check)
        tools_menu.add_command(label="Split Dataset", command=self.split_dataset)
        self.menu_bar.add_cascade(label="Tools", menu=tools_menu)
        
        # Help Menu
        help_menu = tk.Menu(self.menu_bar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        self.menu_bar.add_cascade(label="Help", menu=help_menu)
        self.config(menu=self.menu_bar)

    def create_widgets(self):
        # Left Frame: Image List
        self.left_frame = ttk.Frame(self, padding=10)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.btn_load_folder = ttk.Button(self.left_frame, text="Load Folder", command=self.load_folder)
        self.btn_load_folder.pack(fill=tk.X, pady=5)
        self.tree = ttk.Treeview(self.left_frame, columns=("Filename",), show="headings", height=25)
        self.tree.heading("Filename", text="Image Files")
        self.tree.column("Filename", width=280)
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)
        self.tree.tag_configure("incomplete", background="light coral")
        self.tree.tag_configure("completed", background="light green")

        # Right Frame: Canvas and Class Sidebar
        self.right_frame = ttk.Frame(self, padding=10)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        # Canvas Frame
        self.canvas_frame = ttk.Frame(self.right_frame)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(self.canvas_frame, bg="black", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        # Action Frame
        self.action_frame = ttk.Frame(self.canvas_frame, padding=10, borderwidth=2, relief="groove")
        self.action_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=5)
        self.btn_clear = ttk.Button(self.action_frame, text="Clear Annotations", command=self.clear_annotations)
        self.btn_clear.pack(side=tk.LEFT, padx=5)
        self.btn_save = ttk.Button(self.action_frame, text="Save Annotations", command=self.save_annotations)
        self.btn_save.pack(side=tk.LEFT, padx=5)
        self.btn_annotation_mode = ttk.Button(self.action_frame, text="Mode: BBox", command=self.toggle_annotation_mode)
        self.btn_annotation_mode.pack(side=tk.LEFT, padx=5)
        self.btn_finish_polygon = ttk.Button(self.action_frame, text="Finish Polygon", command=self.finish_polygon)
        self.btn_finish_polygon.pack(side=tk.LEFT, padx=5)
        self.btn_finish_polygon.state(["disabled"])
        self.btn_edit_mode = ttk.Button(self.action_frame, text="Edit Mode: Off", command=self.toggle_edit_mode)
        self.btn_edit_mode.pack(side=tk.LEFT, padx=5)
        self.btn_next = ttk.Button(self.action_frame, text="Next Image", command=self.next_image)
        self.btn_next.pack(side=tk.LEFT, padx=5)

        # Class Sidebar
        self.class_frame = ttk.Frame(self.right_frame, relief="groove", padding=10)
        self.class_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        self.class_frame.columnconfigure(0, weight=1)
        header = ttk.Label(self.class_frame, text="Available Classes", font=("Helvetica", 12, "bold"))
        header.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        self.class_canvas = tk.Canvas(self.class_frame, height=300)
        self.class_canvas.grid(row=1, column=0, sticky="nsew")
        self.class_frame.rowconfigure(1, weight=3)
        self.scrollbar = ttk.Scrollbar(self.class_frame, orient="vertical", command=self.class_canvas.yview)
        self.scrollbar.grid(row=1, column=1, sticky="ns")
        self.class_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.inner_class_frame = ttk.Frame(self.class_canvas)
        self.inner_class_window = self.class_canvas.create_window((0, 0), window=self.inner_class_frame, anchor="nw")
        self.inner_class_frame.bind("<Configure>", lambda e: self.class_canvas.configure(scrollregion=self.class_canvas.bbox("all")))
        self.class_canvas.bind("<Configure>", lambda e: self.class_canvas.itemconfig(self.inner_class_window, width=e.width))
        self.system_message_label = tk.Label(self.class_frame, text="", anchor="center", font=("Helvetica", 10), fg="green", bd=1, relief="sunken")
        self.system_message_label.config(height=3)
        self.system_message_label.grid(row=2, column=0, sticky="ew", pady=5)
        self.add_frame = ttk.Frame(self.class_frame, padding=(0, 10))
        self.add_frame.grid(row=3, column=0, sticky="ew")
        ttk.Label(self.add_frame, text="New Class:", font=("Helvetica", 10)).pack(side=tk.LEFT, padx=2)
        self.entry_new_class = ttk.Entry(self.add_frame, font=("Helvetica", 10))
        self.entry_new_class.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        btn_add_class = ttk.Button(self.add_frame, text="Add", command=self.add_new_class)
        btn_add_class.pack(side=tk.LEFT, padx=2)
        btn_remove_class = ttk.Button(self.add_frame, text="Remove Selected", command=self.remove_selected_class)
        btn_remove_class.pack(side=tk.LEFT, padx=2)
        self.update_class_buttons()

    def update_class_buttons(self):
        for widget in self.inner_class_frame.winfo_children():
            widget.destroy()
        if self.labels:
            if self.selected_class not in self.labels:
                self.selected_class = self.labels[0]
            for lab in self.labels:
                style = "SelectedClass.TButton" if lab == self.selected_class else "Class.TButton"
                btn = ttk.Button(self.inner_class_frame, text=lab, style=style, command=lambda lab=lab: self.select_class(lab))
                btn.pack(fill=tk.X, pady=2, padx=2)
        else:
            self.selected_class = None

    def select_class(self, lab):
        self.selected_class = lab
        self.update_class_buttons()

    def bind_events(self):
        self.canvas.bind("<ButtonPress-1>", self.on_left_button_press)
        self.canvas.bind("<B1-Motion>", self.on_left_button_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_button_release)
        self.canvas.bind("<Double-Button-1>", self.on_double_click)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)
        self.canvas.bind("<ButtonPress-3>", self.on_right_button_press)
        self.canvas.bind("<B3-Motion>", self.on_right_button_drag)
        self.bind("<n>", lambda event: self.next_image())
        self.bind("<N>", lambda event: self.next_image())
        self.bind("<p>", lambda event: self.previous_image())
        self.bind("<d>", lambda event: self.delete_selected_annotation())
        self.bind("<D>", lambda event: self.delete_selected_annotation())
        self.bind("<z>", lambda event: self.undo())
        self.bind("<Z>", lambda event: self.undo())
        self.bind("<y>", lambda event: self.redo())
        self.bind("<Y>", lambda event: self.redo())
        self.bind("<Control-s>", lambda event: self.save_project())
        self.bind("<Control-z>", lambda event: self.undo())
        self.bind("<Control-y>", lambda event: self.redo())
        self.canvas.bind("<Configure>", self.on_canvas_configure)

    def on_canvas_configure(self, event):
        self.redraw_canvas()

    def apply_theme(self):
        if self.current_theme == "dark":
            canvas_bg = "black"
            self.style.configure("TFrame", background="#2e2e2e")
        else:
            canvas_bg = "gray"
            self.style.configure("TFrame", background="white")
        self.left_frame.configure(style="TFrame")
        self.right_frame.configure(style="TFrame")
        self.canvas_frame.configure(style="TFrame")
        self.action_frame.configure(style="TFrame")
        self.canvas.configure(bg=canvas_bg)

    def set_theme(self, theme):
        self.current_theme = theme
        self.apply_theme()

    def load_folder(self):
        folder = filedialog.askdirectory(title="Select Folder with Images")
        if folder:
            extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp")
            self.image_list = []
            for ext in extensions:
                self.image_list.extend(glob.glob(os.path.join(folder, ext)))
            self.image_list.sort()
            if not self.image_list:
                messagebox.showerror("Error", "No image files found in this folder!")
                return
            self.ask_labels()
            self.update_class_buttons()
            self.image_status = {}
            for widget in self.tree.get_children():
                self.tree.delete(widget)
            for idx, img in enumerate(self.image_list):
                self.image_status[img] = False
                self.tree.insert("", "end", iid=idx, values=(os.path.basename(img),), tags=("incomplete",))
            self.current_image_index = 0
            self.load_image(self.image_list[0])

    def ask_labels(self):
        label_str = simpledialog.askstring("Input Labels",
                                           "Enter labels separated by commas (e.g. person,car,cat):",
                                           parent=self)
        if label_str:
            self.labels = [lab.strip() for lab in label_str.split(",") if lab.strip()]
        else:
            self.labels = []

    def load_image(self, image_path):
        self.image_path = image_path
        try:
            self.image_obj = Image.open(image_path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open image: {e}")
            return
        self.zoom_factor = 1.0
        self.pan_offset = [0, 0]
        self.canvas.update_idletasks()
        canvas_width = max(self.canvas.winfo_width(), CANVAS_MIN_WIDTH)
        canvas_height = max(self.canvas.winfo_height(), CANVAS_MIN_HEIGHT)
        orig_width, orig_height = self.image_obj.size
        self.initial_scale = min(canvas_width / orig_width, canvas_height / orig_height, 1)
        self.annotations = []
        self.undo_stack = []
        self.redo_stack = []
        self.selected_annotation = None
        self.redraw_canvas()

    def redraw_canvas(self):
        self.canvas.delete("all")
        if self.image_obj:
            scale = self.initial_scale * self.zoom_factor
            new_size = (int(self.image_obj.width * scale), int(self.image_obj.height * scale))
            try:
                resample_filter = Image.Resampling.LANCZOS
            except AttributeError:
                resample_filter = Image.LANCZOS
            resized_image = self.image_obj.resize(new_size, resample_filter)
            self.display_image = ImageTk.PhotoImage(resized_image)
            self.canvas.create_image(self.pan_offset[0], self.pan_offset[1], anchor=tk.NW, image=self.display_image)
            for ann in self.annotations:
                self.draw_annotation(ann)

    def image_to_canvas(self, x, y):
        scale = self.initial_scale * self.zoom_factor
        return x * scale + self.pan_offset[0], y * scale + self.pan_offset[1]

    def canvas_to_image(self, x, y):
        scale = self.initial_scale * self.zoom_factor
        return (x - self.pan_offset[0]) / scale, (y - self.pan_offset[1]) / scale

    def draw_annotation(self, ann):
        for cid in ann.canvas_ids:
            self.canvas.delete(cid)
        ann.canvas_ids = []
        if ann.type == "bbox":
            x1, y1, x2, y2 = ann.points
            c1 = self.image_to_canvas(x1, y1)
            c2 = self.image_to_canvas(x2, y2)
            rect_id = self.canvas.create_rectangle(c1[0], c1[1], c2[0], c2[1],
                                                   outline="red", width=2, tags=("annotation", "bbox"))
            ann.canvas_ids.append(rect_id)
            text_id = self.canvas.create_text(c1[0] + 5, c1[1] + 5,
                                              anchor=tk.NW, text=ann.label, fill="yellow",
                                              tags=("annotation", "annotation_text"))
            ann.canvas_ids.append(text_id)
        elif ann.type == "polygon":
            pts = []
            for i in range(0, len(ann.points), 2):
                cx, cy = self.image_to_canvas(ann.points[i], ann.points[i+1])
                pts.extend([cx, cy])
            poly_id = self.canvas.create_polygon(pts, outline="green", fill="", width=2,
                                                   tags=("annotation", "polygon"))
            ann.canvas_ids.append(poly_id)
            if len(pts) >= 2:
                text_id = self.canvas.create_text(pts[0] + 5, pts[1] + 5,
                                                  anchor=tk.NW, text=ann.label, fill="yellow",
                                                  tags=("annotation", "annotation_text"))
                ann.canvas_ids.append(text_id)

    def get_selected_label(self):
        if not self.selected_class:
            messagebox.showerror("Error", "Please select a class from the sidebar.")
            return None
        return self.selected_class

    def add_new_class(self):
        new_class = self.entry_new_class.get().strip()
        if new_class:
            if new_class in self.labels:
                messagebox.showerror("Error", "This class already exists.")
            else:
                self.labels.append(new_class)
                if not self.selected_class:
                    self.selected_class = new_class
                self.update_class_buttons()
                self.entry_new_class.delete(0, tk.END)
        else:
            messagebox.showerror("Error", "Please enter a valid class name.")

    def remove_selected_class(self):
        if not self.selected_class:
            messagebox.showerror("Error", "Please select a class to remove.")
        else:
            self.labels.remove(self.selected_class)
            self.selected_class = self.labels[0] if self.labels else None
            self.update_class_buttons()

    def on_left_button_press(self, event):
        if self.edit_mode:
            found = False
            threshold = 10
            for ann in self.annotations:
                if ann.type == "bbox":
                    x1, y1, x2, y2 = ann.points
                    c1 = self.image_to_canvas(x1, y1)
                    c2 = self.image_to_canvas(x2, y2)
                    left, right = min(c1[0], c2[0]), max(c1[0], c2[0])
                    top, bottom = min(c1[1], c2[1]), max(c1[1], c2[1])
                    if left <= event.x <= right and top <= event.y <= bottom:
                        self.selected_annotation = ann
                        found = True
                        handles = {
                            "tl": (left, top),
                            "tr": (right, top),
                            "bl": (left, bottom),
                            "br": (right, bottom)
                        }
                        self.resize_mode = False
                        self.move_mode = False
                        for key, pos in handles.items():
                            dist = ((event.x - pos[0])**2 + (event.y - pos[1])**2)**0.5
                            if dist < threshold:
                                self.resize_mode = True
                                self.resize_handle = key
                                break
                        if not self.resize_mode:
                            self.move_mode = True
                        self.start_point = (event.x, event.y)
                        break
                elif ann.type == "polygon":
                    poly_canvas = []
                    for i in range(0, len(ann.points), 2):
                        pt = self.image_to_canvas(ann.points[i], ann.points[i+1])
                        poly_canvas.append(pt)
                    if self.point_in_polygon(event.x, event.y, poly_canvas):
                        self.selected_annotation = ann
                        found = True
                        break
            if not found:
                self.selected_annotation = None
            return
        else:
            overlapping_items = self.canvas.find_overlapping(event.x, event.y, event.x, event.y)
            for item in overlapping_items:
                if "annotation" in self.canvas.gettags(item):
                    for ann in self.annotations:
                        if item in ann.canvas_ids:
                            self.selected_annotation = ann
                            self.start_point = (event.x, event.y)
                            return
            self.drawing = True
            self.start_point = (event.x, event.y)
            if self.annotation_mode == "bbox":
                self.temp_draw = self.canvas.create_rectangle(event.x, event.y, event.x, event.y,
                                                              outline="red", width=2, dash=(2, 2))
            elif self.annotation_mode == "polygon":
                self.temp_polygon_points.append((event.x, event.y))
                r = 3
                self.canvas.create_oval(event.x - r, event.y - r, event.x + r, event.y + r,
                                        fill="blue", outline="blue", tags="temp_polygon")
                if len(self.temp_polygon_points) > 1:
                    self.canvas.create_line(self.temp_polygon_points[-2][0], self.temp_polygon_points[-2][1],
                                            event.x, event.y, fill="blue", dash=(2, 2), tags="temp_polygon")

    def on_left_button_drag(self, event):
        if self.edit_mode:
            if self.selected_annotation is None:
                return
            scale = self.initial_scale * self.zoom_factor
            if self.selected_annotation.type == "bbox":
                if self.resize_mode:
                    new_img_coord = self.canvas_to_image(event.x, event.y)
                    x_new, y_new = new_img_coord
                    if self.resize_handle == "tl":
                        self.selected_annotation.points[0] = x_new
                        self.selected_annotation.points[1] = y_new
                    elif self.resize_handle == "tr":
                        self.selected_annotation.points[2] = x_new
                        self.selected_annotation.points[1] = y_new
                    elif self.resize_handle == "bl":
                        self.selected_annotation.points[0] = x_new
                        self.selected_annotation.points[3] = y_new
                    elif self.resize_handle == "br":
                        self.selected_annotation.points[2] = x_new
                        self.selected_annotation.points[3] = y_new
                    self.redraw_canvas()
                elif self.move_mode:
                    dx = event.x - self.start_point[0]
                    dy = event.y - self.start_point[1]
                    dx_img = dx / scale
                    dy_img = dy / scale
                    self.selected_annotation.points[0] += dx_img
                    self.selected_annotation.points[1] += dy_img
                    self.selected_annotation.points[2] += dx_img
                    self.selected_annotation.points[3] += dy_img
                    self.start_point = (event.x, event.y)
                    self.redraw_canvas()
            return
        else:
            if self.drawing and self.annotation_mode == "bbox" and hasattr(self, "temp_draw"):
                self.canvas.coords(self.temp_draw, self.start_point[0], self.start_point[1], event.x, event.y)
            elif self.selected_annotation:
                dx = event.x - self.start_point[0]
                dy = event.y - self.start_point[1]
                scale = self.initial_scale * self.zoom_factor
                dx_img = dx / scale
                dy_img = dy / scale
                if self.selected_annotation.type == "bbox":
                    x1, y1, x2, y2 = self.selected_annotation.points
                    self.selected_annotation.points = [x1 + dx_img, y1 + dy_img, x2 + dx_img, y2 + dy_img]
                elif self.selected_annotation.type == "polygon":
                    self.selected_annotation.points = [p + dx_img if i % 2 == 0 else p + dy_img
                                                       for i, p in enumerate(self.selected_annotation.points)]
                self.start_point = (event.x, event.y)
                self.redraw_canvas()

    def on_left_button_release(self, event):
        if self.edit_mode:
            self.move_mode = False
            self.resize_mode = False
            self.resize_handle = None
            return
        else:
            if self.drawing and self.annotation_mode == "bbox":
                self.drawing = False
                if hasattr(self, "temp_draw"):
                    coords = self.canvas.coords(self.temp_draw)
                    self.canvas.delete(self.temp_draw)
                    if abs(coords[2] - coords[0]) < 10 or abs(coords[3] - coords[1]) < 10:
                        return
                    p1 = self.canvas_to_image(coords[0], coords[1])
                    p2 = self.canvas_to_image(coords[2], coords[3])
                    x1, y1 = p1
                    x2, y2 = p2
                    label = self.get_selected_label()
                    if not label:
                        return
                    from .models import Annotation
                    ann = Annotation("bbox", [x1, y1, x2, y2], label)
                    self.annotations.append(ann)
                    self.push_undo_state()
                    self.redraw_canvas()
            elif self.selected_annotation:
                self.push_undo_state()
                self.selected_annotation = None

    def on_double_click(self, event):
        if self.annotation_mode == "polygon" and len(self.temp_polygon_points) > 2:
            self.finish_polygon()

    def finish_polygon(self):
        if self.annotation_mode == "polygon" and len(self.temp_polygon_points) > 2:
            self.canvas.delete("temp_polygon")
            img_points = []
            for pt in self.temp_polygon_points:
                img_pt = self.canvas_to_image(pt[0], pt[1])
                img_points.extend(img_pt)
            label = self.get_selected_label()
            if not label:
                self.temp_polygon_points = []
                return
            from .models import Annotation
            ann = Annotation("polygon", img_points, label)
            self.annotations.append(ann)
            self.push_undo_state()
            self.temp_polygon_points = []
            self.redraw_canvas()

    def on_mouse_wheel(self, event):
        if event.num == 5 or event.delta < 0:
            factor = 0.9
        elif event.num == 4 or event.delta > 0:
            factor = 1.1
        else:
            factor = 1.0
        self.zoom_factor *= factor
        self.redraw_canvas()

    def on_right_button_press(self, event):
        self.pan_start = (event.x, event.y)

    def on_right_button_drag(self, event):
        dx = event.x - self.pan_start[0]
        dy = event.y - self.pan_start[1]
        self.pan_offset[0] += dx
        self.pan_offset[1] += dy
        self.pan_start = (event.x, event.y)
        self.redraw_canvas()

    def toggle_annotation_mode(self):
        if self.annotation_mode == "bbox":
            self.annotation_mode = "polygon"
            self.btn_annotation_mode.config(text="Mode: Polygon")
            self.btn_finish_polygon.state(["!disabled"])
        else:
            self.annotation_mode = "bbox"
            self.btn_annotation_mode.config(text="Mode: BBox")
            self.btn_finish_polygon.state(["disabled"])
            self.temp_polygon_points = []
            self.canvas.delete("temp_polygon")

    def toggle_edit_mode(self):
        self.edit_mode = not self.edit_mode
        if self.edit_mode:
            self.system_message_label.config(text="Edit Mode Activated: Select an annotation to move/resize or delete.")
            self.btn_edit_mode.config(text="Edit Mode: On")
        else:
            self.system_message_label.config(text="Edit Mode Deactivated: Back to annotation mode.")
            self.selected_annotation = None
            self.move_mode = False
            self.resize_mode = False
            self.resize_handle = None
            self.btn_edit_mode.config(text="Edit Mode: Off")
        self.after(3000, lambda: self.system_message_label.config(text=""))

    def push_undo_state(self):
        state = copy.deepcopy([ann.to_dict() for ann in self.annotations])
        self.undo_stack.append(state)
        self.redo_stack.clear()

    def undo(self):
        if self.undo_stack:
            state = self.undo_stack.pop()
            self.redo_stack.append(copy.deepcopy([ann.to_dict() for ann in self.annotations]))
            self.annotations = [Annotation.from_dict(d) for d in state]
            self.redraw_canvas()

    def redo(self):
        if self.redo_stack:
            state = self.redo_stack.pop()
            self.undo_stack.append(copy.deepcopy([ann.to_dict() for ann in self.annotations]))
            self.annotations = [Annotation.from_dict(d) for d in state]
            self.redraw_canvas()

    def delete_selected_annotation(self):
        if self.selected_annotation and self.selected_annotation in self.annotations:
            self.annotations.remove(self.selected_annotation)
            self.selected_annotation = None
            self.push_undo_state()
            self.redraw_canvas()

    def clear_annotations(self):
        if messagebox.askyesno("Clear", "Clear all annotations for current image?"):
            self.annotations = []
            self.push_undo_state()
            self.redraw_canvas()

    def save_annotations(self):
        from .export_tools import export_yolo_format
        export_yolo_format(self)

    def next_image(self):
        if self.current_image_index < len(self.image_list) - 1:
            self.current_image_index += 1
            self.tree.selection_set(self.current_image_index)
            self.load_image(self.image_list[self.current_image_index])
        else:
            messagebox.showinfo("Info", "Last image reached.")

    def previous_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.tree.selection_set(self.current_image_index)
            self.load_image(self.image_list[self.current_image_index])
        else:
            messagebox.showinfo("Info", "This is the first image.")

    def on_tree_select(self, event):
        selected = self.tree.selection()
        if selected:
            idx = int(selected[0])
            self.current_image_index = idx
            self.load_image(self.image_list[idx])

    def save_project(self):
        project = {
            "image_list": self.image_list,
            "current_image_index": self.current_image_index,
            "labels": self.labels,
            "annotations": [ann.to_dict() for ann in self.annotations]
        }
        file_path = filedialog.asksaveasfilename(defaultextension=".json",
                                                 filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, "w") as f:
                json.dump(project, f, indent=2)
            messagebox.showinfo("Project Saved", f"Project saved to {file_path}")

    def load_project(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, "r") as f:
                project = json.load(f)
            self.image_list = project.get("image_list", [])
            self.current_image_index = project.get("current_image_index", 0)
            self.labels = project.get("labels", [])
            self.update_class_buttons()
            ann_dicts = project.get("annotations", [])
            self.annotations = [Annotation.from_dict(d) for d in ann_dicts]
            for widget in self.tree.get_children():
                self.tree.delete(widget)
            for idx, img in enumerate(self.image_list):
                tag = "completed" if self.image_status.get(img, False) else "incomplete"
                self.tree.insert("", "end", iid=idx, values=(os.path.basename(img),), tags=(tag,))
            if self.image_list:
                self.load_image(self.image_list[self.current_image_index])
            messagebox.showinfo("Project Loaded", f"Project loaded from {file_path}")

    def auto_save_project(self):
        temp_file = "autosave_project.json"
        project = {
            "image_list": self.image_list,
            "current_image_index": self.current_image_index,
            "labels": self.labels,
            "annotations": [ann.to_dict() for ann in self.annotations],
            "timestamp": time.time()
        }
        with open(temp_file, "w") as f:
            json.dump(project, f, indent=2)
        self.after(self.auto_save_interval, self.auto_save_project)

    def load_video(self):
        if not cv2:
            messagebox.showerror("Error", "OpenCV is not installed. Video annotation is unavailable.")
            return
        video_path = filedialog.askopenfilename(title="Select Video File",
                                                filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
        if video_path:
            cap = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame)
                frames.append(pil_img)
            cap.release()
            temp_dir = "video_frames_temp"
            os.makedirs(temp_dir, exist_ok=True)
            self.image_list = []
            for idx, frame in enumerate(frames):
                frame_path = os.path.join(temp_dir, f"frame_{idx}.png")
                frame.save(frame_path)
                self.image_list.append(frame_path)
            for widget in self.tree.get_children():
                self.tree.delete(widget)
            for idx, img in enumerate(self.image_list):
                self.tree.insert("", "end", iid=idx, values=(os.path.basename(img),))
            self.current_image_index = 0
            self.load_image(self.image_list[0])
            messagebox.showinfo("Video Loaded", f"Loaded {len(self.image_list)} frames from video.")

    def ai_prelabel(self):
        from .ai_tools import ai_prelabel
        ai_prelabel(self)

    def train_custom_model(self):
        from .ai_tools import train_custom_model
        train_custom_model(self)

    def quality_check(self):
        overlaps = 0
        for i, ann1 in enumerate(self.annotations):
            if ann1.type != "bbox":
                continue
            x1, y1, x2, y2 = ann1.points
            for j, ann2 in enumerate(self.annotations):
                if i >= j or ann2.type != "bbox":
                    continue
                a1 = max(x1, ann2.points[0])
                b1 = max(y1, ann2.points[1])
                a2 = min(x2, ann2.points[2])
                b2 = min(y2, ann2.points[3])
                if a2 > a1 and b2 > b1:
                    overlaps += 1
        messagebox.showinfo("Quality Check", f"Found {overlaps} overlapping bounding boxes.")

    def split_dataset(self):
        messagebox.showinfo("Split Dataset", "Dataset split functionality is not fully implemented in this demo.")

    def manage_labels(self):
        dialog = tk.Toplevel(self)
        dialog.title("Manage Classes")
        dialog.geometry("300x300")
        listbox = tk.Listbox(dialog)
        listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        for lab in self.labels:
            listbox.insert(tk.END, lab)
        entry = ttk.Entry(dialog)
        entry.pack(padx=5, pady=5)
        def add_label():
            new_lab = entry.get().strip()
            if new_lab and new_lab not in self.labels:
                self.labels.append(new_lab)
                listbox.insert(tk.END, new_lab)
                entry.delete(0, tk.END)
                self.update_class_buttons()
        def remove_label():
            sel = listbox.curselection()
            if sel:
                lab = listbox.get(sel)
                self.labels.remove(lab)
                listbox.delete(sel)
                if self.selected_class == lab:
                    self.selected_class = self.labels[0] if self.labels else None
                self.update_class_buttons()
        ttk.Button(dialog, text="Add Class", command=add_label).pack(pady=5)
        ttk.Button(dialog, text="Remove Selected", command=remove_label).pack(pady=5)

    def test_model(self):
        from annotator.ai_tools import test_model
        test_model(self)


    def export_yolo(self):
        from .export_tools import export_yolo_format
        export_yolo_format(self)

    def export_voc(self):
        from .export_tools import export_voc_format
        export_voc_format(self)

    def export_coco(self):
        from .export_tools import export_coco_format
        export_coco_format(self)

    def export_csv(self):
        from .export_tools import export_csv_format
        export_csv_format(self)

    def show_about(self):
        messagebox.showinfo("About", "Image and Video Annotator\nVersion 1.0\nSupports YOLO and Mask R-CNN export formats.")

    def on_exit(self):
        if messagebox.askokcancel("Quit", "Do you really want to quit?"):
            self.destroy()
