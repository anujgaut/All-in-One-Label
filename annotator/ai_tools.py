# ai_tools.py
import os
import glob
import cv2
from tkinter import Canvas, messagebox, filedialog
from PIL import Image, ImageDraw, ImageFont
import os
import glob
import cv2
from tkinter import Toplevel, Frame, Listbox, Scrollbar, Button, Label, messagebox, filedialog, BOTH, LEFT, RIGHT, Y, END, VERTICAL
from PIL import Image, ImageTk, ImageDraw, ImageFont

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

def ai_prelabel(app):
    """Run AI-assisted pre-labeling using a YOLOv11 model."""
    if YOLO is None:
        messagebox.showerror("Error", "Ultralytics package not installed!\nPlease run: pip install ultralytics")
        return

    if not hasattr(app, 'ai_model') or app.ai_model is None:
        app.ai_model = YOLO("yolo11s.pt")

    img = cv2.imread(app.image_path)
    if img is None:
        messagebox.showerror("Error", "Cannot load image for AI pre-labeling!")
        return

    results = app.ai_model(img)
    from annotator.models import Annotation  # Local import to avoid circular dependency

    for result in results:
        for box in result.boxes:
            coords = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = coords
            cls_id = int(box.cls[0])
            if isinstance(app.ai_model.names, dict):
                label = app.ai_model.names.get(cls_id, "object")
            else:
                label = app.ai_model.names[cls_id]
            if label not in app.labels:
                app.labels.append(label)
                app.update_class_buttons()
            ann = Annotation("bbox", [int(x1), int(y1), int(x2), int(y2)], label)
            app.annotations.append(ann)
    app.push_undo_state()
    app.redraw_canvas()
    messagebox.showinfo("AI Pre-label", "AI pre-labeling complete.")

def train_custom_model(app):
    """Initiate custom YOLOv11 training using a provided dataset.yaml and user-specified hyperparameters.
       Training progress will be shown in a dedicated UI window."""
    from tkinter import filedialog, simpledialog, Toplevel, Text, Scrollbar, BOTH, END, Button
    import threading, queue, sys

    if YOLO is None:
        messagebox.showerror("Error", "Ultralytics package not installed!\nPlease run: pip install ultralytics")
        return

    # Select the dataset YAML file.
    dataset_file = filedialog.askopenfilename(
        title="Select dataset.yaml for training",
        filetypes=[("YAML files", "*.yaml")]
    )
    if not dataset_file:
        return

    # Ask the user for various hyperparameters.
    epochs = simpledialog.askinteger("Training", "Enter number of training epochs", initialvalue=50, minvalue=1, maxvalue=500)
    if epochs is None:
        return

    batch_size = simpledialog.askinteger("Training", "Enter batch size", initialvalue=16, minvalue=1, maxvalue=128)
    if batch_size is None:
        return

    image_size = simpledialog.askinteger("Training", "Enter image size (imgsz)", initialvalue=640, minvalue=128, maxvalue=2048)
    if image_size is None:
        return

    num_workers = simpledialog.askinteger("Training", "Enter number of workers", initialvalue=4, minvalue=1, maxvalue=16)
    if num_workers is None:
        return

    learning_rate = simpledialog.askfloat("Training", "Enter learning rate", initialvalue=0.01, minvalue=1e-6, maxvalue=1.0)
    if learning_rate is None:
        return

    momentum = simpledialog.askfloat("Training", "Enter momentum", initialvalue=0.937, minvalue=0.0, maxvalue=1.0)
    if momentum is None:
        return

    weight_decay = simpledialog.askfloat("Training", "Enter weight decay", initialvalue=0.0005, minvalue=0.0, maxvalue=0.1)
    if weight_decay is None:
        return

    # Create a new window to show training progress.
    progress_window = Toplevel(app)
    progress_window.title("Training Progress")
    progress_window.geometry("600x400")

    text_widget = Text(progress_window, wrap="word")
    text_widget.pack(fill=BOTH, expand=True)

    scrollbar = Scrollbar(text_widget)
    scrollbar.pack(side="right", fill="y")
    text_widget.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=text_widget.yview)

    # Create a thread-safe queue for log messages.
    log_queue = queue.Queue()

    # Define a custom stream that writes log messages to the queue.
    class QueueStream:
        def __init__(self, q):
            self.q = q

        def write(self, msg):
            self.q.put(msg)

        def flush(self):
            pass

    # Save the original stdout and stderr.
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Define the training function that will run in a separate thread.
    def training_thread():
        # Redirect stdout and stderr to our queue stream.
        sys.stdout = QueueStream(log_queue)
        sys.stderr = QueueStream(log_queue)
        try:
            model = YOLO("yolo11s.yaml")
            model.train(
                data=dataset_file,
                epochs=epochs,
                imgsz=image_size,
                batch=batch_size,
                workers=num_workers,
                lr0=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay
            )
        except Exception as e:
            print("Error during training:", e)
        finally:
            # Restore the original stdout and stderr.
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            print("Training completed.")

    # Start the training in a new thread.
    t = threading.Thread(target=training_thread)
    t.start()

    # Define a function that polls the log queue and updates the text widget.
    def update_log():
        try:
            while True:
                line = log_queue.get_nowait()
                text_widget.insert(END, line)
                text_widget.see(END)
        except queue.Empty:
            pass
        # If the thread is still running, poll again after 100ms.
        if t.is_alive():
            progress_window.after(100, update_log)
        else:
            # Final update after thread completion.
            progress_window.after(100, update_log)
            # Optionally, add a button to allow the user to close the progress window.
            Button(progress_window, text="Close", command=progress_window.destroy).pack()

    # Start polling the log queue.
    update_log()


def test_model_ui(app):
    """
    Opens a new window that allows the user to select a trained model file
    and a test images folder. The left pane shows a list of image files; when
    an image is selected and the user clicks "Predict", the model predictions are
    overlaid on that image and displayed in the same window.
    """
    if YOLO is None:
        messagebox.showerror("Error", "Ultralytics package not installed!\nPlease run: pip install ultralytics")
        return

    # Ask the user to select the trained model file.
    model_file = filedialog.askopenfilename(
        title="Select Trained Model File",
        filetypes=[("PyTorch Model Files", "*.pt"), ("All Files", "*.*")]
    )
    if not model_file:
        return

    try:
        test_model = YOLO(model_file)
    except Exception as e:
        messagebox.showerror("Error", f"Could not load the model:\n{e}")
        return

    # Ask the user to select a folder with test images.
    test_folder = filedialog.askdirectory(title="Select Folder with Test Images")
    if not test_folder:
        return

    # Gather image files in the folder.
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp")
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(test_folder, ext)))
    image_files.sort()
    if not image_files:
        messagebox.showerror("Error", "No image files found in the selected folder!")
        return

    # Create a new Toplevel window.
    win = Toplevel(app)
    win.title("Test Model - Interactive Interface")
    win.geometry("1000x600")

    # Create left frame for the image list.
    left_frame = Frame(win, width=300)
    left_frame.pack(side=LEFT, fill=Y, padx=5, pady=5)

    Label(left_frame, text="Test Images").pack()

    scrollbar = Scrollbar(left_frame, orient=VERTICAL)
    listbox = Listbox(left_frame, yscrollcommand=scrollbar.set, width=40)
    scrollbar.config(command=listbox.yview)
    scrollbar.pack(side=RIGHT, fill=Y)
    listbox.pack(side=LEFT, fill=Y, expand=True)

    # Populate the listbox with image filenames.
    for img_path in image_files:
        listbox.insert(END, os.path.basename(img_path))

    # Create right frame for the image preview.
    right_frame = Frame(win)
    right_frame.pack(side=RIGHT, fill=BOTH, expand=True, padx=5, pady=5)
    canvas = app.canvas  # Reuse the main app's canvas, or create a new one if preferred.
    # Here we create a new canvas inside the Toplevel:
    canvas = app.__class__(win).canvas  # Alternative: create a fresh canvas; for clarity we create one below.
    canvas = Canvas(right_frame, bg="black")
    canvas.pack(fill=BOTH, expand=True)

    # We'll store the currently loaded image (PIL object) and its PhotoImage.
    win.current_image = None
    win.current_photo = None
    win.current_image_path = None

    # Define a helper function to display an image on the canvas.
    def display_image(pil_img):
        # Resize image if necessary (for example, to fit the canvas)
        # Here we simply convert to PhotoImage and display.
        win.current_photo = ImageTk.PhotoImage(pil_img)
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=win.current_photo)

    # Define a function to load the selected image.
    def load_selected_image(event=None):
        selection = listbox.curselection()
        if not selection:
            return
        index = selection[0]
        img_path = image_files[index]
        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image:\n{e}")
            return
        win.current_image = pil_img.copy()  # store original
        win.current_image_path = img_path
        display_image(pil_img)

    # Bind selection event on the listbox.
    listbox.bind("<<ListboxSelect>>", load_selected_image)

    # Create a Predict button at the bottom.
    def run_prediction():
        if win.current_image is None:
            messagebox.showerror("Error", "Please select an image first.")
            return

        # Run inference on the current image using the test_model.
        try:
            results = test_model.predict(source=win.current_image_path, save=False, verbose=False)
        except Exception as e:
            messagebox.showerror("Error", f"Error during inference:\n{e}")
            return

        # Create a copy of the current image to draw predictions.
        pred_img = win.current_image.copy()
        draw = ImageDraw.Draw(pred_img)

        # Optional: set a font.
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except Exception:
            font = None

        # Process predictions and draw boxes and labels.
        for result in results:
            for box in result.boxes:
                coords = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = coords
                cls_id = int(box.cls[0])
                if isinstance(test_model.names, dict):
                    label = test_model.names.get(cls_id, "object")
                else:
                    label = test_model.names[cls_id]
                # Draw a blue rectangle and label.
                draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)
                draw.text((x1 + 5, y1 + 5), label, fill="blue", font=font)

        # Update the canvas with the prediction image.
        display_image(pred_img)
        messagebox.showinfo("Test Model", "Prediction complete for the selected image.")

    predict_btn = Button(win, text="Predict", command=run_prediction)
    predict_btn.pack(side="bottom", pady=10)

    # Optionally, load the first image immediately.
    listbox.select_set(0)
    load_selected_image()

    # Keep the window open.
    win.mainloop()