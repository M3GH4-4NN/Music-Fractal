import wx
import numpy as np
import matplotlib
matplotlib.use("WXAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import librosa

# 1. FRACTAL LAUNCHER WINDOW

class FractalLauncherFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title="Fractal Selector", size=(700, 400))

        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)

        title = wx.StaticText(panel, label="Choose a Fractal:")
        font = title.GetFont()
        font.PointSize += 4
        font.MakeBold()
        title.SetFont(font)
        sizer.Add(title, 0, wx.ALL | wx.CENTER, 15)
        hbox = wx.BoxSizer(wx.HORIZONTAL)

        # Tree Button 
        tree_img = wx.Image(r"C:\Users\jinto\Desktop\Megha\JackFruit\Python\tree.png", wx.BITMAP_TYPE_ANY).Scale(180, 180)
        self.tree_btn = wx.StaticBitmap(panel, bitmap=wx.Bitmap(tree_img))
        self.tree_btn.Bind(wx.EVT_LEFT_DOWN, self.open_tree)
        

        #Duffing Button 
        duff_img = wx.Image(r"C:\Users\jinto\Desktop\Megha\JackFruit\Python\duffing.png", wx.BITMAP_TYPE_ANY).Scale(180, 180)
        self.duff_btn = wx.StaticBitmap(panel, bitmap=wx.Bitmap(duff_img))
        self.duff_btn.Bind(wx.EVT_LEFT_DOWN, self.open_duffing)
        hbox.Add(self.tree_btn, 0, wx.ALL, 15)
        hbox.Add(self.duff_btn, 0, wx.ALL, 15)

        sizer.Add(hbox, 0, wx.CENTER)
        panel.SetSizer(sizer)
        self.Centre()
        self.Show()

    def open_tree(self, event):
        frame = TreeFractalFrame()
        frame.Show()

    def open_duffing(self, event):
        frame = DuffingFrame()
        frame.Show()

#feature extraction for tree fractal
def load_audio_features_basic(path):
    y, sr = librosa.load(path, sr=None)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]

    centroid /= max(centroid.max(), 1e-8)
    bandwidth /= max(bandwidth.max(), 1e-8)

    return y, sr, centroid, bandwidth

#feature extraction for Duffing fractal
def load_audio_features_duffing(path):
    y, sr = librosa.load(path, sr=None)

    S = np.abs(librosa.stft(y))
    rms = librosa.feature.rms(y=y)[0]

    pitch_mat, _ = librosa.piptrack(y=y, sr=sr)
    pitch = pitch_mat.max(axis=0)
    pitch[pitch == 0] = np.nan
    pitch = np.nan_to_num(pitch, nan=np.nanmean(pitch))

    L = min(S.shape[1], len(rms), len(pitch))
    amplitude = np.mean(S[:, :L], axis=0)
    pitch = pitch[:L]
    rms = rms[:L]

    amplitude = amplitude ** 1.4
    pitch = pitch ** 1.1

    threshold = 0.03 * amplitude.max()
    valid = amplitude > threshold

    amplitude = amplitude[valid]
    pitch = pitch[valid]
    rms = rms[valid]

    if len(amplitude) < 10:
        raise ValueError("Audio too silent to visualize")

    gamma = np.interp(amplitude, (amplitude.min(), amplitude.max()), (0.5, 5.0))
    omega = np.interp(pitch, (pitch.min(), pitch.max()), (1.0, 15.0))
    colors = np.interp(rms, (rms.min(), rms.max()), (0, 1))

    return gamma, omega, colors


#tree fractal
ALL_SEGMENTS = []

def store_segment(x1, y1, x2, y2, color, lw, alpha):
    ALL_SEGMENTS.append((x1, y1, x2, y2, color, lw, alpha))

def draw_tree(x, y, angle, scale,
              lengths, angles, colors,
              depth, target_depth, max_depth):
    if depth > target_depth or depth >= max_depth:
        return

    idx = depth % len(lengths)

    if depth == 0:
        L = 1.2
        final_angle = np.pi / 2
    else:
        if depth == target_depth:
            L = lengths[idx] * scale * 2.0
            jitter = np.random.uniform(-0.03, 0.03)
            final_angle = angle + angles[idx] + jitter
        else:
            L = lengths[0] * scale
            final_angle = angle

    new_x = x + L * np.cos(final_angle)
    new_y = y + L * np.sin(final_angle)

    color = plt.cm.hsv((colors[idx] + depth * 0.05) % 1.0)
    alpha = 0.85 * (1 - depth / max_depth)
    lw = max(0.6, 4 * (1 - depth / max_depth))

    store_segment(x, y, new_x, new_y, color, lw, alpha)

    for ba in (final_angle + np.pi / 6,
               final_angle - np.pi / 6,
               final_angle):
        draw_tree(new_x, new_y, ba, scale * 0.7,
                  lengths, angles, colors,
                  depth + 1, target_depth, max_depth)


class TreeFractalFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title="Tree Fractal", size=(950, 800))

        self.audio_path = None
        self.centroid = None
        self.bandwidth = None

        self.chunk_size = 50
        self.frame_count = 0
        self.anim = None

        self.build_ui()
        self.Centre()
        self.Bind(wx.EVT_CLOSE, self.on_close)
        self.Show()

    def build_ui(self):
        panel = wx.Panel(self)
        main = wx.BoxSizer(wx.VERTICAL)

        self.file_btn = wx.Button(panel, label="Select Audio File")
        self.file_label = wx.StaticText(panel, label="No file selected")
        self.file_btn.Bind(wx.EVT_BUTTON, self.on_select_file)

        main.Add(self.file_btn, 0, wx.EXPAND | wx.ALL, 5)
        main.Add(self.file_label, 0, wx.ALL, 5)

        main.Add(wx.StaticText(panel, label="Amplitude Sensitivity"), 0, wx.LEFT)
        self.amp_slider = wx.Slider(panel, value=50, minValue=10, maxValue=200)
        main.Add(self.amp_slider, 0, wx.EXPAND | wx.ALL, 5)

        main.Add(wx.StaticText(panel, label="Pitch Sensitivity"), 0, wx.LEFT)
        self.pitch_slider = wx.Slider(panel, value=50, minValue=10, maxValue=200)
        main.Add(self.pitch_slider, 0, wx.EXPAND | wx.ALL, 5)

        main.Add(wx.StaticText(panel, label="Branch Depth (Recursion Level)"), 0, wx.LEFT)
        self.depth_slider = wx.Slider(panel, value=10, minValue=2, maxValue=12)
        main.Add(self.depth_slider, 0, wx.EXPAND | wx.ALL, 5)

        self.start_btn = wx.Button(panel, label="Start Visualization")
        self.start_btn.Bind(wx.EVT_BUTTON, self.on_start)
        main.Add(self.start_btn, 0, wx.EXPAND | wx.ALL, 10)

        self.fig = plt.Figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111)
        self.ax.axis("off")

        self.canvas = FigureCanvas(panel, -1, self.fig)
        main.Add(self.canvas, 1, wx.EXPAND | wx.ALL, 5)

        panel.SetSizer(main)

    def on_select_file(self, event):
        with wx.FileDialog(self, "Open audio file",
                           wildcard="Audio (*.wav;*.mp3)|*.wav;*.mp3",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dlg:
            if dlg.ShowModal() == wx.ID_CANCEL:
                return
            self.audio_path = dlg.GetPath()
            self.file_label.SetLabel(self.audio_path)

    def on_start(self, event):
        global ALL_SEGMENTS
        ALL_SEGMENTS = []

        if not self.audio_path:
            wx.MessageBox("Select an audio file first", "Error")
            return

        _, _, self.centroid, self.bandwidth = load_audio_features_basic(self.audio_path)

        self.frame_count = len(self.centroid) // self.chunk_size
        if self.frame_count <= 0:
            wx.MessageBox("Audio too short", "Error")
            return

        self.ax.cla()
        self.ax.axis("off")

        if self.anim:
            self.anim.event_source.stop()

        self.anim = FuncAnimation(self.fig,
                                  self.update_frame,
                                  frames=self.frame_count,
                                  interval=80,
                                  blit=False)
        self.canvas.draw()

    def update_frame(self, frame):
        #Clear the global segment list so we don't draw duplicates from previous frames
        global ALL_SEGMENTS
        ALL_SEGMENTS = [] 

        amp_factor = self.amp_slider.GetValue() / 50
        pitch_factor = self.pitch_slider.GetValue() / 50
        current_max_depth = self.depth_slider.GetValue()

        self.ax.cla()
        self.ax.axis("off")

        #Calculate params for THIS frame
        depth_now = min(frame, current_max_depth - 1)
        i = frame * self.chunk_size
        j = i + self.chunk_size

        chunk_c = self.centroid[i:j]
        chunk_bw = self.bandwidth[i:j]

        if len(chunk_c) == 0:
            return

        lengths = np.full(len(chunk_c), (1 + 0.6 * frame / self.frame_count) * amp_factor)
        angles = (chunk_c - 0.5) * (np.pi / 5) * pitch_factor
        colors = chunk_bw

        #Generate the NEW tree structure (this fills ALL_SEGMENTS)
        draw_tree(0, -2.5, np.pi / 2, 1.0,
                  lengths, angles, colors,
                  0, depth_now, current_max_depth)

        #plots the segments we just generated
        if ALL_SEGMENTS:
            # Unzipping for faster plotting is often cleaner, but iterating is fine for this scale
            for seg in ALL_SEGMENTS:
                x1, y1, x2, y2, c, lw, a = seg
                self.ax.plot([x1, x2], [y1, y2], color=c, lw=lw, alpha=a)

        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-3, 2)
        self.canvas.draw()
        
    def on_close(self, event):
        if getattr(self, "anim", None):
            try:
                self.anim.event_source.stop()
            except:
                pass
            self.anim = None
        event.Skip()



#Duffing Fractal 

def duffing(x, v, t, gamma, omega, delta=0.2, alpha=-1.0, beta=1.0):
    dxdt = v
    dvdt = -delta * v - alpha * x - beta * x ** 3 + gamma * np.cos(omega * t)
    return dxdt, dvdt


def integrate_duffing(gamma, omega, steps=3000, dt=0.005):
    x, v = 0.1, 0.0
    traj = []
    for i in range(steps):
        dxdt, dvdt = duffing(x, v, i * dt, gamma, omega)
        x += dxdt * dt
        v += dvdt * dt
        traj.append((x, v))
    return np.array(traj)


class DuffingFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title="Duffing Fractal", size=(1000, 800))

        self.filename = None
        self.trajectories = []
        self.colors = []
        self.anim = None

        self.build_ui()
        self.Bind(wx.EVT_CLOSE, self.on_close)
        self.Show()


    def build_ui(self):
        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        self.file_btn = wx.Button(panel, label="Select Audio File")
        self.file_btn.Bind(wx.EVT_BUTTON, self.on_load_file)
        vbox.Add(self.file_btn, 0, wx.EXPAND | wx.ALL, 5)

        self.file_label = wx.StaticText(panel, label="No file selected")
        vbox.Add(self.file_label, 0, wx.ALL, 5)

        self.amp_scale = wx.Slider(panel, minValue=1, maxValue=50, value=20)
        vbox.Add(wx.StaticText(panel, label="Amplitude Sensitivity"), 0, wx.LEFT, 5)
        vbox.Add(self.amp_scale, 0, wx.EXPAND | wx.ALL, 5)

        self.pitch_scale = wx.Slider(panel, minValue=1, maxValue=100, value=50)
        vbox.Add(wx.StaticText(panel, label="Pitch Sensitivity"), 0, wx.LEFT, 5)
        vbox.Add(self.pitch_scale, 0, wx.EXPAND | wx.ALL, 5)

        start_btn = wx.Button(panel, label="Start Visualization")
        start_btn.Bind(wx.EVT_BUTTON, self.start_animation)
        vbox.Add(start_btn, 0, wx.EXPAND | wx.ALL, 10)

        self.fig = matplotlib.figure.Figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.line, = self.ax.plot([], [], lw=2)

        self.canvas = FigureCanvas(panel, -1, self.fig)
        vbox.Add(self.canvas, 1, wx.EXPAND | wx.ALL, 5)

        panel.SetSizer(vbox)

    def on_load_file(self, event):
        with wx.FileDialog(self, "Open audio file",
                           wildcard="Audio (*.mp3;*.wav)|*.mp3;*.wav",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dlg:
            if dlg.ShowModal() == wx.ID_CANCEL:
                return
            self.filename = dlg.GetPath()
            self.file_label.SetLabel(f"Loaded: {self.filename}")
    def start_animation(self, event):
        if not self.filename:
            wx.MessageBox("Select an audio file first", "Error")
            return

        try:
            self.raw_gamma, self.raw_omega, self.color_vals = load_audio_features_duffing(self.filename)
        except ValueError:
            self.file_label.SetLabel("Audio too silent")
            return

        self.frame_count = min(120, len(self.raw_gamma))

        def init():
            self.line.set_data([], [])
            return self.line,

        def animate(i):
            amp_factor = self.amp_scale.GetValue() / 20
            pitch_factor = self.pitch_scale.GetValue() / 50

            g = self.raw_gamma[i] * amp_factor
            w = self.raw_omega[i] * pitch_factor

            traj = integrate_duffing(g, w)

            self.line.set_data(traj[:, 0], traj[:, 1])
            self.line.set_color(plt.cm.plasma(self.color_vals[i]))

            return self.line,

        if self.anim:
            self.anim.event_source.stop()

        self.anim = animation.FuncAnimation(
            self.fig,
            animate,
            init_func=init,
            frames=self.frame_count,
            interval=150,
            blit=False
        )

        self.canvas.draw()

    def on_close(self, event):
        if getattr(self, "anim", None):
            try:
                self.anim.event_source.stop()
            except:
                pass
            self.anim = None
        event.Skip()



#Main

class MyApp(wx.App):
    def OnInit(self):
        self.frame=FractalLauncherFrame()
        return True

if __name__ == "__main__":
    app = MyApp(False)
    app.MainLoop()