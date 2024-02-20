import os
import wave
import time
import threading
import tkinter as tk
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import firwin, lfilter, bartlett

class VoiceRecorder:
    def __init__(self):
        self.root = tk.Tk()
        self.root.resizable(False, False)
        self.root.geometry("700x700")

        self.record_label = tk.Label(text="Mulai Merekam", font=("Roboto", 16, "bold"))
        self.record_label.pack()

        self.button = tk.Button(text="⚫", font=("Roboto", 80, "bold"), command=self.click_handler)
        self.button.pack()

        self.filter_button = tk.Button(text="Nonaktifkan Filter", font=("Roboto", 12), command=self.toggle_filter)
        self.filter_button.pack()

        self.recording = False
        self.filter_enabled = True
        self.label = tk.Label(text="00:00:00")
        self.label.pack()

        self.figure_waveform = plt.Figure(figsize=(5, 3))
        self.plot_waveform = self.figure_waveform.add_subplot(111)
        self.canvas_waveform_agg = FigureCanvasTkAgg(self.figure_waveform, master=self.root)
        self.canvas_waveform_agg.get_tk_widget().pack(fill="both", expand=True)

        self.root.protocol("WM_DELETE_WINDOW", self.close_program)

        self.sampling_rate = 44100
        self.lowcut = 1000
        self.highcut = 2500
        self.order = 17

        self.root.mainloop()

    def click_handler(self):
        if self.recording:
            self.recording = False
            self.button.config(fg="black")
        else:
            self.recording = True
            self.button.config(fg="red")
            threading.Thread(target=self.record).start()

    def toggle_filter(self):
        self.filter_enabled = not self.filter_enabled
        if self.filter_enabled:
            self.filter_button.config(text="Nonaktifkan Filter", fg="red")
        else:
            self.filter_button.config(text="Aktifkan Filter")

    def record(self):
        self.p = pyaudio.PyAudio()
        stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=self.sampling_rate, input=True,
                             frames_per_buffer=1024)
        frames = []
        start = time.time()
        while self.recording:
            data = stream.read(1024)
            frames.append(data)

            passed = time.time() - start
            secs = passed % 60
            mins = passed // 60
            hours = mins // 60
            self.label.config(text=f"{int(hours):02d}:{int(mins):02d}:{int(secs):02d}")

            if len(frames) % 10 == 0:
                self.root.after(0, self.plot_waveform_realtime, frames)

        stream.stop_stream()
        stream.close()
        self.p.terminate()

        exists = True
        i = 1
        while exists:
            if os.path.exists(f"recording_filter{i}.wav"):
                i += 1
            else:
                exists = False

        sound_file = wave.open(f"recording_filter{i}.wav", "wb")
        sound_file.setnchannels(1)
        sound_file.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
        sound_file.setframerate(self.sampling_rate)
        sound_file.writeframes(b"".join(frames))
        sound_file.close()

        self.root.after(0, self.plot_waveform_realtime, frames)
        self.root.after(0, self.plot_fft, f"recording_filter{i}.wav")

    def plot_waveform_realtime(self, frames):
        signal = np.frombuffer(b"".join(frames), dtype=np.int16)
        windowed_signal = self.apply_bartlett_window(signal)
        if self.filter_enabled:
            filtered_signal = self.bandstop_filter(windowed_signal)
        else:
            filtered_signal = windowed_signal

        time_per_sample = 1 / self.sampling_rate
        total_time = len(signal) * time_per_sample
        time_axis = np.arange(0, total_time, time_per_sample)[:len(filtered_signal)]

        self.plot_waveform.clear()
        self.plot_waveform.plot(time_axis, filtered_signal, color='b')
        self.plot_waveform.set_title("Sinyal Domain Waktu", fontsize=14, fontweight='bold')
        self.plot_waveform.set_xlabel("Waktu (detik)", fontsize=12)
        self.plot_waveform.set_ylabel("Amplitudo", fontsize=12)
        self.canvas_waveform_agg.draw()

    def bandstop_filter(self, signal):
        nyquist_freq = 0.5 * self.sampling_rate
        low = self.lowcut / nyquist_freq
        high = self.highcut / nyquist_freq
        taps = firwin(self.order, [low, high], pass_zero=False)
        filtered_signal = lfilter(taps, 1.0, signal)
        return filtered_signal

    def apply_bartlett_window(self, signal):
        windowed_signal = signal * bartlett(len(signal))
        return windowed_signal

    def plot_fft(self, filename):
        sound_file = wave.open(filename, "rb")
        frames = sound_file.readframes(-1)
        signal = np.frombuffer(frames, dtype=np.int16)
        filtered_signal = self.bandstop_filter(signal)
        duration = sound_file.getnframes() / sound_file.getframerate()
        sound_file.close()

        fs = len(signal) / duration

        plt.figure(figsize=(5, 3))
        n = len(filtered_signal)
        freq_range = np.arange(0, fs, 1000)
        freq_spectrum = np.abs(np.fft.fft(filtered_signal)[:n//2])
        freq_spectrum = freq_spectrum[:len(freq_range)]
        plt.plot(freq_range, freq_spectrum, color='b')
        plt.title("Spectrum Frekuensi (FFT)")
        plt.xlabel("Frekuensi (Hz)")
        plt.ylabel("Magnitude")
        plt.tight_layout()
        plt.show()

    def close_program(self):
        self.recording = False
        if self.p:
            self.p.terminate()
        self.root.destroy()

VoiceRecorder()
