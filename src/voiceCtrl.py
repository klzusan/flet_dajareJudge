import flet as ft
import pyaudio
import numpy as np
import whisper
import queue
import threading

# MyLibrary
import components as cp

@ft.control
class voiceControlApp(ft.Container):
    def __init__(self):
        super().__init__()

        # 音声認識エンジンを保持
        self.recog_engine = VoiceRecog(on_update_callback=self.update_result_ui)
        
        self.status_text = ft.Text("待機中...", color=ft.Colors.GREY_600)
        self.result_display = ft.Text("ここに（ry")

        # Fletページ
        self.content = ft.Column(
            controls=[
                self.status_text,
                self.result_display,
                cp.VoiceRecogButton(
                    content="Push to talk",
                    icon=ft.Icons.KEYBOARD_VOICE,
                    on_click=self.start_button_clicked,
                ),
            ]
        )

    def start_button_clicked(self, e):
        self.status_text.value = "マイク使用中..."
        self.status_text.color = ft.Colors.RED_400
        self.update()
        # 解析エンジンの開始
        self.recog_engine.start()

    def update_result_ui(self, text):
        self.result_display.value = text
        self.update()

class VoiceRecog:
    CHUNK = 1024
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 16000

    def __init__(self, on_update_callback):
        # 認識結果をUIに渡すためのコールバック関数
        self.on_update_callback = on_update_callback
        # マイクチェック（デバッグ）
        self.mic_check()
        # Whisperモデルのロード
        self.whisper_model = whisper.load_model("small")
        # 音声データを格納するキュー
        self.audio_queue = queue.Queue()
        # 音声認識の結果を格納する変数
        self.transciption = ""
        # 音声入力のセットアップ
        self.is_running = False
        self.stream = None
        self.p = pyaudio.PyAudio()

    def mic_check(self):
        self.p = pyaudio.PyAudio()
        try:
            self.default_info = self.p.get_default_input_device_info()
            print("[Dev] --- Microphone Info ---")
            print(f"Input mic: {self.default_info['name']}")
            print(f"Index: {self.default_info['index']}")
            print(f"Sampling Rate: {self.default_info['defaultSampleRate']}")
        except IOError:
            print("使用可能なマイクが見つかりません。")

        self.p.terminate()

    def start(self):
        if self.is_running:
            return
        
        self.is_running = True
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            stream_callback=self.audio_callback
        )

        # 音声処理スレッドの開始
        self.processing_thread = threading.Thread(target=self.proc_audio)
        self.processing_thread.start()

    def audio_callback(self, in_data, frame_count, time_info, status):
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.audio_queue.put(audio_data)
        return (in_data, pyaudio.paContinue)
    
    def proc_audio(self):
        accumulated_audio = np.array([], dtype=np.float32)

        # しきい値の設定
        SILENCE_THRESHOLD = 0.01
        SILENCE_DURATION = 0.5
        # 無音時間計測変数
        silence_passtime = 0

        while self.is_running:
            # print(f"[Dev:silence_passtime] {silence_passtime}")
            try:
                # キューからデータを取得
                data = self.audio_queue.get(timeout=1.0)

                # データの音量を計算
                rms = np.sqrt(np.mean(data**2))
                if rms < SILENCE_THRESHOLD:
                    # 無音のとき
                    if silence_passtime < SILENCE_DURATION:
                        silence_passtime += (len(data) / self.RATE)
                else:
                    # 音があるとき
                    silence_passtime = 0
                    accumulated_audio = np.append(accumulated_audio, data)

                # 解析
                if silence_passtime >= SILENCE_DURATION and len(accumulated_audio) > 100:
                    result = self.whisper_model.transcribe(accumulated_audio, language='ja')
                    text = str(result["text"]).strip()
                    print(f"[Dev:transcribe] {text}")

                    if text:
                        # UI側の更新関数を呼び出す
                        self.on_update_callback(text)

                    # バッファリセット
                    accumulated_audio = np.array([], dtype=np.float32)

            except queue.Empty:
                continue

    def stop(self):
        self.is_running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()