import flet as ft
import pyaudio
import numpy as np
import whisper
import queue
import asyncio
from google import genai
from google.genai import types
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# MyLibrary
import components as cp

@ft.control
class voiceControlApp(ft.Container):
    def __init__(self):
        super().__init__()
        self.width=350

        # 音声認識エンジンを保持
        self.recog_engine = VoiceRecog(on_update_callback=self.update_result_ui)
        # ダジャレを採点するLLM
        self.judge_engine = LLM_Proc(on_update_callback=self.update_llm_ui)
        
        # 画面表示
        self.appTitle = cp.GenTxt("がっくんのダジャレ採点器", weight=ft.FontWeight.BOLD, size = 30)
        self.status_text = cp.GenTxt("準備完了", weight=ft.FontWeight.BOLD)
        self.result_display = cp.GenTxt("")

        # LLMの結果表示
        self.llm_result_area = ft.Column(
            visible=False,
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            width=self.width
        )
        self.dajare_text = cp.GenTxt("", size=20)
        self.eval_text = cp.GenTxt("", size=30, weight=ft.FontWeight.BOLD, color=ft.Colors.RED)
        self.reason_text = cp.GenTxt("")
        self.llm_result_area.controls = [
            self.dajare_text,
            self.eval_text,
            self.reason_text
        ]

        self.is_recoding = False
        self.btn_start = cp.StartVoiceButton(
            content="ダジャレを言う",
            icon=ft.Icons.KEYBOARD_VOICE,
            on_click=self.button_clicked,
            visible=True,
        )
        self.btn_finish = cp.FinishVoiceButton(
            content="判定する",
            icon=ft.Icons.KEYBOARD_VOICE_OUTLINED,
            on_click=self.button_clicked,
            visible=False,
        )

        # Fletページ
        self.content = ft.Column(
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[
                self.appTitle,
                self.status_text,
                self.result_display,
                self.llm_result_area,
                self.btn_start,
                self.btn_finish,
            ],
            spacing=10,
        )

    async def button_clicked(self, e):
        self.is_recoding = not self.is_recoding
        self.btn_start.visible = not self.btn_start.visible
        self.btn_finish.visible = not self.btn_finish.visible
        if self.is_recoding:
            self.llm_result_area.visible = False
            self.status_text.value = "マイク使用中..."
            self.status_text.color = ft.Colors.RED_400
            self.page.update()
            # 解析エンジンの開始
            await self.recog_engine.start()
        else:
            self.status_text.value = ""
            # 解析エンジンの終了
            self.recog_engine.stop()
            self.status_text.value = "採点中..."
            self.page.update()

            # 音声認識結果をLLMに渡す
            last_text = self.recog_engine.transciption
            await self.judge_engine.start(last_text)

    async def update_result_ui(self, text):
        self.result_display.value = f"認識中: {text}"

        if self.page:
            self.page.update()

    async def update_llm_ui(self, response):
        self.result_display.visible = False
        if response is None:
            self.status_text.value = "サーバが混雑しています．もう一度試してね"
            self.page.update()
            return
        res_data = response.parsed

        self.dajare_text.value = f"「{res_data.dajare}」の判定結果"
        self.eval_text.value = res_data.evaluation
        self.reason_text.value = res_data.reason

        self.status_text.value = "結果発表"
        self.llm_result_area.visible = True
        if self.page:
            self.page.update()
        

class VoiceRecog():
    CHUNK = 1024
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 16000

    def __init__(self, on_update_callback):
        # PyAudioインスタンスを設定
        self.p = pyaudio.PyAudio()
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

        # 以下でプロジェクト特有の変数・関数を設定
        # 認識結果をUIに渡すためのコールバック関数
        self.on_update_callback = on_update_callback  

    def mic_check(self):
        try:
            self.default_info = self.p.get_default_input_device_info()
            print("[Dev] --- Microphone Info ---")
            print(f"Input mic: {self.default_info['name']}")
            print(f"Index: {self.default_info['index']}")
            print(f"Sampling Rate: {self.default_info['defaultSampleRate']}")
        except IOError:
            print("使用可能なマイクが見つかりません。")

    async def start(self):
        if self.is_running:
            return
        
        # 前回のデータが残っていたら空にする
        while not self.audio_queue.empty():
            self.audio_queue.get()
        
        self.is_running = True
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            stream_callback=self.audio_callback
        )

        # 音声処理
        asyncio.create_task(self.proc_audio())

    def audio_callback(self, in_data, frame_count, time_info, status):
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.audio_queue.put(audio_data)
        return (in_data, pyaudio.paContinue)
    
    async def proc_audio(self):
        accumulated_audio = np.array([], dtype=np.float32)

        # しきい値の設定
        SILENCE_THRESHOLD = 0.01
        SILENCE_DURATION = 0.1
        # 無音時間計測変数
        silence_passtime = 0

        while self.is_running:
            try:
                # キューからデータを取得
                data = self.audio_queue.get_nowait()

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
                    
                    if result["text"]:
                        self.transciption = str(result["text"]).strip()
                        print(f"[Dev:transcribe] {self.transciption}")
                        await self.result(self.transciption)

                    # バッファリセット
                    accumulated_audio = np.array([], dtype=np.float32)

            except queue.Empty:
                await asyncio.sleep(0.1)

    async def result(self, text):
        # 認識結果を用いた処理をここで行う
        if self.on_update_callback:
            await self.on_update_callback(text)

    def stop(self):
        self.is_running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        print(f"[Dev] Recording ended.")

    def __del__(self):
        try:
            self.p.terminate()
        except:
            pass

class LLM_Proc:
    class DajareEval(BaseModel):
        dajare: str
        evaluation: str
        reason: str

    def __init__(self, on_update_callback):
        # APIキーの設定
        self.dotenv_path = ".env"
        load_dotenv(self.dotenv_path)
        API_KEY = os.getenv('GEMINI_API_KEY')

        # Gemini APIの設定
        self.client = genai.Client(api_key=API_KEY)
        self.on_update_callback = on_update_callback

        self.instruction = f"""
            # Identity
            あなたはユーザが入力したダジャレが面白かったかどうかを判定するAIです．
          
            # Instructions
            ユーザが発言した内容が音声認識で解析され変数に格納されているが，解析の精度ゆえ，
            そのままの文字列では駄洒落になっていないことがある．
            例えば「(発言)お金はおっかねー」→「(文字列)お金はお金」，「(発言)布団がふっとんだ」→「(文字列)布団が布団だ」等．
            ユーザが本当に言いたかったダジャレを推測し，変数「dajare」として出力せよ．
            次に「dajare」のダジャレが面白いかどうかを「Evaluate」の評価方法に従って評価し，評価結果を変数「evaluate」として出力せよ．
            最後に評価の理由を端的に説明し，「reason」として出力せよ．
            なお，評価は辛口気味にするが，毎回「がっくし」ではなく，バラエティに富んだ評価をせよ．

            # Evaluate
            評価は「樂学士」「がっくん」「がっくし」の3段階評価とし，「樂学士」が一番面白く，
            「がっくし」が一番つまらない評価である．

            # Output format
            各結果の例をExampleに示す．dajare, evaluation, reasonをすべてJSON形式で出力せよ．

            # Example
            voice_inputが「布団が布団だ」のとき:
            dajare: 「布団がふっとんだ」
            evaluation: 「がっくし」
            reason: 「ハッキリ言ってつまらない．どのような生活をして何を食べていればそのようなつまらないダジャレを思い浮かぶのか分からない．
                    その無い思考力をもっと他のことに使えばいいのに」
        """

    async def start(self, voice_text):
        if not voice_text:
            return
        await self.evaluate(voice_text)

    # 503エラーなどの時に、最大3回まで、待ち時間を増やしながらリトライする
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def evaluate(self, voice_input):
        response = await self.client.aio.models.generate_content(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                system_instruction=self.instruction,
                response_mime_type="application/json",
                response_schema= self.DajareEval,
            ),
            contents=voice_input,
        )
        await self.result(response)

    async def result(self, response):
        print(f"[Dev:response.parsed]{response.parsed}")
        # 認識結果を用いた処理をここで行う
        if self.on_update_callback:
            await self.on_update_callback(response)
