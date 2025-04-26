import cv2
import flet as ft
import base64
import numpy as np
from multiprocessing import Process, Queue, Event
from threading import Thread
import json
import random
import time
import os
from openai import OpenAI
from fer import FER
import math

os.environ["OPENAI_API_KEY"] = ""


class ER():
    def __init__(self, img):
        super().__init__()
        self.detector = FER()
        h, w, c = img.shape
        self.center_x = w / 2
        self.center_y = h / 2

    def emotion_reconiton(self, img):
        emotions = self.detector.detect_emotions(img)

        if len(emotions) == 0:
            return img, {emotion: 0 for emotion in ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]}
        min_distance = float('inf')
        closest_rect = None
        for i, emo in enumerate(emotions):
            x, y, w, h = emo['box']
            rect_center_x = x + w / 2
            rect_center_y = y + h / 2
            distance = math.hypot(self.center_x - rect_center_x, self.center_y - rect_center_y)
            if distance < min_distance:
                min_distance = distance
                closest_rect = i
        
        img = cv2.rectangle(img, (emotions[closest_rect]['box'][0], emotions[closest_rect]['box'][1]), (emotions[closest_rect]['box'][0]+emotions[closest_rect]['box'][2], emotions[closest_rect]['box'][1]+emotions[closest_rect]['box'][3]), (255, 0, 0))
        
        return img, emotions[closest_rect]['emotions']


class OpenAIAPI:
    def __init__(self, client, model=""):
        self.client = client
        self.model = model


class ChatGPT(OpenAIAPI):
    def __init__(self, client, system_prompt_text="", model="gpt-4o"):
        super().__init__(client, model)
        self.system_prompt_text = system_prompt_text
        self.message_history = []
        
        self.set_system_prompt(system_prompt_text)


    def set_system_prompt(self, text):
        self.system_prompt_text = text
        system_message = {"role": "system", "content": text}
        self.message_history.append(system_message)


    def chat(self, text):
        user_message = {"role": "user", "content": text}
        self.message_history.append(user_message)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.message_history
            )
        except Exception as e:
            print(f"[ChatGPT] API処理中のエラー: {e}")
            return False

        response_text = response.choices[0].message.content
        assistant_message = {"role": "assistant", "content": response_text}
        self.message_history.append(assistant_message)
        print(f"[ChatGPT] レスポンステキスト:\n{response_text}")

        return response_text
    

class Text2Speech(OpenAIAPI):
    def __init__(self, client, model="tts-1", path="./text2speech.wav"):
        super().__init__(client, model)
        self.voice = "nova"
        self.format = "wav"
        self.path = path


    def speech(self, text):
        try:
            response = self.client.audio.speech.create(
                model=self.model,
                voice=self.voice,
                response_format=self.format,
                input=text,
            )
            response.stream_to_file(self.path)
            print(f"[Text2Speech] 音声ファイルに保存しました: {self.path}")
            return True
        except Exception as e:
            print(f"[Text2Speech] API処理中のエラー: {e}")
            return False


class EmotionDetection(Process):
    def __init__(self, frame_queue, emotion_queue, close_event):
        super().__init__()
        self.frame_queue = frame_queue
        self.emotion_queue = emotion_queue
        self.close_event = close_event
        self.cap = None

    def close(self):
        if self.cap is not None:
            self.cap.release()
            print("Close camera capture.")
        self.cap = None

    def cap_open(self):
        for i in range(20):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap = cap
                print(f"Capture opened on device {i}.")
                return True
            cap.release()
        print("Failed to open any capture device.")
        return False

    def recognizer(self, frame=None):
        # ダミーでランダムな感情スコアを生成
        er = ER(frame)
        frame, emotions = er.emotion_reconiton(frame)
        #data = {emotion: random.random() for emotion in ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]}
        return frame, json.dumps(emotions)

    def run(self):
        if not self.cap_open():
            return
        while not self.close_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                continue
            frame, emotion = self.recognizer(frame)
            if emotion[0] != None:
                try:
                    self.frame_queue.put(frame, block=False)
                    self.emotion_queue.put(emotion, block=False)
                except Exception:
                    pass
        self.close()


def ai_agent(user_message: str, flg) -> str:
    global emotion_data
    if flg:
        chat_response = chatgpt.chat(user_message)
        return chat_response
    
    chatgpt_cot = ChatGPT(client, system_prompt2)

    message = f"ユーザーのメッセージ：{user_message} / 感情データ：{emotion_data}"
    chat_response_cot = chatgpt_cot.chat(message)

    message = f"ユーザーのメッセージ：{user_message} / 感情データ：{emotion_data} / 考慮事項：{chat_response_cot}"
    chat_response = chatgpt.chat(message)

    return f"{chat_response}"


def to_base64(image):
    _, buf = cv2.imencode(".jpg", image)
    return base64.b64encode(buf).decode()


def update_loop(camera_image: ft.Image,
                frame_queue: Queue,
                emotion_queue: Queue,
                line_chart: ft.LineChart,
                current_text: ft.Text,
                emotions: list,
                page: ft.Page):
    # グラフ用にタイマーを初期化
    start_time = time.time()
    last_emotion_time = 0

    global emotion_data

    while True:
        # カメラフレーム更新
        frame = frame_queue.get()
        camera_image.src_base64 = to_base64(frame)
        camera_image.update()

        # 感情データを非同期で取得
        try:
            emotion_json = emotion_queue.get_nowait()
        except Exception:
            continue

        now = time.time()
        # 1秒間隔で更新
        if now - last_emotion_time >= 1.0:
            elapsed = int(now - start_time)
            
            data = json.loads(emotion_json)
            emotion_data = data.copy()
            # 各感情スコアをプロット
            for idx, emotion in enumerate(emotions):
                val = data.get(emotion, 0)
                dp = ft.LineChartDataPoint(elapsed, val)
                series = line_chart.data_series[idx]
                series.data_points.append(dp)
                # 直近60秒だけを保持
                series.data_points = [pt for pt in series.data_points if pt.x > elapsed - 60]

            line_chart.update()

            # 最新値と最大感情を表示
            latest_vals = [s.data_points[-1].y for s in line_chart.data_series]
            max_idx = int(np.argmax(latest_vals))
            max_emotion = emotions[max_idx]
            max_value = latest_vals[max_idx]
            current_text.value = f"Current: {max_emotion} ({max_value:.2f})"
            current_text.update()

            page.update()
            last_emotion_time = now


def main(page: ft.Page):
    page.title = "InoChat"
    page.bgcolor = "#F5F5F5"

    # キューとイベントを初期化
    frame_queue = Queue(maxsize=3)
    emotion_queue = Queue(maxsize=3)
    close_event = Event()

    # 感情認識プロセスの起動
    detect = EmotionDetection(frame_queue, emotion_queue, close_event)
    detect.daemon = True
    detect.start()

    # 初期フレーム
    frame_a = cv2.imread("./ChatGPT_Image_2025426_15_41_07.png")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cam_agent = ft.Image(src_base64=to_base64(frame_a[30:800, :, :]), expand=True, fit=ft.ImageFit.CONTAIN)
    cam_user = ft.Image(src_base64=to_base64(frame), expand=True, fit=ft.ImageFit.CONTAIN)

    emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    color_map = {
        "angry": "#E53935",
        "disgust": "#8E24AA",
        "fear": "#3949AB",
        "happy": "#FDD835",
        "sad": "#1E88E5",
        "surprise": "#FB8C00",
        "neutral": "#43A047"
    }

    data_series = [
        ft.LineChartData(
            data_points=[],
            color=color_map[e],
            stroke_width=2,
            curved=True,
            stroke_cap_round=True
        ) for e in emotions
    ]
    bottom_axis = ft.ChartAxis()
    left_axis = ft.ChartAxis(title=ft.Text("Score", size=12))
    line_chart = ft.LineChart(
        expand=True,
        data_series=data_series,
        interactive=True,
        bottom_axis=bottom_axis,
        left_axis=left_axis
    )
    current_text = ft.Text("Current: -", size=16)

    # レイアウト作成
    left_top_card = ft.Card(
        elevation=4,
        content=ft.Container(
            content=ft.Row([
                ft.Column([
                    cam_agent,
                    ft.Text("エージェント", weight=ft.FontWeight.BOLD, size=18, text_align=ft.TextAlign.CENTER)
                ], alignment=ft.MainAxisAlignment.CENTER, spacing=5, expand=True),
                ft.Column([
                    cam_user,
                    ft.Text("あなた", weight=ft.FontWeight.BOLD, size=18, text_align=ft.TextAlign.CENTER)
                ], alignment=ft.MainAxisAlignment.CENTER, spacing=5, expand=True)
            ], alignment=ft.MainAxisAlignment.CENTER, expand=True),
            padding=10,
            bgcolor="#FFFFFF"
        ),
        color="#64B5F6"
    )

    left_bottom_card = ft.Card(
        elevation=4,
        content=ft.Container(
            content=ft.Column([line_chart, current_text], spacing=10, expand=True),
            padding=10,
            bgcolor="#FFFFFF"
        ),
        color="#FFB74D"
    )

    chat_history = ft.ListView(expand=True, spacing=8, auto_scroll=True)
    text_input = ft.TextField(hint_text="メッセージを入力...",
                              expand=True)
    def send_message(e):
        msg = text_input.value.strip()
        if not msg:
            return
        chat_history.controls.append(ft.Text(f"あなた: {msg}", color="#1E88E5"))
        page.update()
        text_input.value = ""
        reply = ai_agent(msg, False)
        chat_history.controls.append(ft.Text(f"エージェント: {reply}", color="#43A047"))
        page.update()

    def send_message_end(e):
        chat_history.controls.append(ft.Text(f" --- まとめ ---"))
        msg = "これまでのやり取りを全て見直して、最初と最後でどれだけ心の健康状態が良くなったか、詳細に考察してまとめてください。"
        #reply = ai_agent(msg, True)
        chatgpt3 = ChatGPT(client, "")
        chatgpt3.message_history = chatgpt.message_history
        reply = chatgpt3.chat(msg)
        chat_history.controls.append(ft.Text(f"まとめ: {reply}"))
        page.update()
    
    send_button = ft.IconButton(icon=ft.Icons.SEND, on_click=send_message)
    end_button = ft.IconButton(icon=ft.Icons.CLOSE, on_click=send_message_end)
    input_row = ft.Row([text_input, send_button, end_button], spacing=5)
    right_card = ft.Card(
        elevation=4,
        content=ft.Container(
            content=ft.Column([chat_history, input_row], spacing=5, expand=True),
            padding=10,
            bgcolor="#FFFFFF"
        ),
        color="#81C784"
    )

    layout = ft.Row([
        ft.Container(content=ft.Column([left_top_card, left_bottom_card], expand=True), expand=True),
        ft.Container(content=right_card, expand=True)
    ], expand=True)

    page.add(layout)
    page.update()

    # 更新スレッド起動
    update_thread = Thread(
        target=update_loop,
        args=(cam_user, frame_queue, emotion_queue, line_chart, current_text, emotions, page),
        daemon=True
    )
    update_thread.start()


system_prompt1 = """
あなたは「ユーザーの感情を良い（健康的）方向へ導くプロフェッショナル」です。
次に示す【ユーザーのメッセージ】と【感情データ】をもとに、
【考慮事項リスト】に従って、最適なレスポンスを日本語で作成してください。

【入力情報】
・ユーザーのメッセージ：ここにユーザーの書いたテキストが入る
・感情データ：ここに推定された感情ラベル：例「悲しみ」「怒り」「不安」などが入る
・考慮事項リスト

【出力指示】
【考慮事項リスト】をすべて満たすように文章を作成してください
・文章の長さは自然な会話1〜3文程度（150文字以内推奨）
・文体はやさしく丁寧な日本語
・ユーザーの感情に共感を示した後、希望や安心感を与える流れで
・押しつけがましい励ましや過剰なポジティブ表現は避ける
・感情データに応じて、口調やテンポを微調整する
"""

system_prompt2 = """
あなたは「ユーザーの感情状態を良い（健康的）方向に導く専門家」です。
以下に与えられる情報は、
・ユーザーが書いた文章
・その文章から推定された感情データ（例：悲しみ、怒り、不安、喜び、驚き など）
です。

この情報をもとに、ユーザーの感情を少しでも明るく、健康的な方向に導くために、
**返答する言葉を考える際の【考慮事項】**をリストアップしてください。

【考慮事項に必ず含めるべきポイント】
・現在の感情を正しく受け止め、共感を示す
・無理にポジティブを押し付けない（自然な気持ちの流れを尊重する）
・小さな希望や安心感を与える
・具体的に寄り添った言葉を使う（抽象的な励ましより、状況に合わせたフレーズを）
・否定せずに、肯定的な視点を提案する

次に取れそうな小さな行動や考え方を優しく提案する
・過度なアドバイスや指示にならないよう注意する
・丁寧で温かみのある言葉遣いを意識する
・文章のトーンを、ユーザーのテンポに合わせる（急がせない、焦らせない）
・相手の「頑張り」や「存在」そのものを肯定する

【出力形式】
・箇条書き形式で10項目以上
・必ず簡潔かつ実用的な日本語で書く
・特に重要なポイントは強調（例：「無理にポジティブにしない」）
"""
client = OpenAI()
chatgpt = ChatGPT(client, system_prompt1)

emotion_data = {}

if __name__ == "__main__":
    ft.app(target=main)
