import flet as ft
from dataclasses import field

# 各種ボタンなどのコンポーネント
@ft.control
class VoiceButton(ft.Button):
    expand: bool = field(default=True)

@ft.control
class StartVoiceButton(VoiceButton):
    bgcolor: ft.Colors = ft.Colors.GREY_900

@ft.control
class FinishVoiceButton(VoiceButton):
    bgcolor: ft.Colors = ft.Colors.RED

@ft.control
class GenTxt(ft.Text):
    color: ft.Colors = ft.Colors.BLACK
    textAlign: ft.TextAlign = ft.TextAlign.CENTER