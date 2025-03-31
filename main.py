import os
import sys  # 添加sys模块导入  import sys module
import cv2
import numpy as np
import msvcrt
import json  # 添加到文件顶部  Add to file top
import time  # 添加time模块导入  import time module
import torch  # 添加torch模块导入 import torch module import
from typing import Dict, List, Optional, Union, Tuple
from services.emotion_detector import EmotionDetector
from services.language_processor import LanguageProcessor
from services.response_generator import ResponseGenerator
# Oscar modification
from services.tts_engine import TTSEngine  # 导入TTSEngine以实现TTS功能  Import TTSEngine for TTS functionality

class MentalHealthChatbot:
    """
    聊天机器人类，用于处理用户输入并生成响应，新增TTS功能。
    A chatbot with tts feature and can process and response to user's input
    """

    def __init__(self, model_path: str, class_labels_path: str):
        # Initialize components
        self.emotion_detector = EmotionDetector(model_path, class_labels_path)
        self.language_processor = LanguageProcessor()
        self.response_generator = ResponseGenerator()
        # Oscar modification
        self.tts_engine = TTSEngine(rate=150, volume=1.0)  # 用男性声音初始化TTS引擎  Initialize TTS engine with male voice

        # 加载情绪标签  Load emotion labels
        with open(class_labels_path, 'r') as f:
            self.emotion_labels = list(json.load(f).keys())

        #初始化对话历史 Initialize conversation history
        self.conversation_history: List[Dict[str, str]] = []

        # 初始化时不立即打开摄像头，等待用户选择  Do not open the camera immediately upon initialization, wait for user selection
        self.cap = None

        # 添加开发者模式标志  Add developer mode flag
        self.dev_mode = False
        self.windows_created = False  # 添加窗口创建状态标志  Add window creation status flag
        self.camera_enabled = False  # 添加摄像头状态标志  Add camera status flag

        # 添加帧率计算相关变量  Add variables related to frame rate calculation
        self.frame_count = 0
        self.fps = 0
        self.fps_time = time.time()

        # 获取设备信息  Get device information
        self.device = "CPU"
        if torch.cuda.is_available():
            self.device = f"GPU ({torch.cuda.get_device_name(0)})"
        # Oscar modification
        # 添加TTS控制标志  Add TTS control flag
        self.tts_enabled = True  # TTS默认开启  TTS enabled by default

    def process_frame(self):
        """处理视频帧并返回情绪概率
            Process the image input and return the emotions probability
        """

        try:
            ret, frame = self.cap.read()
            if not ret:
                return None

            # 计算帧率  calculate the probability
            self.frame_count += 1
            if (time.time() - self.fps_time) > 1:
                self.fps = self.frame_count / (time.time() - self.fps_time)
                self.frame_count = 0
                self.fps_time = time.time()

            # 使用EmotionDetector处理帧  use EmotionDetector for frame processing
            processed_frame = self.emotion_detector.process_frame(frame)

            # 获取最新的情绪数据  get the latest emotions data
            if self.emotion_detector.latest_emotions:
                # 取第一个检测到的人脸的情绪  get the first person's emotions
                emotion_data = self.emotion_detector.latest_emotions[0]
                emotion = emotion_data['label']
                probabilities = emotion_data['probabilities']

                # 根据开发者模式决定显示内容  determine the shown content according to the developer mode
                if self.dev_mode:
                    # 在帧上添加帧率和设备信息  Add frame rate and device information on the frame
                    fps_text = f"FPS: {self.fps:.1f}"
                    device_text = f"设备: {self.device}"
                    cv2.putText(processed_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(processed_frame, device_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # 显示处理后的帧  Display the processed frame
                    cv2.imshow('Emotion Detection', processed_frame)
                    if not self.windows_created:
                        cv2.moveWindow('Emotion Detection', 50, 50)
                        self.windows_created = True

                return {'emotion': emotion, 'probabilities': probabilities}
            return None

        except Exception as e:
            print(f"An error occurred when analyze the video: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def run(self):
        """运行聊天机器人，新增TTS功能"""
        # Oscar modification
        print("Mental Health Support Chatbot with Text-to-Speech")
        # 带文本转语音的心理健康支持聊天机器人  Mental Health Support Chatbot with Text-to-Speech
        print("==========================================")
        print("Type 'tts off' to disable speech, 'tts on' to enable it.")
        # 新增模式选择  Add mode selection
        while True:
            mode = input("please choose the mode ([1]user mode [2]developer mode): ").strip()
            if mode == '1':
                self.dev_mode = False
                consent = input(
                    "Do you agree to let this program detect your facial information locally only for analyze your emotion?(y/n): ").lower()
                if consent == 'y':
                    print("facial analyzing...")
                    try:
                        self.cap = cv2.VideoCapture(0)
                        if self.cap.isOpened():
                            self.camera_enabled = True
                            print("Camera is successfully opened, facial analyzing is enabled")
                        else:
                            print(
                                "Can not open the camera, facial analyzing is disabled. Only text emotion analysis will be used.")
                            self.cap = None
                    except Exception as e:
                        print(f"fail to use camera: {str(e)}")
                        self.cap = None
                else:
                    print("Only text emotion analysis will be used.")
                    self.cap = None
                break
            elif mode == '2':
                self.dev_mode = True
                print("Opening developer mode...")
                try:
                    self.cap = cv2.VideoCapture(0)
                    if self.cap.isOpened():
                        self.camera_enabled = True
                        print("Developer mode is enabled with camera")
                    else:
                        print("WARNING: Can not open the camera, developer mode is enabled without camera")
                except Exception as e:
                    print(f"Failed to use camera: {str(e)}")
                    self.cap = None
                break
            else:
                print("Invalid choice，please input 1 or 2")

        print("\nPress Enter after typing your message")
        print("input 'q' to exit the chat")
        print("==========================================")
        print("chatbot is starting,please enter your message...\n")

        try:
            chat_active = True

            # 简化用户模式的交互方式  Simplify interaction method for user mode
            if not self.dev_mode:
                while chat_active:
                    # 处理视频帧获取情绪数据  process the emotions data get from the image frames
                    visual_emotion_data = None
                    if self.camera_enabled and self.cap and self.cap.isOpened():
                        visual_emotion_data = self.process_frame()
                        # 处理视频窗口的按键事件  process the key input from the image window
                        key = cv2.waitKey(30) & 0xFF
                        if key == ord('q'):
                            if input("\nAre you sure to exit？(y/n): ").lower() == 'y':
                                chat_active = False
                                continue

                    # 获取用户输入  get the users input
                    user_input = input("You: ").strip()
                    if not user_input:
                        continue

                    if user_input.lower() == 'q':
                        if input("Sure to exit？(y/n): ").lower() == 'y':
                            chat_active = False
                            continue
                    # Oscar modification
                    elif user_input.lower() == 'tts off':
                        self.tts_enabled = False
                        print("Text-to-Speech disabled.")
                        continue
                    elif user_input.lower() == 'tts on':
                        self.tts_enabled = True
                        print("Text-to-Speech enabled.")
                        continue

                    # 处理输入并生成响应  process input and generate responses
                    response = self.process_user_input(user_input, visual_emotion_data)
                    print(f"Chatbot: {response}\n")
                    # Oscar modification
                    if self.tts_enabled:
                        self.tts_engine.speak(response)  # 大声朗读回复  Speak the response aloud

            else:
                # 开发者模式使用非阻塞式输入  Developer mode uses non-blocking input
                input_buffer = ""
                print("You: ", end="", flush=True)  # 提示用户输入  Prompt user input

                while chat_active:
                    # 处理视频帧   process image frame
                    visual_emotion_data = None
                    if self.camera_enabled and self.cap and self.cap.isOpened():
                        visual_emotion_data = self.process_frame()
                        if visual_emotion_data is None and self.dev_mode:
                            print("\rWarning: Unable to get video frame, but will continue running", end="", flush=True)
                            print("\nYou: " + input_buffer, end="", flush=True)  # 恢复输入提示

                    # 开发者模式专用显示  Developer mode only interface
                    if self.dev_mode and visual_emotion_data:
                        emotion_text = visual_emotion_data.get('emotion', 'unknown')
                        # 在新行显示实时数据，不覆盖输入  Display real-time data on a new line, without overwriting input
                        print(
                            f"\r[Real-time] Detected emotion: {emotion_text} | FPS: {self.fps:.1f} | Device: {self.device}                ")
                        print("You: " + input_buffer, end="", flush=True)  # 恢复输入提示  refresh the input prompt

                    # 检查键盘输入  check keyboard input
                    if msvcrt.kbhit():
                        char = msvcrt.getch().decode('utf-8', errors='ignore')

                        # 处理回车键  process Enter
                        if char == '\r':
                            print()  # 换行  return
                            user_input = input_buffer.strip()
                            input_buffer = ""

                            if not user_input:
                                print("You: ", end="", flush=True)
                                continue

                            if user_input.lower() == 'q':
                                if input("Confirm exit? (y/n): ").lower() == 'y':
                                    chat_active = False
                                    continue
                                else:
                                    print("You: ", end="", flush=True)
                                    continue
                            # Oscar modification
                            elif user_input.lower() == 'tts off':
                                self.tts_enabled = False
                                print("Text-to-Speech disabled.")
                                print("You: ", end="", flush=True)
                                continue
                            elif user_input.lower() == 'tts on':
                                self.tts_enabled = True
                                print("Text-to-Speech enabled.")
                                print("You: ", end="", flush=True)
                                continue

                            # 处理输入并生成响应  process input and generate responses
                            response = self.process_user_input(user_input, visual_emotion_data)
                            print(f"Chatbot: {response}\n")
                            # Oscar modification
                            if self.tts_enabled:
                                self.tts_engine.speak(response)  #大声朗读回复  Speak the response aloud

                            print("You: ", end="", flush=True)

                        # 处理退格键  process backspace
                        elif char == '\b':
                            if input_buffer:
                                input_buffer = input_buffer[:-1]
                                sys.stdout.write('\b \b')  # 删除一个字符  delete one word
                                sys.stdout.flush()

                        # 处理其他可打印字符  process other input-able key
                        elif char.isprintable():
                            input_buffer += char
                            sys.stdout.write(char)
                            sys.stdout.flush()

                    # 处理退出逻辑  process exiting logic
                    if self.camera_enabled:
                        key = cv2.waitKey(30) & 0xFF
                        if key == ord('q'):
                            print("\n")
                            if input("Confirm exit? (y/n): ").lower() == 'y':
                                chat_active = False
                                continue
                            else:
                                print("You: " + input_buffer, end="", flush=True)

                    # 添加短暂延迟，减少CPU使用率  Add a short delay to reduce CPU usage
                    cv2.waitKey(10)  # 10毫秒延迟  10ms delay

        except KeyboardInterrupt:
            print("\nProgram interrupted by user")
        except Exception as e:
            print(f"\nAn error occurred during execution: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            if self.cap and self.cap.isOpened():
                self.cap.release()
            cv2.destroyAllWindows()
            # Oscar modification

            self.tts_engine.stop()  # 确保TTS干净地停止  Ensure TTS stops cleanly

            print("\nResources released, thank you for using!")

    def process_user_input(self, user_input: str,
                           visual_emotion_data: Optional[Dict[str, Union[str, Dict[str, float]]]] = None) -> str:
        """处理用户输入并生成响应
            process user inputs and generate responses
        """
        try:
            # 检测文本情绪  detect text emotions
            text_emotion, text_confidence = self.language_processor.detect_text_emotion(user_input)

            # 获取上一次的情绪状态（如果有）  get the last emotions status (If applicable)
            previous_emotion = None
            if self.conversation_history:
                previous_emotion = self.conversation_history[-1].get('emotion')

            # 检查用户是否直接表达了情绪  Check if the user directly expressed an emotion
            explicit_emotions = {
                'sad': ['sad', 'unhappy', 'depressed', 'down', 'blue', 'gloomy', 'miserable', 'tired', 'exhausted',
                        'fatigue'],
                'angry': ['angry', 'mad', 'furious', 'irritated', 'annoyed', 'frustrated'],
                'happy': ['happy', 'glad', 'joyful', 'excited', 'pleased', 'delighted', 'cheerful'],
                'fear': ['scared', 'afraid', 'fearful', 'terrified', 'anxious', 'worried', 'nervous'],
                'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'stunned'],
                'disgust': ['disgusted', 'revolted', 'repulsed', 'gross'],
                'neutral': ['neutral', 'okay', 'fine', 'alright', 'normal']
            }

            # 检查用户输入是否直接包含情绪词  Check if the user input directly contains emotion words
            user_explicit_emotion = None
            lower_input = user_input.lower()

            # 特殊情况处理：疲惫通常表示悲伤情绪  Special case handling: Fatigue usually indicates sadness
            if any(word in lower_input for word in ['tired', 'exhausted', 'fatigue', 'so tired']):
                user_explicit_emotion = 'sad'
                text_confidence = 0.9
                text_emotion = 'sad'
            else:
                # 常规情绪检测  Regular emotion detection
                for emotion, keywords in explicit_emotions.items():
                    if any(keyword in lower_input for keyword in keywords):
                        user_explicit_emotion = emotion
                        text_confidence = 0.9  # 提高用户明确表达情绪的置信度
                        text_emotion = emotion
                        break

            # 检查危机指标  Check crisis indicators
            crisis_indicators = self.language_processor.extract_crisis_indicators(user_input)

            # 验证情绪状态  Verify the emotional status
            final_emotion = text_emotion
            if visual_emotion_data and 'emotion' in visual_emotion_data and 'probabilities' in visual_emotion_data:
                try:
                    # 如果用户明确表达了情绪，优先使用用户表达的情绪  if the user clearly express his emotion, priorities on using the user's expressed emotional
                    if user_explicit_emotion:
                        final_emotion = user_explicit_emotion
                    else:
                        final_emotion = self.emotion_detector.verify_emotion_state(
                            visual_emotion_data['emotion'],
                            visual_emotion_data['probabilities'],
                            text_emotion,
                            text_confidence
                        )
                except AttributeError:
                    # 如果方法不存在，继续使用文本情绪或上一次的情绪  if the method does not exist, continue to use the text emotion or the prior emotional
                    print(
                        "Warning: Emotion verification function is not available, will use text emotion analysis only")
                    # 如果用户明确表达了情绪，优先使用用户表达的情绪
                    if user_explicit_emotion:
                        final_emotion = user_explicit_emotion
                    # 如果文本情绪是neutral且有上一次的情绪，使用上一次的情绪  if the text emotion is neutral and the prior emotion exists, use the prior emotion
                    elif text_emotion == 'neutral' and previous_emotion and previous_emotion != 'neutral':
                        final_emotion = previous_emotion
                        print(f"Using previous emotion state: {final_emotion}")
                    else:
                        final_emotion = text_emotion
            elif user_explicit_emotion:
                # 如果没有视觉数据但用户明确表达了情绪  if there is no visual data but the user clearly express his emotions
                final_emotion = user_explicit_emotion
            # 如果文本情绪是neutral且有上一次的情绪，使用上一次的情绪  if the text emotion is neutral and the prior emotion exists, use the prior emotion
            elif text_emotion == 'neutral' and previous_emotion and previous_emotion != 'neutral':
                final_emotion = previous_emotion
                print(f"Using previous emotion state: {final_emotion}")

            # 生成响应  generate responses
            response = self.response_generator.generate_response(
                user_input,
                final_emotion,
                crisis_indicators,
                self.conversation_history
            )

            # 更新对话历史  append the text history
            self.conversation_history.append({
                'user_input': user_input,
                'emotion': final_emotion,
                'response': response
            })

            # 保持最后10轮对话  Keep the last 10 rounds of conversation
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]

            return response
        except Exception as e:
            print(f"处理用户输入时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return "I'm sorry, I encountered a problem processing your message. Please try again."


def main():
    try:
        # 检查模型档案是否存在  Check if model files exist
        model_path = 'models/best_model.pth'
        class_labels_path = 'models/class_labels.json'

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file '{model_path}' does not exist. Please ensure the model file is correctly placed in the models directory.")

        if not os.path.exists(class_labels_path):
            raise FileNotFoundError(
                f"Label file '{class_labels_path}' does not exist. Please ensure the label file is correctly placed in the models directory.")

        # 以模型路徑以初始化語言模型  Initialize chatbot with model paths
        chatbot = MentalHealthChatbot(
            model_path=model_path,
            class_labels_path=class_labels_path
        )

        # 運行語言  Run chatbot
        chatbot.run()

    except FileNotFoundError as e:
        print(f"\nError: {str(e)}")
        print("Please ensure all required model files are correctly placed in the project directory.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("If the problem persists, please check if the model files are corrupted or contact technical support.")


if __name__ == '__main__':
    main()