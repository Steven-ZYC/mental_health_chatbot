import os
import sys  # 添加sys模块导入
import cv2
import numpy as np
import msvcrt
import json  # 添加到文件顶部
import time  # 添加time模块导入
import torch  # 添加torch模块导入
from typing import Dict, List, Optional, Union, Tuple
from services.emotion_detector import EmotionDetector
from services.language_processor import LanguageProcessor
from services.response_generator import ResponseGenerator


class MentalHealthChatbot:
    """
    聊天机器人类，用于处理用户输入并生成响应。
    """
    def __init__(self, model_path: str, class_labels_path: str):
        # Initialize components
        self.emotion_detector = EmotionDetector(model_path, class_labels_path)
        self.language_processor = LanguageProcessor()
        self.response_generator = ResponseGenerator()
        
        # 加载情绪标签
        with open(class_labels_path, 'r') as f:
            self.emotion_labels = list(json.load(f).keys())
        
        # Initialize conversation history
        self.conversation_history: List[Dict[str, str]] = []
        
        # 初始化时不立即打开摄像头，等待用户选择
        self.cap = None
        
        # 添加开发者模式标志
        self.dev_mode = False
        self.windows_created = False  # 添加窗口创建状态标志
        self.camera_enabled = False   # 添加摄像头状态标志
        
        # 添加帧率计算相关变量
        self.frame_count = 0
        self.fps = 0
        self.fps_time = time.time()
        
        # 获取设备信息
        self.device = "CPU"
        if torch.cuda.is_available():
            self.device = f"GPU ({torch.cuda.get_device_name(0)})"
    
    def process_frame(self):
        """处理视频帧并返回情绪概率"""
        try:
            ret, frame = self.cap.read()
            if not ret:
                return None
            
            # 计算帧率
            self.frame_count += 1
            if (time.time() - self.fps_time) > 1:
                self.fps = self.frame_count / (time.time() - self.fps_time)
                self.frame_count = 0
                self.fps_time = time.time()
                
            # 使用EmotionDetector处理帧
            processed_frame = self.emotion_detector.process_frame(frame)
            
            # 获取最新的情绪数据
            if self.emotion_detector.latest_emotions:
                # 取第一个检测到的人脸的情绪
                emotion_data = self.emotion_detector.latest_emotions[0]
                emotion = emotion_data['label']
                probabilities = emotion_data['probabilities']
                
                # 根据开发者模式决定显示内容
                if self.dev_mode:
                    # 在帧上添加帧率和设备信息
                    fps_text = f"FPS: {self.fps:.1f}"
                    device_text = f"设备: {self.device}"
                    cv2.putText(processed_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(processed_frame, device_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # 显示处理后的帧
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
        """运行聊天机器人"""
        print("Mental Health Support Chatbot")
        print("==========================================")
        # 新增模式选择
        while True:
            mode = input("please choose the mode ([1]user mode [2]developer mode): ").strip()
            if mode == '1':
                self.dev_mode = False
                consent = input("Do you agree to let this program detect your facial information locally only for analyze your emotion?(y/n): ").lower()
                if consent == 'y':
                    print("facial analyzing...")
                    try:
                        self.cap = cv2.VideoCapture(0)
                        if self.cap.isOpened():
                            self.camera_enabled = True
                            print("Camara is successfully opened, facial analyzing is enabled")
                        else:
                            print("Can not open the camera, facial analyzing is disabled. Only text emotion analysis will be used.")
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
            
            # 简化用户模式的交互方式
            if not self.dev_mode:
                while chat_active:
                    # 处理视频帧获取情绪数据
                    visual_emotion_data = None
                    if self.camera_enabled and self.cap and self.cap.isOpened():
                        visual_emotion_data = self.process_frame()
                        # 处理视频窗口的按键事件
                        key = cv2.waitKey(30) & 0xFF
                        if key == ord('q'):
                            if input("\nAre you sure to exit？(y/n): ").lower() == 'y':
                                chat_active = False
                                continue
                    
                    # 获取用户输入
                    user_input = input("You: ").strip()
                    if not user_input:
                        continue
                        
                    if user_input.lower() == 'q':
                        if input("Sure to exit？(y/n): ").lower() == 'y':
                            chat_active = False
                            continue
                    
                    # 处理输入并生成响应
                    response = self.process_user_input(user_input, visual_emotion_data)
                    print(f"Chatbot: {response}\n")
            else:
                # 开发者模式使用非阻塞式输入
                input_buffer = ""
                print("You: ", end="", flush=True)  # 提示用户输入
                
                while chat_active:
                    # 处理视频帧
                    visual_emotion_data = None
                    if self.camera_enabled and self.cap and self.cap.isOpened():
                        visual_emotion_data = self.process_frame()
                        if visual_emotion_data is None and self.dev_mode:
                            print("\rWarning: Unable to get video frame, but will continue running", end="", flush=True)
                            print("\nYou: " + input_buffer, end="", flush=True)  # 恢复输入提示

                    # 开发者模式专用显示
                    if self.dev_mode and visual_emotion_data:
                        emotion_text = visual_emotion_data.get('emotion', 'unknown')
                        # 在新行显示实时数据，不覆盖输入
                        print(f"\r[Real-time] Detected emotion: {emotion_text} | FPS: {self.fps:.1f} | Device: {self.device}                ")
                        print("You: " + input_buffer, end="", flush=True)  # 恢复输入提示

                    # 检查键盘输入
                    if msvcrt.kbhit():
                        char = msvcrt.getch().decode('utf-8', errors='ignore')
                        
                        # 处理回车键
                        if char == '\r':
                            print()  # 换行
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
                            
                            # 处理输入并生成响应
                            response = self.process_user_input(user_input, visual_emotion_data)
                            print(f"Chatbot: {response}\n")
                            print("You: ", end="", flush=True)
                        
                        # 处理退格键
                        elif char == '\b':
                            if input_buffer:
                                input_buffer = input_buffer[:-1]
                                sys.stdout.write('\b \b')  # 删除一个字符
                                sys.stdout.flush()
                        
                        # 处理其他可打印字符
                        elif char.isprintable():
                            input_buffer += char
                            sys.stdout.write(char)
                            sys.stdout.flush()
                    
                    # 处理退出逻辑
                    if self.camera_enabled:
                        key = cv2.waitKey(30) & 0xFF
                        if key == ord('q'):
                            print("\n")
                            if input("Confirm exit? (y/n): ").lower() == 'y':
                                chat_active = False
                                continue
                            else:
                                print("You: " + input_buffer, end="", flush=True)

                    # 添加短暂延迟，减少CPU使用率
                    cv2.waitKey(10)  # 10毫秒延迟

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
            print("\nResources released, thank you for using!")

    def process_user_input(self, user_input: str, visual_emotion_data: Optional[Dict[str, Union[str, Dict[str, float]]]] = None) -> str:
        """处理用户输入并生成响应"""
        try:
            # 检测文本情绪
            text_emotion, text_confidence = self.language_processor.detect_text_emotion(user_input)
            
            # 获取上一次的情绪状态（如果有）
            previous_emotion = None
            if self.conversation_history:
                previous_emotion = self.conversation_history[-1].get('emotion')
            
            # 检查用户是否直接表达了情绪
            explicit_emotions = {
                'sad': ['sad', 'unhappy', 'depressed', 'down', 'blue', 'gloomy', 'miserable', 'tired', 'exhausted', 'fatigue'],
                'angry': ['angry', 'mad', 'furious', 'irritated', 'annoyed', 'frustrated'],
                'happy': ['happy', 'glad', 'joyful', 'excited', 'pleased', 'delighted', 'cheerful'],
                'fear': ['scared', 'afraid', 'fearful', 'terrified', 'anxious', 'worried', 'nervous'],
                'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'stunned'],
                'disgust': ['disgusted', 'revolted', 'repulsed', 'gross'],
                'neutral': ['neutral', 'okay', 'fine', 'alright', 'normal']
            }
            
            # 检查用户输入是否直接包含情绪词
            user_explicit_emotion = None
            lower_input = user_input.lower()
            
            # 特殊情况处理：疲惫通常表示悲伤情绪
            if any(word in lower_input for word in ['tired', 'exhausted', 'fatigue', 'so tired']):
                user_explicit_emotion = 'sad'
                text_confidence = 0.9
                text_emotion = 'sad'
            else:
                # 常规情绪检测
                for emotion, keywords in explicit_emotions.items():
                    if any(keyword in lower_input for keyword in keywords):
                        user_explicit_emotion = emotion
                        text_confidence = 0.9  # 提高用户明确表达情绪的置信度
                        text_emotion = emotion
                        break
            
            # 检查危机指标
            crisis_indicators = self.language_processor.extract_crisis_indicators(user_input)
            
            # 验证情绪状态
            final_emotion = text_emotion
            if visual_emotion_data and 'emotion' in visual_emotion_data and 'probabilities' in visual_emotion_data:
                try:
                    # 如果用户明确表达了情绪，优先使用用户表达的情绪
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
                    # 如果方法不存在，继续使用文本情绪或上一次的情绪
                    print("Warning: Emotion verification function is not available, will use text emotion analysis only")
                    # 如果用户明确表达了情绪，优先使用用户表达的情绪
                    if user_explicit_emotion:
                        final_emotion = user_explicit_emotion
                    # 如果文本情绪是neutral且有上一次的情绪，使用上一次的情绪
                    elif text_emotion == 'neutral' and previous_emotion and previous_emotion != 'neutral':
                        final_emotion = previous_emotion
                        print(f"Using previous emotion state: {final_emotion}")
                    else:
                        final_emotion = text_emotion
            elif user_explicit_emotion:
                # 如果没有视觉数据但用户明确表达了情绪
                final_emotion = user_explicit_emotion
            # 如果文本情绪是neutral且有上一次的情绪，使用上一次的情绪
            elif text_emotion == 'neutral' and previous_emotion and previous_emotion != 'neutral':
                final_emotion = previous_emotion
                print(f"Using previous emotion state: {final_emotion}")
            
            # 生成响应
            response = self.response_generator.generate_response(
                user_input,
                final_emotion,
                crisis_indicators,
                self.conversation_history
            )
            
            # 更新对话历史
            self.conversation_history.append({
                'user_input': user_input,
                'emotion': final_emotion,
                'response': response
            })
            
            # 保持最后10轮对话
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
        # Check if model files exist
        model_path = 'models/best_model.pth'
        class_labels_path = 'models/class_labels.json'
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' does not exist. Please ensure the model file is correctly placed in the models directory.")
        
        if not os.path.exists(class_labels_path):
            raise FileNotFoundError(f"Label file '{class_labels_path}' does not exist. Please ensure the label file is correctly placed in the models directory.")
        
        # Initialize chatbot with model paths
        chatbot = MentalHealthChatbot(
            model_path=model_path,
            class_labels_path=class_labels_path
        )
        
        # Run chatbot
        chatbot.run()
    
    except FileNotFoundError as e:
        print(f"\nError: {str(e)}")
        print("Please ensure all required model files are correctly placed in the project directory.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("If the problem persists, please check if the model files are corrupted or contact technical support.")

if __name__ == '__main__':
    main()