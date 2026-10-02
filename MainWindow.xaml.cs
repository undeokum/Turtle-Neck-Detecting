using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using OpenCvSharp.WpfExtensions;

namespace TurtleNeckDetector
{
    public partial class MainWindow : System.Windows.Window
    {
        private VideoCapture? _capture;
        private BackgroundWorker? _worker;
        private bool _isTracking = false;

        // ONNX AI 추론 세션
        private InferenceSession? _onnxSession;

        // 임계 각도 설정 (측면 기준)
        private const double TURTLE_ANGLE_THRESHOLD = 25.0;

        // 세션 타이머 및 집중도 산출 관련 변수
        private DispatcherTimer _sessionTimer;
        private int _totalSecondsRemaining = 0;
        private float _currentFocusScore = 100f;

        private bool _isBadPosture = false;
        private bool _wasBadPosturePrevious = false;
        private int _badPostureChangeCount = 0;
        private double _currentNeckAngle = 0.0;
        private bool _isPersonDetected = false;

        public MainWindow()
        {
            InitializeComponent();
            InitOnnxModel();

            // 1초 단위 타이머 생성
            _sessionTimer = new DispatcherTimer
            {
                Interval = TimeSpan.FromSeconds(1)
            };
            _sessionTimer.Tick += SessionTimer_Tick;
        }

        // 1. ONNX 모델 로드
        private void InitOnnxModel()
        {
            string modelPath = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "yolov8n-pose.onnx");
            if (File.Exists(modelPath))
            {
                try
                {
                    _onnxSession = new InferenceSession(modelPath);
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"ONNX 모델 로드 실패: {ex.Message}");
                }
            }
            else
            {
                MessageBox.Show("yolov8n-pose.onnx 파일이 실행 폴더에 없습니다.\n파일을 실행 폴더(bin/.../net10.0-windows)에 넣고 다시 실행하세요.");
            }
        }

        // 2. 측정 시작 버튼 클릭
        private void BtnStart_Click(object sender, RoutedEventArgs e)
        {
            if (_isTracking) return;

            _capture = new VideoCapture(0);
            _capture.Set(VideoCaptureProperties.FrameWidth, 640);
            _capture.Set(VideoCaptureProperties.FrameHeight, 480);

            if (!_capture.IsOpened())
            {
                MessageBox.Show("웹캠을 열 수 없습니다.");
                return;
            }

            // 선택한 세션 시간(분) 읽기
            var selectedItem = (ComboBoxItem)ComboDuration.SelectedItem;
            int minutes = int.Parse(selectedItem.Tag.ToString() ?? "30");

            // 집중도 변수 초기화
            _totalSecondsRemaining = minutes * 60;
            _currentFocusScore = 100f;
            _badPostureChangeCount = 0;
            _wasBadPosturePrevious = false;

            _isTracking = true;
            BtnStart.IsEnabled = false;
            BtnStop.IsEnabled = true;
            ComboDuration.IsEnabled = false;

            _sessionTimer.Start();

            _worker = new BackgroundWorker { WorkerSupportsCancellation = true };
            _worker.DoWork += Worker_DoWork;
            _worker.RunWorkerAsync();
        }

        // 3. 웹캠 프레임 수집 및 ONNX 추론 루프
        private void Worker_DoWork(object? sender, DoWorkEventArgs e)
        {
            using (Mat frame = new Mat())
            {
                while (_isTracking && _capture != null && !_capture.IsDisposed)
                {
                    _capture.Read(frame);
                    if (frame.Empty()) continue;

                    // 좌우 반전 (거울 모드)
                    Cv2.Flip(frame, frame, FlipMode.Y);

                    OpenCvSharp.Point? earPos = null;
                    OpenCvSharp.Point? shoulderPos = null;

                    if (_onnxSession != null)
                    {
                        var keypoints = DetectPose(frame);
                        if (keypoints != null && keypoints.Count >= 17)
                        {
                            // YOLOv8 Keypoint Index: 3=Right Ear, 5=Right Shoulder / 4=Left Ear, 6=Left Shoulder
                            earPos = keypoints[3].Confidence > 0.3 ? keypoints[3].Point : keypoints[4].Point;
                            shoulderPos = keypoints[5].Confidence > 0.3 ? keypoints[5].Point : keypoints[6].Point;
                        }
                    }

                    if (earPos.HasValue && shoulderPos.HasValue)
                    {
                        _isPersonDetected = true;
                        OpenCvSharp.Point ear = earPos.Value;
                        OpenCvSharp.Point shoulder = shoulderPos.Value;

                        double xDiff = Math.Abs(ear.X - shoulder.X);
                        double yDiff = Math.Abs(ear.Y - shoulder.Y);
                        double angleRad = Math.Atan(xDiff / (yDiff == 0 ? 1 : yDiff));
                        _currentNeckAngle = angleRad * (180.0 / Math.PI);

                        bool isTurtle = _currentNeckAngle > TURTLE_ANGLE_THRESHOLD;

                        // 관절 시각화
                        Cv2.Line(frame, ear, shoulder, Scalar.Cyan, 2);
                        Cv2.Circle(frame, ear, 6, Scalar.Blue, -1);
                        Cv2.Circle(frame, shoulder, 6, Scalar.Green, -1);

                        BitmapSource bitmapSource = WriteableBitmapConverter.ToWriteableBitmap(frame);
                        bitmapSource.Freeze();

                        Dispatcher.Invoke(() =>
                        {
                            CamImage.Source = bitmapSource;
                            TxtAngle.Text = $"목 각도: {_currentNeckAngle:F1}°";

                            if (isTurtle)
                            {
                                TxtStatus.Text = "경고: 거북목 감지됨!";
                                TxtStatus.Foreground = Brushes.Red;
                            }
                            else
                            {
                                TxtStatus.Text = "상태: 바른 자세";
                                TxtStatus.Foreground = Brushes.Green;
                            }
                        });
                    }
                    else
                    {
                        _isPersonDetected = false;
                        BitmapSource bitmapSource = WriteableBitmapConverter.ToWriteableBitmap(frame);
                        bitmapSource.Freeze();

                        Dispatcher.Invoke(() =>
                        {
                            CamImage.Source = bitmapSource;
                            TxtAngle.Text = "목 각도: 측정 불가";
                            TxtStatus.Text = "상태: 관절 인식 중...";
                            TxtStatus.Foreground = Brushes.Orange;
                        });
                    }
                }
            }
        }

        // 4. 1초 주기 타이머 틱 - 복합 집중도 점수 계산
        private void SessionTimer_Tick(object? sender, EventArgs e)
        {
            if (_totalSecondsRemaining > 0)
            {
                _totalSecondsRemaining--;

                // 남은 시간 표출
                TimeSpan ts = TimeSpan.FromSeconds(_totalSecondsRemaining);
                TxtRemainingTime.Text = $"남은 시간: {ts.Hours * 60 + ts.Minutes:D2}:{ts.Seconds:D2}";

                // 관절 인식이 정상일 때만 집중도 연산 적용
                if (_isPersonDetected)
                {
                    _isBadPosture = _currentNeckAngle > TURTLE_ANGLE_THRESHOLD;

                    // [빈도 감점] 바른 자세 -> 거북목으로 넘어간 첫 순간
                    if (_isBadPosture && !_wasBadPosturePrevious)
                    {
                        _badPostureChangeCount++;
                        _currentFocusScore -= 3.0f; // 이탈 1회당 -3점
                    }
                    _wasBadPosturePrevious = _isBadPosture;

                    // [누적 중증도 및 회복 연산]
                    if (_isBadPosture)
                    {
                        float severityPenalty = (float)(_currentNeckAngle - TURTLE_ANGLE_THRESHOLD) * 0.05f;
                        _currentFocusScore -= (0.2f + severityPenalty);
                    }
                    else
                    {
                        _currentFocusScore += 0.15f;
                    }

                    _currentFocusScore = Math.Clamp(_currentFocusScore, 0f, 100f);
                }

                UpdateFocusUI();
            }
            else
            {
                StopCamera();
                MessageBox.Show($"집중도 측정 세션이 완료되었습니다!\n\n" +
                                $"최종 집중도: {(int)_currentFocusScore}점\n" +
                                $"자세 이탈 횟수: {_badPostureChangeCount}회",
                                "세션 완료", MessageBoxButton.OK, MessageBoxImage.Information);
            }
        }

        // 5. 집중도 UI 업데이트
        private void UpdateFocusUI()
        {
            FocusProgressBar.Value = _currentFocusScore;

            string focusState;
            Brush stateColor;

            if (_currentFocusScore >= 80)
            {
                focusState = "몰입";
                stateColor = (Brush)new BrushConverter().ConvertFrom("#059669")!;
                FocusProgressBar.Foreground = (Brush)new BrushConverter().ConvertFrom("#10B981")!;
            }
            else if (_currentFocusScore >= 50)
            {
                focusState = "보통";
                stateColor = Brushes.DarkOrange;
                FocusProgressBar.Foreground = Brushes.Orange;
            }
            else
            {
                focusState = "산만/피로";
                stateColor = Brushes.Red;
                FocusProgressBar.Foreground = Brushes.Red;
            }

            TxtFocusScore.Text = $"{(int)_currentFocusScore}점 ({focusState})";
            TxtFocusScore.Foreground = stateColor;

            TxtSessionStats.Text = $"자세 이탈 횟수: {_badPostureChangeCount}회 | 실시간 각도: {_currentNeckAngle:F1}°";
        }

        // 6. YOLOv8-Pose 전처리 및 추론 파서
        private List<(OpenCvSharp.Point Point, float Confidence)>? DetectPose(Mat src)
        {
            int netWidth = 640;
            int netHeight = 640;

            using Mat resized = new Mat();
            Cv2.Resize(src, resized, new OpenCvSharp.Size(netWidth, netHeight));

            var inputTensor = new DenseTensor<float>(new[] { 1, 3, netHeight, netWidth });
            for (int y = 0; y < netHeight; y++)
            {
                for (int x = 0; x < netWidth; x++)
                {
                    Vec3b color = resized.At<Vec3b>(y, x);
                    inputTensor[0, 0, y, x] = color.Item2 / 255.0f; // R
                    inputTensor[0, 1, y, x] = color.Item1 / 255.0f; // G
                    inputTensor[0, 2, y, x] = color.Item0 / 255.0f; // B
                }
            }

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("images", inputTensor)
            };

            using var results = _onnxSession?.Run(inputs);
            var output = results?.First().AsTensor<float>();
            if (output == null) return null;

            float scaleX = (float)src.Width / netWidth;
            float scaleY = (float)src.Height / netHeight;

            int bestCol = -1;
            float maxScore = 0f;

            for (int i = 0; i < 8400; i++)
            {
                float score = output[0, 4, i];
                if (score > maxScore && score > 0.5f)
                {
                    maxScore = score;
                    bestCol = i;
                }
            }

            if (bestCol == -1) return null;

            var keypoints = new List<(OpenCvSharp.Point Point, float Confidence)>();
            for (int k = 0; k < 17; k++)
            {
                float kx = output[0, 5 + k * 3, bestCol] * scaleX;
                float ky = output[0, 5 + k * 3 + 1, bestCol] * scaleY;
                float conf = output[0, 5 + k * 3 + 2, bestCol];

                keypoints.Add((new OpenCvSharp.Point((int)kx, (int)ky), conf));
            }

            return keypoints;
        }

        // 7. 정지 및 정리 로직
        private void StopCamera()
        {
            _isTracking = false;
            _sessionTimer?.Stop();

            _capture?.Release();
            _capture?.Dispose();

            CamImage.Source = null;
            TxtStatus.Text = "상태: 대기 중";
            TxtStatus.Foreground = Brushes.Gray;
            TxtRemainingTime.Text = "남은 시간: --:--";

            BtnStart.IsEnabled = true;
            BtnStop.IsEnabled = false;
            ComboDuration.IsEnabled = true;
        }

        private void BtnStop_Click(object sender, RoutedEventArgs e)
        {
            StopCamera();
        }

        protected override void OnClosing(CancelEventArgs e)
        {
            _onnxSession?.Dispose();
            StopCamera();
            base.OnClosing(e);
        }

        private void BtnExit_Click(object sender, RoutedEventArgs e)
        {
            // 측정 진행 중일 경우 카메라 및 타이머 정지
            StopCamera();

            // ONNX 세션 메모리 해제
            _onnxSession?.Dispose();

            // 애플리케이션 종료
            Application.Current.Shutdown();
        }
    }
}