import cv2
import numpy as np
import os
from pathlib import Path
import time
import subprocess
from ultralytics import YOLO
from typing import List, Tuple, Optional

# ===================== 核心配置（可根据需求调整） =====================
# 人脸筛选配置
FACE_SIZE_THRESHOLD = 0.15    # 人脸占比阈值（相对帧宽高）
ALLOWED_FACE_COUNT = 1       # 仅允许单人脸
DET_SCORE_THRESHOLD = 0.7    # 检测置信度阈值
BATCH_SIZE = 8               # 批量处理大小
# 视频处理配置
FRAME_SKIP = 0               # 帧跳过数（0=逐帧检测）
MIN_VALID_DURATION = 5       # 最小合格片段时长（秒）
VIDEO_FPS = 0                # 0=使用原视频FPS
SPEED_PRINT_INTERVAL = 50    # 速度打印间隔（帧）
TOLERANCE_FRAMES = 5         # 连续不合格帧数容错阈值（仅逐帧时生效）

# ===================== 初始化YOLO模型 =====================
def init_yolo_model(model_path: str = "../yolov8l_100e.pt") -> YOLO:
    """初始化YOLOv8人脸检测模型（兼容不同ultralytics版本）"""
    try:
        model = YOLO(model_path)
        # 兼容式获取设备信息（优先用predictor.device，备用torch判断）
        if hasattr(model, 'predictor') and hasattr(model.predictor, 'device'):
            device = model.predictor.device
        else:
            import torch
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # 格式化输出设备信息
        device_str = '0' if str(device).startswith('cuda') else 'cpu'
        print(f"🔧 模型加载成功，使用设备: {device_str}")
        return model
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        raise


# ===================== 核心筛选函数 =====================
def is_high_quality_face(boxes_obj, img_w: int, img_h: int) -> Tuple[bool, str]:
    """
    判断人脸是否符合质量要求（数量+占比）
    :param boxes_obj: ultralytics.engine.results.Boxes 对象（包含坐标、置信度、类别）
    :param img_w: 帧宽度
    :param img_h: 帧高度
    :return: (是否合格, 原因)
    """
    """
    # 测试人脸尺寸
    for i in range(len(scores)):
        x1i, y1i, x2i, y2i = coords[i]
        face_wi = x2i- x1i
        face_hi = y2i - y1i
        face_w_ratioi = face_wi / img_w
        face_h_ratioi = face_hi / img_h
        print(f"    第{i}张人脸占比 face_w_ratio: {face_w_ratioi}; face_h_ratio: {face_h_ratioi}")
    """
    # 1. 空检测结果直接返回不合格
    if len(boxes_obj) == 0:
        return False, "未检测到人脸"
    
    # 2. 第一步：过滤低置信度人脸（置信度阈值优先）
    coords = boxes_obj.xyxy.cpu().numpy()       # 所有检测框坐标 (N,4)
    scores = boxes_obj.conf.cpu().numpy()       # 所有检测框置信度 (N,)
    print(f"    原始检测人脸数: {len(scores)}, 置信度列表: {scores.round(2)}")
    # 测试人脸尺寸
    for i in range(len(scores)):
        x1i, y1i, x2i, y2i = coords[i]
        face_wi = x2i- x1i
        face_hi = y2i - y1i
        face_w_ratioi = face_wi / img_w
        face_h_ratioi = face_hi / img_h
        print("     原始检测人脸的比例")
        print(f"    第{i}张人脸占比 face_w_ratio: {face_w_ratioi}; face_h_ratio: {face_h_ratioi}")
    
    # 过滤低置信度人脸
    conf_mask = scores >= DET_SCORE_THRESHOLD
    conf_valid_coords = coords[conf_mask]       # 置信度达标的坐标
    conf_valid_scores = scores[conf_mask]       # 置信度达标的分数
    if len(conf_valid_coords) == 0:
        return False, f"无置信度达标人脸（阈值={DET_SCORE_THRESHOLD}）"
    
    # 3. 第二步：过滤尺寸不达标人脸（在置信度合格的基础上）
    size_valid_coords = []
    size_valid_scores = []
    for i in range(len(conf_valid_coords)):
        x1i, y1i, x2i, y2i = conf_valid_coords[i]
        face_wi = x2i - x1i
        face_hi = y2i - y1i
        face_w_ratioi = face_wi / img_w
        face_h_ratioi = face_hi / img_h
        print(f"    置信度达标人脸{i} - 占比宽: {face_w_ratioi:.2f}, 占比高: {face_h_ratioi:.2f}")
        
        # 尺寸达标则保留
        if face_w_ratioi >= FACE_SIZE_THRESHOLD and face_h_ratioi >= FACE_SIZE_THRESHOLD:
            size_valid_coords.append(conf_valid_coords[i])
            size_valid_scores.append(conf_valid_scores[i])
        else:
            print(f"    人脸{i}尺寸不达标（阈值={FACE_SIZE_THRESHOLD}），过滤")
    
    # 转换为numpy数组（方便后续处理）
    size_valid_coords = np.array(size_valid_coords)
    size_valid_scores = np.array(size_valid_scores)
    if len(size_valid_coords) == 0:
        return False, f"置信度达标但无尺寸合格人脸（占比阈值={FACE_SIZE_THRESHOLD}）"
    
    # 4. 第三步：判断人脸数量（最后判断数量）
    if len(size_valid_coords) != ALLOWED_FACE_COUNT:
        return False, f"尺寸+置信度达标人脸数={len(size_valid_coords)}（要求{ALLOWED_FACE_COUNT}张）"
    
    # 所有条件达标
    final_face = size_valid_coords[0]
    face_w = final_face[2] - final_face[0]
    face_h = final_face[3] - final_face[1]
    face_w_ratio = face_w / img_w
    face_h_ratio = face_h / img_h
    print(f"    最终合格人脸 - 占比宽: {face_w_ratio:.2f}, 占比高: {face_h_ratio:.2f}, 置信度: {size_valid_scores[0]:.2f}")
    
    return True, "高质量人脸（置信度+尺寸+数量均达标）"


def get_frame_timestamp(frame_idx: int, fps: float) -> float:
    """将帧索引转换为时间戳（秒）"""
    return frame_idx / fps

def cut_video_by_timestamp(input_path: str, output_path: str, start_ts: float, end_ts: float) -> bool:
    """使用ffmpeg裁剪视频（保留音频）"""
    duration = end_ts - start_ts
    if duration < MIN_VALID_DURATION:
        print(f"⚠️  片段时长{duration:.2f}秒 < 最小阈值{MIN_VALID_DURATION}秒，跳过保存")
        return False
    
    cmd = [
        "ffmpeg",
        "-ss", str(start_ts),
        "-i", input_path,
        "-to", str(end_ts),
        "-c:v", "copy",
        "-c:a", "copy",
        "-y",
        "-loglevel", "error",
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ 保存片段：{output_path}（时长：{duration:.2f}秒）")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 裁剪失败：{output_path}，错误：{e}")
        return False

def process_video(video_path: str, output_dir: str = ".", model: YOLO = None) -> None:
    """处理视频（批量检测+片段裁剪）"""
    if model is None:
        model = init_yolo_model()
    
    os.makedirs(output_dir, exist_ok=True)
    video_name = Path(video_path).stem
    cap = cv2.VideoCapture(video_path)
    
    # 获取视频基础信息
    fps = cap.get(cv2.CAP_PROP_FPS) if VIDEO_FPS == 0 else VIDEO_FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / fps if fps > 0 else 0
    
    if total_frames == 0:
        print(f"错误：无法读取视频 {video_path}")
        return
    
    # 初始化变量
    clip_num = 0
    frame_idx = 0
    processed_frames = 0
    start_time = time.time()
    valid_clip_start_ts: Optional[float] = None
    consecutive_invalid = 0
    batch_frames: List[np.ndarray] = []
    batch_indices: List[int] = []  # 记录批次中帧的原始索引
    
    print(f"📽️  开始处理：{video_path}")
    print(f"📊 视频信息：FPS={fps:.2f}, 分辨率={width}x{height}, 总帧数={total_frames}, 总时长={total_duration:.2f}秒")
    print(f"⚙️  配置：置信度={DET_SCORE_THRESHOLD}, 批量大小={BATCH_SIZE}, 跳帧数={FRAME_SKIP}, 最小片段时长={MIN_VALID_DURATION}秒")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 帧跳过处理
        if frame_idx % (FRAME_SKIP + 1) != 0:
            frame_idx += 1
            continue
        
        # 收集帧到批次
        batch_frames.append(frame)
        batch_indices.append(frame_idx)
        
        # 批次满了或最后一批不足批量大小时进行检测
        if len(batch_frames) >= BATCH_SIZE:
            # 批量检测
            results = model(batch_frames, verbose=False)
            # print(f"batch_frames' results: \n{results}")
            
            # 处理批次结果
            for i, (result, frame_idx_batch) in enumerate(zip(results, batch_indices)):
                # boxes = result.boxes.xyxy.cpu().numpy()  # 格式：[x1, y1, x2, y2, score]
                # print(f"result: \n{result}")
                # print(f"result.boxes: \n{result.boxes}")
                is_valid, reason = is_high_quality_face(result.boxes, width, height)
                
                # 处理片段逻辑
                current_ts = get_frame_timestamp(frame_idx_batch, fps)
                
                if is_valid:
                    consecutive_invalid = 0
                    if valid_clip_start_ts is None:
                        valid_clip_start_ts = current_ts
                        print(f"🔄 开始合格片段：帧{frame_idx_batch}（时间戳={valid_clip_start_ts:.2f}秒）")
                else:
                    # 处理不合格帧
                    if FRAME_SKIP == 0:  # 逐帧模式
                        consecutive_invalid += 1
                        if valid_clip_start_ts is not None and consecutive_invalid > TOLERANCE_FRAMES:
                            end_ts = get_frame_timestamp(frame_idx_batch - consecutive_invalid, fps)
                            output_path = os.path.join(output_dir, f"{video_name}_croped{clip_num}.mp4")
                            print(f"裁剪命令：开始时间={valid_clip_start_ts:.2f}秒，结束时间={end_ts:.2f}秒")
                            if cut_video_by_timestamp(video_path, output_path, valid_clip_start_ts, end_ts):
                                clip_num += 1
                            valid_clip_start_ts = None
                            consecutive_invalid = 0
                            print(f"🔚 结束合格片段：帧{frame_idx_batch}（时间戳={current_ts:.2f}秒），原因：{reason}")
                    else:  # 跳帧模式
                        if valid_clip_start_ts is not None:
                            # 以上一个检测帧时间作为结束
                            end_ts = get_frame_timestamp(batch_indices[i-1] if i > 0 else frame_idx_batch, fps)
                            output_path = os.path.join(output_dir, f"{video_name}_croped{clip_num}.mp4")
                            if cut_video_by_timestamp(video_path, output_path, valid_clip_start_ts, end_ts):
                                clip_num += 1
                            valid_clip_start_ts = None
                            print(f"🔚 结束合格片段：帧{frame_idx_batch}（时间戳={current_ts:.2f}秒），原因：{reason}")
                
                # 打印帧信息
                status = "✅" if is_valid else "❌"
                print(f"帧{frame_idx_batch} {status} - {reason}")
                
                processed_frames += 1
                # 速度统计
                if processed_frames % SPEED_PRINT_INTERVAL == 0:
                    elapsed = time.time() - start_time
                    speed = processed_frames / elapsed
                    print(f"📈 已处理{processed_frames}帧，速度：{speed:.2f}帧/秒")
            
            # 重置批次
            batch_frames = []
            batch_indices = []
        
        frame_idx += 1

    # 处理最后一批剩余帧
    if batch_frames:
        results = model(batch_frames, verbose=False)
        for i, (result, frame_idx_batch) in enumerate(zip(results, batch_indices)):
            is_valid, reason = is_high_quality_face(result.boxes, width, height)
            current_ts = get_frame_timestamp(frame_idx_batch, fps)
            
            if is_valid and valid_clip_start_ts is None:
                valid_clip_start_ts = current_ts
            
            processed_frames += 1

    # 处理最后一段合格片段
    if valid_clip_start_ts is not None:
        output_path = os.path.join(output_dir, f"{video_name}_croped{clip_num}.mp4")
        cut_video_by_timestamp(video_path, output_path, valid_clip_start_ts, total_duration)
        clip_num += 1

    # 收尾统计
    total_elapsed = time.time() - start_time
    avg_speed = processed_frames / total_elapsed if total_elapsed > 0 else 0
    print(f"\n🏁 处理完成！")
    print(f"⏱️  总耗时：{total_elapsed:.2f}秒，平均速度：{avg_speed:.2f}帧/秒")
    print(f"📦 生成合格片段数：{clip_num}（保存路径：{os.path.abspath(output_dir)}）")

    cap.release()

# ===================== 主函数 =====================
if __name__ == "__main__":
    test_video_path = "24494339-1-192.mp4"
    output_directory = "./output/24494339-1-192_yolov8l_100e_test3"
    
    # cut_video_by_timestamp(test_video_path,'./test.mp4',20,30)
    # # 检查ffmpeg是否可用
    # try:
    #     subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    # except (subprocess.CalledProcessError, FileNotFoundError):
    #     print(f"❌ 未找到FFmpeg，请确保已安装并加入环境变量")
    #     exit(1)
    
    # # 初始化模型并处理视频
    # model = init_yolo_model()
    # process_video(test_video_path, output_directory, model)
