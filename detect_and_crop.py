import cv2
import numpy as np
import os
from pathlib import Path
import time
import subprocess
from ultralytics import YOLO
from typing import List, Tuple, Optional, Dict, Any
import json

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

# 裁剪配置
CROP_PADDING_RATIO = 0.1     # 裁剪框额外填充比例（10%）
MIN_CROP_SIZE = 64           # 最小裁剪尺寸（像素）

# ===================== 初始化YOLO模型 =====================
def init_yolo_model(model_path: str = "yolov8l_100e.pt") -> YOLO:
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
def is_high_quality_face(boxes_obj, img_w: int, img_h: int) -> Tuple[bool, str, Optional[np.ndarray]]:
    """
    判断人脸是否符合质量要求（数量+占比）
    :param boxes_obj: ultralytics.engine.results.Boxes 对象（包含坐标、置信度、类别）
    :param img_w: 帧宽度
    :param img_h: 帧高度
    :return: (是否合格, 原因, 检测框坐标[x1,y1,x2,y2]或None)
    """
    # 1. 空检测结果直接返回不合格
    if len(boxes_obj) == 0:
        return False, "未检测到人脸", None
    
    # 2. 第一步：过滤低置信度人脸（置信度阈值优先）
    coords = boxes_obj.xyxy.cpu().numpy()       # 所有检测框坐标 (N,4)
    scores = boxes_obj.conf.cpu().numpy()       # 所有检测框置信度 (N,)
    print(f"    原始检测人脸数: {len(scores)}, 置信度列表: {scores.round(2)}")
    
    # 过滤低置信度人脸
    conf_mask = scores >= DET_SCORE_THRESHOLD
    conf_valid_coords = coords[conf_mask]       # 置信度达标的坐标
    conf_valid_scores = scores[conf_mask]       # 置信度达标的分数
    if len(conf_valid_coords) == 0:
        return False, f"无置信度达标人脸（阈值={DET_SCORE_THRESHOLD}）", None
    
    # 3. 第二步：过滤尺寸不达标人脸（在置信度合格的基础上）
    size_valid_coords = []
    size_valid_scores = []
    for i in range(len(conf_valid_coords)):
        x1i, y1i, x2i, y2i = conf_valid_coords[i]
        face_wi = x2i - x1i
        face_hi = y2i - y1i
        face_w_ratioi = face_wi / img_w
        face_h_ratioi = face_hi / img_h
        
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
        return False, f"置信度达标但无尺寸合格人脸（占比阈值={FACE_SIZE_THRESHOLD}）", None
    
    # 4. 第三步：判断人脸数量（最后判断数量）
    if len(size_valid_coords) != ALLOWED_FACE_COUNT:
        return False, f"尺寸+置信度达标人脸数={len(size_valid_coords)}（要求{ALLOWED_FACE_COUNT}张）", None
    
    # 所有条件达标
    final_face = size_valid_coords[0]
    return True, "高质量人脸（置信度+尺寸+数量均达标）", final_face


def get_frame_timestamp(frame_idx: int, fps: float) -> float:
    """将帧索引转换为时间戳（秒）"""
    return frame_idx / fps


def calculate_max_bbox(bbox_list: List[np.ndarray], img_width: int, img_height: int) -> Tuple[int, int, int, int]:
    """
    计算一系列检测框的最大边界框
    :param bbox_list: 检测框列表，每个为[x1, y1, x2, y2]
    :param img_width: 图像宽度
    :param img_height: 图像高度
    :return: 最大边界框 (x1, y1, x2, y2)
    """
    if not bbox_list:
        return 0, 0, img_width, img_height
    
    # 将所有坐标堆叠
    all_coords = np.vstack(bbox_list)
    
    # 计算最小值和最大值
    min_x = int(np.min(all_coords[:, 0]))
    min_y = int(np.min(all_coords[:, 1]))
    max_x = int(np.max(all_coords[:, 2]))
    max_y = int(np.max(all_coords[:, 3]))
    
    # 计算原始宽高
    width = max_x - min_x
    height = max_y - min_y
    
    # 添加填充
    padding_x = int(width * CROP_PADDING_RATIO)
    padding_y = int(height * CROP_PADDING_RATIO)
    
    # 应用填充并确保边界在图像内
    x1 = max(0, min_x - padding_x)
    y1 = max(0, min_y - padding_y)
    x2 = min(img_width, max_x + padding_x)
    y2 = min(img_height, max_y + padding_y)
    
    # 确保最小尺寸
    crop_width = x2 - x1
    crop_height = y2 - y1
    
    if crop_width < MIN_CROP_SIZE:
        diff = MIN_CROP_SIZE - crop_width
        x1 = max(0, x1 - diff // 2)
        x2 = min(img_width, x2 + diff - diff // 2)
    
    if crop_height < MIN_CROP_SIZE:
        diff = MIN_CROP_SIZE - crop_height
        y1 = max(0, y1 - diff // 2)
        y2 = min(img_height, y2 + diff - diff // 2)
    
    return x1, y1, x2, y2


def crop_video_by_timestamp_with_bbox(
    input_path: str, 
    output_path: str, 
    start_ts: float, 
    end_ts: float,
    bbox_list: List[np.ndarray],
    original_width: int,
    original_height: int
) -> bool:
    """
    使用ffmpeg裁剪视频并应用最大边界框裁剪
    :param input_path: 输入视频路径
    :param output_path: 输出视频路径
    :param start_ts: 开始时间（秒）
    :param end_ts: 结束时间（秒）
    :param bbox_list: 该时间段内所有检测框列表
    :param original_width: 原始视频宽度
    :param original_height: 原始视频高度
    :return: 是否成功
    """
    duration = end_ts - start_ts
    if duration < MIN_VALID_DURATION:
        print(f"⚠️  片段时长{duration:.2f}秒 < 最小阈值{MIN_VALID_DURATION}秒，跳过保存")
        return False
    
    # 计算最大边界框
    x1, y1, x2, y2 = calculate_max_bbox(bbox_list, original_width, original_height)
    width = x2 - x1
    height = y2 - y1
    
    print(f"    最大边界框: ({x1}, {y1}, {x2}, {y2}), 尺寸: {width}x{height}")
    
    # 构建裁剪命令
    crop_filter = f"crop={width}:{height}:{x1}:{y1}"
    
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-ss", str(start_ts),
        "-to", str(end_ts),
        "-filter:v", crop_filter,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-preset", "medium",
        "-crf", "23",
        "-y",
        "-loglevel", "error",
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ 保存裁剪片段：{output_path}")
        print(f"   📏 裁剪区域: {width}x{height}, 位置: ({x1}, {y1})")
        print(f"   ⏱️  时长: {duration:.2f}秒 ({start_ts:.2f}秒 - {end_ts:.2f}秒)")
        
        # 保存裁剪信息到JSON文件（可选）
        info_path = output_path.replace('.mp4', '_info.json')
        crop_info = {
            "original_video": input_path,
            "output_video": output_path,
            "start_time": start_ts,
            "end_time": end_ts,
            "duration": duration,
            "crop_region": {
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "width": width, "height": height
            },
            "num_frames_with_faces": len(bbox_list),
            "padding_ratio": CROP_PADDING_RATIO
        }
        with open(info_path, 'w') as f:
            json.dump(crop_info, f, indent=2)
        
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
    
    # 用于跟踪当前片段
    valid_clip_start_ts: Optional[float] = None
    valid_clip_bboxes: List[np.ndarray] = []  # 记录当前片段的所有检测框
    valid_clip_frame_indices: List[int] = []  # 记录当前片段的帧索引
    
    consecutive_invalid = 0
    batch_frames: List[np.ndarray] = []
    batch_indices: List[int] = []  # 记录批次中帧的原始索引
    
    print(f"📽️  开始处理：{video_path}")
    print(f"📊 视频信息：FPS={fps:.2f}, 分辨率={width}x{height}, 总帧数={total_frames}, 总时长={total_duration:.2f}秒")
    print(f"⚙️  配置：置信度={DET_SCORE_THRESHOLD}, 批量大小={BATCH_SIZE}, 跳帧数={FRAME_SKIP}, 最小片段时长={MIN_VALID_DURATION}秒")
    print(f"✂️  裁剪配置：填充比例={CROP_PADDING_RATIO}, 最小裁剪尺寸={MIN_CROP_SIZE}像素")

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
            
            # 处理批次结果
            for i, (result, frame_idx_batch) in enumerate(zip(results, batch_indices)):
                is_valid, reason, bbox = is_high_quality_face(result.boxes, width, height)
                
                # 处理片段逻辑
                current_ts = get_frame_timestamp(frame_idx_batch, fps)
                
                if is_valid:
                    consecutive_invalid = 0
                    if valid_clip_start_ts is None:
                        # 开始新的合格片段
                        valid_clip_start_ts = current_ts
                        valid_clip_bboxes = [bbox]  # 添加第一个检测框
                        valid_clip_frame_indices = [frame_idx_batch]
                        print(f"🔄 开始合格片段：帧{frame_idx_batch}（时间戳={valid_clip_start_ts:.2f}秒）")
                    else:
                        # 继续当前片段，记录检测框
                        valid_clip_bboxes.append(bbox)
                        valid_clip_frame_indices.append(frame_idx_batch)
                else:
                    # 处理不合格帧
                    if FRAME_SKIP == 0:  # 逐帧模式
                        consecutive_invalid += 1
                        if valid_clip_start_ts is not None and consecutive_invalid > TOLERANCE_FRAMES:
                            # 结束当前片段
                            end_frame_idx = valid_clip_frame_indices[-1]  # 最后一个合格帧
                            end_ts = get_frame_timestamp(end_frame_idx, fps)
                            
                            # 生成输出路径
                            output_path = os.path.join(output_dir, f"{video_name}_face_crop_{clip_num}.mp4")
                            
                            # 使用最大边界框裁剪视频
                            print(f"🎬 裁剪片段{clip_num}：开始时间={valid_clip_start_ts:.2f}秒，结束时间={end_ts:.2f}秒")
                            print(f"    合格帧数：{len(valid_clip_bboxes)}，检测框数量：{len(valid_clip_bboxes)}")
                            
                            if crop_video_by_timestamp_with_bbox(
                                video_path, output_path, valid_clip_start_ts, end_ts,
                                valid_clip_bboxes, width, height
                            ):
                                clip_num += 1
                            
                            # 重置片段状态
                            valid_clip_start_ts = None
                            valid_clip_bboxes = []
                            valid_clip_frame_indices = []
                            consecutive_invalid = 0
                            print(f"🔚 结束合格片段：帧{frame_idx_batch}（时间戳={current_ts:.2f}秒），原因：{reason}")
                    else:  # 跳帧模式
                        if valid_clip_start_ts is not None:
                            # 结束当前片段
                            end_frame_idx = valid_clip_frame_indices[-1]  # 最后一个合格帧
                            end_ts = get_frame_timestamp(end_frame_idx, fps)
                            
                            output_path = os.path.join(output_dir, f"{video_name}_face_crop_{clip_num}.mp4")
                            
                            print(f"🎬 裁剪片段{clip_num}：开始时间={valid_clip_start_ts:.2f}秒，结束时间={end_ts:.2f}秒")
                            print(f"    合格帧数：{len(valid_clip_bboxes)}，检测框数量：{len(valid_clip_bboxes)}")
                            
                            if crop_video_by_timestamp_with_bbox(
                                video_path, output_path, valid_clip_start_ts, end_ts,
                                valid_clip_bboxes, width, height
                            ):
                                clip_num += 1
                            
                            valid_clip_start_ts = None
                            valid_clip_bboxes = []
                            valid_clip_frame_indices = []
                            print(f"🔚 结束合格片段：帧{frame_idx_batch}（时间戳={current_ts:.2f}秒），原因：{reason}")
                
                # 打印帧信息
                status = "✅" if is_valid else "❌"
                if is_valid:
                    bbox_str = f" [{int(bbox[0])},{int(bbox[1])},{int(bbox[2])},{int(bbox[3])}]"
                else:
                    bbox_str = ""
                print(f"帧{frame_idx_batch} {status} - {reason}{bbox_str}")
                
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
            is_valid, reason, bbox = is_high_quality_face(result.boxes, width, height)
            current_ts = get_frame_timestamp(frame_idx_batch, fps)
            
            if is_valid:
                if valid_clip_start_ts is None:
                    valid_clip_start_ts = current_ts
                    valid_clip_bboxes = [bbox]
                    valid_clip_frame_indices = [frame_idx_batch]
                else:
                    valid_clip_bboxes.append(bbox)
                    valid_clip_frame_indices.append(frame_idx_batch)
            
            processed_frames += 1

    # 处理最后一段合格片段
    if valid_clip_start_ts is not None and len(valid_clip_bboxes) > 0:
        end_frame_idx = valid_clip_frame_indices[-1]
        end_ts = get_frame_timestamp(end_frame_idx, fps)
        output_path = os.path.join(output_dir, f"{video_name}_face_crop_{clip_num}.mp4")
        
        print(f"🎬 裁剪最后一段片段{clip_num}：开始时间={valid_clip_start_ts:.2f}秒，结束时间={end_ts:.2f}秒")
        print(f"    合格帧数：{len(valid_clip_bboxes)}，检测框数量：{len(valid_clip_bboxes)}")
        
        if crop_video_by_timestamp_with_bbox(
            video_path, output_path, valid_clip_start_ts, end_ts,
            valid_clip_bboxes, width, height
        ):
            clip_num += 1

    # 收尾统计
    total_elapsed = time.time() - start_time
    avg_speed = processed_frames / total_elapsed if total_elapsed > 0 else 0
    print(f"\n🏁 处理完成！")
    print(f"⏱️  总耗时：{total_elapsed:.2f}秒，平均速度：{avg_speed:.2f}帧/秒")
    print(f"📦 生成面部特写片段数：{clip_num}")
    print(f"📁 保存路径：{os.path.abspath(output_dir)}")

    cap.release()


# ===================== 主函数 =====================
if __name__ == "__main__":
    test_video_path = "./test/24494339-1-192.mp4"
    output_directory = "./output/24494339-1-192_face_crops"
    model = init_yolo_model()
    process_video(test_video_path, output_directory, model)
