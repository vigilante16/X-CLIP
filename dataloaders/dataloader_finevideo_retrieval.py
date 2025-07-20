from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import json
import numpy as np
from torch.utils.data import Dataset
from dataloaders.rawvideo_util import RawVideoExtractor  # 复用原视频提取工具


class FineVideo_DataLoader(Dataset):
    """新数据集的数据加载器（视频-文本检索任务）"""
    def __init__(
            self,
            subset,
            data_root,  # 数据集根目录（包含videos和metadata子文件夹）
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
            frame_order=0,  # 0:正常顺序, 1:逆序, 2:随机顺序
            slice_framepos=0,  # 0:从头取帧, 1:从尾取帧, 2:均匀采样
    ):
        # 基本参数配置
        self.subset = subset
        self.data_root = data_root
        self.videos_dir = os.path.join(data_root, "videos_compressed")  # 视频文件夹路径
        self.metadata_dir = os.path.join(data_root, "metadata")  # 元数据文件夹路径
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.max_frames = max_frames
        self.frame_order = frame_order
        self.slice_framepos = slice_framepos
        self.feature_framerate = feature_framerate
        self.image_resolution = image_resolution

        # 验证文件夹存在
        assert os.path.exists(self.videos_dir), f"视频文件夹不存在: {self.videos_dir}"
        assert os.path.exists(self.metadata_dir), f"元数据文件夹不存在: {self.metadata_dir}"
        assert self.subset in ["train", "val", "test"], "subset必须为train/val/test"

        # 读取子集划分（用户需提前准备，格式同MSVD：每行一个视频ID）
        self.subset_file = os.path.join(data_root, f"splits/{subset}_list.txt")
        assert os.path.exists(self.subset_file), f"子集划分文件不存在: {self.subset_file}"
        with open(self.subset_file, 'r') as f:
            self.video_ids = [line.strip() for line in f.readlines()]  # 视频ID列表（不含扩展名）

        # 建立视频路径映射（视频ID -> 视频文件路径）
        self.video_path_dict = {}
        for video_fn in os.listdir(self.videos_dir):
            if video_fn.endswith(".mp4"):
                video_id = os.path.splitext(video_fn)[0]  # 从文件名提取视频ID（去除.mp4）
                if video_id in self.video_ids:
                    self.video_path_dict[video_id] = os.path.join(self.videos_dir, video_fn)

        # 读取元数据并提取文本描述（构建视频-文本对）
        self.sentences_dict = {}  # 索引 -> (video_id, caption)
        self.cut_off_points = []  # 用于多句子检索的标签分割点
        for video_id in self.video_ids:
            # 读取对应元数据json
            meta_fn = f"{video_id}.json"
            meta_path = os.path.join(self.metadata_dir, meta_fn)
            assert os.path.exists(meta_path), f"元数据文件不存在: {meta_path}"
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)

            # 从元数据中提取文本描述（可根据需求扩展）
            captions = []
            # 1. 视频整体描述
            overall_desc = meta["content_metadata"].get("description", "")
            if overall_desc:
                captions.append(overall_desc)
            # 2. 场景活动描述
            for scene in meta["content_metadata"].get("scenes", []):
                for activity in scene.get("activities", []):
                    act_desc = activity.get("description", "")
                    if act_desc:
                        captions.append(act_desc)
            # 3. Q&A中的答案（可选）
            for qa in meta["content_metadata"].get("qAndA", []):
                ans = qa.get("answer", "")
                if ans:
                    captions.append(ans)

            # 构建文本索引映射
            for cap in captions:
                self.sentences_dict[len(self.sentences_dict)] = (video_id, cap.strip())
            # 记录当前视频的文本结束索引（用于多句子检索）
            self.cut_off_points.append(len(self.sentences_dict))

        # 多句子检索相关参数
        self.multi_sentence_per_video = True
        if self.subset in ["val", "test"]:
            self.sentence_num = len(self.sentences_dict)
            self.video_num = len(self.video_ids)
            assert len(self.cut_off_points) == self.video_num, "分割点数量与视频数量不匹配"
            print(f"[{subset}] 文本数量: {self.sentence_num}, 视频数量: {self.video_num}")

        # 视频提取器初始化
        self.raw_video_extractor = RawVideoExtractor(
            framerate=feature_framerate,
            size=image_resolution
        )
        # 文本特殊标记
        self.SPECIAL_TOKEN = {
            "CLS_TOKEN": "<|startoftext|>",
            "SEP_TOKEN": "<|endoftext|>",
            "PAD_TOKEN": "[PAD]"
        }

        print(f"加载完成: 视频{len(self.video_path_dict)}个, 文本-视频对{len(self.sentences_dict)}个")

    def __len__(self):
        return len(self.sentences_dict)

    def _process_text(self, caption):
        """处理单条文本，确保输出长度严格等于max_words，且格式正确"""
        # 处理空文本
        if not caption or not isinstance(caption, str) or caption.strip() == "":
            caption = "unknown"  # 替换为空文本的默认值
        
        # 1. 分词并添加特殊标记
        words = self.tokenizer.tokenize(caption)
        
        # 检查分词结果是否异常
        if not words:
            words = ["unknown"]  # 若分词结果为空，使用默认token
        
        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
        total_length = self.max_words - 1
        
        # 2. 截断到最大长度
        if len(words) > total_length:
            words = words[:total_length]
        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
        
        # 3. 转换为ID
        try:
            input_ids = self.tokenizer.convert_tokens_to_ids(words)
        except Exception as e:
            print(f"警告: 分词器转换失败，文本: {caption}")
            # 使用默认token
            default_token = self.tokenizer.encoder.get("unknown", 0)
            input_ids = [default_token] * self.max_words
        
        # 4. 验证input_ids是否全部为整数
        if not all(isinstance(i, int) for i in input_ids):
            print(f"警告: 发现非整数token ID，文本: {caption}")
            # 替换非整数元素
            input_ids = [i if isinstance(i, int) else 0 for i in input_ids]
        
        # 5. 创建mask和segment_ids
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)
        
        # 6. 强制填充到max_words
        if len(input_ids) < self.max_words:
            pad_id = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKEN["PAD_TOKEN"])
            # 确保pad_id是单个整数
            if not isinstance(pad_id, int):
                pad_id = 0  # 使用默认pad ID
            
            pad_length = self.max_words - len(input_ids)
            input_ids += [pad_id] * pad_length
            input_mask += [0] * pad_length
            segment_ids += [0] * pad_length
        else:
            input_ids = input_ids[:self.max_words]
            input_mask = input_mask[:self.max_words]
            segment_ids = segment_ids[:self.max_words]
        
        # 最终验证
        assert len(input_ids) == self.max_words, f"input_ids长度错误: {len(input_ids)}"
        assert len(input_mask) == self.max_words, f"input_mask长度错误: {len(input_mask)}"
        assert len(segment_ids) == self.max_words, f"segment_ids长度错误: {len(segment_ids)}"
        
        return np.array(input_ids), np.array(input_mask), np.array(segment_ids)

    def _get_text(self, video_id, caption):
        """获取文本特征（适配批量处理格式）"""
        # 此处k=1表示每个样本只取1条文本（可扩展为多文本对比）
        k = 1
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        # 处理文本
        input_ids, input_mask, segment_ids = self._process_text(caption)
        pairs_text[0] = input_ids
        pairs_mask[0] = input_mask
        pairs_segment[0] = segment_ids

        return pairs_text, pairs_mask, pairs_segment, [video_id]

    def _get_video(self, video_id):
        """提取视频帧特征"""
        video_path = self.video_path_dict[video_id]
        try:
            raw_video_data = self.raw_video_extractor.get_video_data(video_path)
        except Exception as e:
            print(f"视频{video_id}解码失败: {e}，跳过该视频")
            # 返回空视频（需确保模型能处理空输入，或在DataLoader中过滤）
            return np.zeros((1, self.max_frames, 3, self.image_resolution, self.image_resolution)), np.zeros((1, self.max_frames))
        # 调用视频提取器获取原始帧数据
        raw_video = raw_video_data["video"]  # 格式: [L, T, 3, H, W]（L为片段数，T为每片段帧数）

        # 处理帧序列（截断/采样到max_frames）
        if len(raw_video.shape) > 3:
            # 合并片段（取第一个片段或直接合并）
            raw_video_slice = raw_video[0] if raw_video.shape[0] > 0 else raw_video
            # 根据策略截取帧
            if self.max_frames < raw_video_slice.shape[0]:
                if self.slice_framepos == 0:
                    video_slice = raw_video_slice[:self.max_frames, ...]  # 从头取
                elif self.slice_framepos == 1:
                    video_slice = raw_video_slice[-self.max_frames:, ...]  # 从尾取
                else:
                    # 均匀采样
                    sample_idx = np.linspace(0, raw_video_slice.shape[0]-1, self.max_frames, dtype=int)
                    video_slice = raw_video_slice[sample_idx, ...]
            else:
                video_slice = raw_video_slice  # 不足则保留全部

            # 调整帧顺序（数据增强）
            video_slice = self.raw_video_extractor.process_frame_order(video_slice, self.frame_order)
            slice_len = video_slice.shape[0]
        else:
            # 视频读取失败
            print(f"警告: 视频{video_id}读取失败，路径: {video_path}")
            video_slice = np.zeros((0, 3, self.image_resolution, self.image_resolution))
            slice_len = 0

        # 构建视频张量和掩码
        video = np.zeros((1, self.max_frames, 1, 3, self.image_resolution, self.image_resolution), dtype=np.float32)
        video_mask = np.zeros((1, self.max_frames), dtype=np.long)
        if slice_len > 0:
            video[0, :slice_len, ...] = video_slice
            video_mask[0, :slice_len] = 1

        return video, video_mask

    def __getitem__(self, idx):
        """获取索引对应的样本"""
        video_id, caption = self.sentences_dict[idx]
        # 获取文本特征
        pairs_text, pairs_mask, pairs_segment, video_ids = self._get_text(video_id, caption)
        # 获取视频特征
        video, video_mask = self._get_video(video_id)
        return pairs_text, pairs_mask, pairs_segment, video, video_mask