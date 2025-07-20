import os
import random
from typing import List, Dict, Optional

class DatasetSplitter:
    """将数据集按比例划分为训练集、验证集和测试集"""
    
    def __init__(self, 
                 data_dir: str,
                 output_dir: str,
                 video_prefix: str = "sample_",
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 seed: int = 42):
        """
        初始化数据集划分器
        
        参数:
            data_dir: 数据集根目录，包含videos和metadata文件夹
            output_dir: 划分文件输出目录
            video_prefix: 视频ID前缀
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            seed: 随机数种子，确保划分可重复
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.video_prefix = video_prefix
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置随机种子
        random.seed(seed)
        
        # 检查目录结构
        self.videos_dir = os.path.join(data_dir, "videos")
        self.metadata_dir = os.path.join(data_dir, "metadata")
        
        if not os.path.exists(self.videos_dir):
            raise ValueError(f"视频文件夹不存在: {self.videos_dir}")
        
        if not os.path.exists(self.metadata_dir):
            raise ValueError(f"元数据文件夹不存在: {self.metadata_dir}")
    
    def get_all_video_ids(self) -> List[str]:
        """获取所有视频ID"""
        video_ids = []
        
        # 从视频文件夹获取ID
        for filename in os.listdir(self.videos_dir):
            if filename.endswith(".mp4"):
                # 提取ID，例如从sample_0.mp4提取sample_0
                video_id = filename.rsplit('.', 1)[0]
                if video_id.startswith(self.video_prefix):
                    video_ids.append(video_id)
        
        # 从元数据文件夹获取ID（双重检查）
        for filename in os.listdir(self.metadata_dir):
            if filename.endswith(".json"):
                video_id = filename.rsplit('.', 1)[0]
                if video_id.startswith(self.video_prefix):
                    if video_id not in video_ids:
                        print(f"警告: 元数据中存在但视频文件不存在的ID: {video_id}")
        
        print(f"找到 {len(video_ids)} 个视频ID")
        return video_ids
    
    def split_dataset(self, video_ids: List[str]) -> Dict[str, List[str]]:
        """
        按比例划分数据集
        
        返回:
            字典，包含三个键：'train', 'val', 'test'，对应划分后的ID列表
        """
        # 打乱顺序
        shuffled_ids = sorted(video_ids)  # 先排序确保结果可重复
        random.shuffle(shuffled_ids)
        
        # 计算各集合大小
        total = len(shuffled_ids)
        train_size = int(total * self.train_ratio)
        val_size = int(total * self.val_ratio)
        
        # 划分数据集
        splits = {
            'train': shuffled_ids[:train_size],
            'val': shuffled_ids[train_size:train_size+val_size],
            'test': shuffled_ids[train_size+val_size:]
        }
        
        print(f"数据集划分结果:")
        print(f"  训练集: {len(splits['train'])} 个样本")
        print(f"  验证集: {len(splits['val'])} 个样本")
        print(f"  测试集: {len(splits['test'])} 个样本")
        
        return splits
    
    def write_split_files(self, splits: Dict[str, List[str]]) -> None:
        """
        将划分结果写入文件
        
        参数:
            splits: 划分结果字典
        """
        for subset, ids in splits.items():
            output_file = os.path.join(self.output_dir, f"{subset}_list.txt")
            
            with open(output_file, 'w') as f:
                for video_id in ids:
                    f.write(f"{video_id}\n")
            
            print(f"已写入 {subset} 集文件: {output_file}, 包含 {len(ids)} 个样本")
    
    def run(self) -> None:
        """执行数据集划分流程"""
        # 获取所有视频ID
        video_ids = self.get_all_video_ids()
        
        # 划分数据集
        splits = self.split_dataset(video_ids)
        
        # 写入划分文件
        self.write_split_files(splits)

def main():
    # 使用示例
    data_dir = "./"  # 包含videos和metadata的目录
    output_dir = os.path.join(data_dir, "splits")  # 划分文件输出目录
    
    splitter = DatasetSplitter(
        data_dir=data_dir,
        output_dir=output_dir,
        video_prefix="sample_",
        train_ratio=0.9,
        val_ratio=0.05,
        test_ratio=0.05,
        seed=42
    )
    
    splitter.run()

if __name__ == "__main__":
    main()    