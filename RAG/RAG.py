import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPVisionModel, Blip2ForConditionalGeneration, AutoTokenizer
import faiss
import numpy as np
from typing import List, Dict, Any, Optional

class ImageDataset(Dataset):
    def __init__(self, image_dir: str, processor: CLIPProcessor):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.processor = processor

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt", padding=True)
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "image_path": img_path
        }

class VisionRAG:
    def __init__(self, 
                 vision_model_name: str = "microsoft/xclip-base-patch32",
                 multimodal_model_name: str = "Salesforce/blip2-flan-t5-xl",
                 index_dim: int = 512,  # X-CLIP的特征维度
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        
        self.device = device
        self.vision_model = CLIPVisionModel.from_pretrained(vision_model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(vision_model_name)
        self.multimodal_model = Blip2ForConditionalGeneration.from_pretrained(
            multimodal_model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(multimodal_model_name)
        
        # 初始化FAISS索引
        self.index = faiss.IndexFlatL2(index_dim)
        self.image_embeddings = {}  # 存储图像路径到嵌入的映射
        self.image_id_to_path = {}  # 存储图像ID到路径的映射
        
    def extract_image_features(self, image: Image.Image) -> torch.Tensor:
        """提取图像特征向量"""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.vision_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
    
    def build_index(self, image_dir: str, batch_size: int = 32, num_workers: int = 4):
        """构建图像向量索引"""
        dataset = ImageDataset(image_dir, self.processor)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        
        current_id = 0
        all_embeddings = []
        
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(self.device)
            image_paths = batch["image_path"]
            
            with torch.no_grad():
                outputs = self.vision_model(pixel_values=pixel_values)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
            for i, embedding in enumerate(embeddings):
                self.image_embeddings[image_paths[i]] = embedding
                self.image_id_to_path[current_id] = image_paths[i]
                current_id += 1
                
            all_embeddings.append(embeddings)
        
        # 构建FAISS索引
        all_embeddings = np.vstack(all_embeddings)
        self.index.add(all_embeddings)
        print(f"成功构建索引，包含 {self.index.ntotal} 个图像向量")
    
    def retrieve_relevant_images(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """检索与文本查询相关的图像"""
        # 使用X-CLIP文本编码器获取查询的文本嵌入
        text_inputs = self.processor(text=query, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            # 注意：X-CLIP的文本编码器可能需要单独加载
            # 这里假设与视觉模型一起加载
            text_features = self.vision_model.get_text_features(**text_inputs).cpu().numpy()
        
        # 检索相似图像
        distances, indices = self.index.search(text_features, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx in self.image_id_to_path:
                results.append({
                    "image_path": self.image_id_to_path[idx],
                    "distance": distances[0][i],
                    "embedding": self.image_embeddings[self.image_id_to_path[idx]]
                })
        return results
    
    def generate_response(self, query: str, retrieved_images: List[Dict[str, Any]], 
                         max_length: int = 512) -> str:
        """结合查询和检索到的图像生成回答"""
        # 加载并处理检索到的图像
        images = []
        for item in retrieved_images:
            image = Image.open(item["image_path"]).convert("RGB")
            images.append(image)
        
        # 为多模态模型准备输入
        inputs = self.processor(images=images, text=query, return_tensors="pt", padding=True).to(self.device)
        
        # 生成回答
        with torch.no_grad():
            outputs = self.multimodal_model.generate(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                max_length=max_length
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def save_index(self, index_path: str):
        """保存FAISS索引"""
        faiss.write_index(self.index, index_path)
        print(f"索引已保存到 {index_path}")
    
    def load_index(self, index_path: str):
        """加载FAISS索引"""
        self.index = faiss.read_index(index_path)
        print(f"已加载索引，包含 {self.index.ntotal} 个图像向量")

# 使用示例
if __name__ == "__main__":
    # 初始化RAG系统，指定X-CLIP模型
    rag_system = VisionRAG(
        vision_model_name="microsoft/xclip-base-patch32",
        index_dim=512  # X-CLIP基础模型的特征维度
    )
    
    # 构建图像索引
    image_directory = "path/to/your/images"
    rag_system.build_index(image_directory)
    
    # 用户查询（支持多种语言）
    user_query = "找出所有含有猫的图片并描述它们的姿态"
    
    # 检索相关图像
    relevant_images = rag_system.retrieve_relevant_images(user_query, k=3)
    
    # 生成回答
    response = rag_system.generate_response(user_query, relevant_images)
    
    print("用户查询:", user_query)
    print("系统回答:", response)
    print("检索到的图像:", [item["image_path"] for item in relevant_images])    