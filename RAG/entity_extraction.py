# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
# import re
# import os


# class TextEntityRelationHandler:
#     def __init__(self, model_path="/root/autodl-tmp/X-CLIP/shibie/qwen7b"):  # 默认使用纯文本模型
#         # 初始化纯文本模型和分词器
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             model_path,
#             trust_remote_code=True
#         )
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_path,
#             torch_dtype=torch.bfloat16,
#             device_map="auto",
#             trust_remote_code=True
#         )
#         self.model.eval()

#         # 提示词：专注于文本内容的实体关系提取
#         self.prompt_template = """
# 请分析下面的文本内容，提取其中的实体和实体间关系：

# 1. 首先识别所有实体（人物、物体、地点、事件等）
# 2. 然后分析实体之间的关系
# 3. 最后必须以三元组列表形式呈现结果，格式为：
# (实体1, 关系, 实体2)
# (实体3, 关系, 实体4)
# ...
# 4. 先详细说明分析过程，最后务必列出所有三元组，不要遗漏

# # 文本内容如下：
# {text}
# """

#     def _extract_triplets(self, text):
#         """提取三元组的方法"""
#         triplet_pattern = r'\(([^,()]+?),\s*([^,()]+?),\s*([^,()]+?)\)'
#         matches = re.findall(triplet_pattern, text)

#         cleaned_triplets = []
#         for triplet in matches:
#             cleaned = tuple([t.strip().strip('"\'') for t in triplet])
#             if all(cleaned):
#                 cleaned_triplets.append(cleaned)

#         return cleaned_triplets

#     def process(self, text, max_new_tokens=512, temperature=0.3):
#         """处理文本输入，提取实体关系"""
#         # 构建提示词
#         formatted_prompt = self.prompt_template.format(text=text)
        
#         # 构建对话格式（适应聊天模型）
#         messages = [
#             {"role": "system", "content": "你是一个实体关系提取专家，擅长从文本中识别实体并提取它们之间的关系。"},
#             {"role": "user", "content": formatted_prompt}
#         ]
        
#         # 转换为模型输入格式
#         text = self.tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True
#         )
#         inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

#         # 生成结果
#         with torch.no_grad():
#             outputs = self.model.generate(
#                 **inputs,
#                 max_new_tokens=max_new_tokens,
#                 temperature=temperature,
#                 do_sample=False,
#                 pad_token_id=self.tokenizer.eos_token_id
#             )

#         # 解码结果
#         response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#         # 提取助手的回答部分
#         assistant_response = response.split("assistant\n")[-1].strip()

#         # 提取三元组
#         triplets = self._extract_triplets(assistant_response)

#         print("\n===== 模型完整响应 =====")
#         print(assistant_response)
#         print("========================\n")

#         return {"thought_chain": assistant_response, "triplets": triplets}


# if __name__ == "__main__":
#     # 确保此处路径指向纯文本模型，如 Qwen-7B-Chat 的正确路径
#     model_path = "/root/autodl-tmp/X-CLIP/shibie/llama-8b"  # 纯文本模型
#     handler = TextEntityRelationHandler(model_path)

#     # 文本任务示例
#     print("===== 纯文本任务 =====")
#     text_prompt = "This video tutorial focuses on demonstrating the Low Cable Chest Flies exercise, featuring a detailed explanation, proper form, and promotional content."
#     text_result = handler.process(text_prompt)
#     print(f"文本内容：{text_prompt}")
#     print("\n思维链分析过程：")
#     print(text_result["thought_chain"])
#     print("\n实体关系三元组：")
#     if text_result["triplets"]:
#         for i, triplet in enumerate(text_result["triplets"], 1):
#             print(f"{i}. {triplet}")
#     else:
#         print("未提取到实体关系三元组")
    

import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import os


class TextEntityRelationHandler:
    def __init__(self, model_path="/root/autodl-tmp/X-CLIP/shibie/qwen7b"):
        # 初始化纯文本模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()

        # 提示词：专注于文本内容的实体关系提取
        self.prompt_template = """
请分析下面的文本内容，提取其中的实体和实体间关系：

1. 首先识别所有实体（人物、物体、地点、事件等）
2. 然后分析实体之间的关系
3. 最后必须以三元组列表形式呈现结果，格式为：
(实体1, 关系, 实体2)
(实体3, 关系, 实体4)
...
4. 先详细说明分析过程，最后务必列出所有三元组，不要遗漏

# 文本内容如下：
{text}
"""

    def _extract_triplets(self, text):
        """提取三元组的方法"""
        triplet_pattern = r'\(([^,()]+?),\s*([^,()]+?),\s*([^,()]+?)\)'
        matches = re.findall(triplet_pattern, text)

        cleaned_triplets = []
        for triplet in matches:
            cleaned = tuple([t.strip().strip('"\'') for t in triplet])
            if all(cleaned):
                cleaned_triplets.append(cleaned)

        return cleaned_triplets

    def process(self, text, max_new_tokens=512, temperature=0.3):
        """处理文本输入，提取实体关系"""
        # 构建提示词
        formatted_prompt = self.prompt_template.format(text=text)
        
        # 构建对话格式（适应聊天模型）
        messages = [
            {"role": "system", "content": "你是一个实体关系提取专家，擅长从文本中识别实体并提取它们之间的关系。"},
            {"role": "user", "content": formatted_prompt}
        ]
        
        # 转换为模型输入格式
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # 生成结果
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # 解码结果
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 提取助手的回答部分
        assistant_response = response.split("assistant\n")[-1].strip()

        # 提取三元组
        triplets = self._extract_triplets(assistant_response)

        return {"thought_chain": assistant_response, "triplets": triplets}


def process_json_file(json_path, model_path, output_path):
    """处理JSON文件中的所有text部分并保存结果"""
    # 加载JSON数据
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 初始化处理器
    handler = TextEntityRelationHandler(model_path)
    
    # 处理每个样本
    results = []
    for item in data:
        sample_name = item.get("sample_name")
        text = item.get("text", "")
        metadata_path = item.get("metadata_path")
        
        print(f"正在处理: {sample_name}")
        try:
            result = handler.process(text)
            results.append({
                "sample_name": sample_name,
                "metadata_path": metadata_path,
                "original_text": text,
                "analysis_result": result
            })
        except Exception as e:
            print(f"处理 {sample_name} 时出错: {str(e)}")
            results.append({
                "sample_name": sample_name,
                "metadata_path": metadata_path,
                "original_text": text,
                "error": str(e)
            })
    
    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"所有处理结果已保存至: {output_path}")


if __name__ == "__main__":
    # 配置路径
    json_file_path = "extracted_samples_texts.json"  # 输入JSON文件路径
    model_path = "/root/autodl-tmp/X-CLIP/shibie/llama-8b"  # 模型路径
    output_file_path = "entity_relation_results.json"  # 输出结果路径
    
    # 处理JSON文件并保存结果
    process_json_file(json_file_path, model_path, output_file_path)