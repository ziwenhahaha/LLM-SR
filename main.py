import os
from argparse import ArgumentParser
import numpy as np
import torch
import pandas as pd
import json

from llmsr import pipeline
from llmsr import config
from llmsr import sampler
from llmsr import evaluator


def str2bool(v):
    """将字符串转换为布尔值，用于argparse"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')


# 命令行参数解析
parser = ArgumentParser(description='Scientific Intelligent Modelling using LLMs')
parser.add_argument('--port', type=int, default=None, help='Port for local LLM server')
parser.add_argument('--use_api', type=str2bool, default=False, help='Whether to use API models instead of local models')
parser.add_argument('--api_model', type=str, default="gpt-3.5-turbo", 
                    help='API model to use (e.g., gpt-4, anthropic/claude-3, google/gemini-pro, deepseek/deepseek-chat)')
parser.add_argument('--spec_path', type=str, required=True, help='Path to prompt specification file')
parser.add_argument('--log_path', type=str, default="./logs/oscillator1", help='Directory for logs')
parser.add_argument('--problem_name', type=str, default="oscillator1", help='Problem name')
parser.add_argument('--run_id', type=int, default=1, help='Run identifier')
parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for sampling')
parser.add_argument('--api_key', type=str, default=None, help='API key for the specified model')
parser.add_argument('--api_params', type=str, default=None, 
                    help='Additional API parameters as JSON string (e.g., \'{"top_p": 0.95, "max_tokens": 1000}\')')
parser.add_argument('--batch_inference', type=str2bool, default=True, help='Whether to use batch inference for local models')
parser.add_argument('--api_base', type=str, default=None, help='Custom API base URL')
parser.add_argument('--debug', type=str2bool, default=False, help='Enable verbose debugging output')
parser.add_argument('--samples_per_prompt', type=int, default=5, help='Number of samples to generate per prompt')
parser.add_argument('--max_samples', type=int, default=10000, help='Maximum number of samples to generate')

args = parser.parse_args()


if __name__ == '__main__':
    # 设置API密钥（如果提供）
    if args.api_key:
        # 根据模型名称自动设置适当的环境变量
        if args.api_model.startswith('gpt-') or 'openai' in args.api_model:
            os.environ["OPENAI_API_KEY"] = args.api_key
        elif 'anthropic' in args.api_model or 'claude' in args.api_model:
            os.environ["ANTHROPIC_API_KEY"] = args.api_key
        elif 'google' in args.api_model or 'gemini' in args.api_model:
            os.environ["GOOGLE_API_KEY"] = args.api_key
        elif 'mistral' in args.api_model:
            os.environ["MISTRAL_API_KEY"] = args.api_key
        elif 'deepseek' in args.api_model:
            os.environ["DEEPSEEK_API_KEY"] = args.api_key
        else:
            # 默认设为OPENAI_API_KEY
            os.environ["OPENAI_API_KEY"] = args.api_key
    
    # 设置API基地址（如果提供）
    if args.api_base:
        if 'deepseek' in args.api_model:
            os.environ["DEEPSEEK_API_BASE"] = args.api_base
        else:
            os.environ["OPENAI_API_BASE"] = args.api_base

    # 解析API参数
    api_params = {}
    if args.api_params:
        try:
            api_params = json.loads(args.api_params)
        except json.JSONDecodeError:
            print(f"警告: 无法解析API参数: {args.api_params}. 使用空字典。")
    
    # 加载配置和参数
    class_config = config.ClassConfig(
        llm_class=sampler.UnifiedLLM,  # 使用UnifiedLLM以支持多种API
        sandbox_class=evaluator.LocalSandbox
    )
    
    # 创建配置对象
    config_obj = config.Config(
        use_api=args.use_api,
        api_model=args.api_model
    )
    
    # 设置调试模式
    if args.debug:
        import litellm
        litellm.verbose = True
    
    # 设置全局最大样本数
    global_max_sample_num = args.max_samples

    # 加载提示规范
    with open(
        os.path.join(args.spec_path),
        encoding="utf-8",
    ) as f:
        specification = f.read()
    
    # 加载数据集
    problem_name = args.problem_name
    df = pd.read_csv('./data/'+problem_name+'/train.csv')
    data = np.array(df)
    X = data[:, :-1]
    y = data[:, -1].reshape(-1)
    if 'torch' in args.spec_path:
        X = torch.Tensor(X)
        y = torch.Tensor(y)
    data_dict = {'inputs': X, 'outputs': y}
    dataset = {'data': data_dict} 
    
    # 运行主流程
    pipeline.main(
        specification=specification,
        inputs=dataset,
        config=config_obj,
        max_sample_nums=global_max_sample_num,
        class_config=class_config,
        log_dir=args.log_path,
        samples_per_prompt=args.samples_per_prompt,
    )
##