"""Use an LLM to extract structured clinical abnormalities from reports."""

import os
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# API credentials must be supplied through the environment.
API_KEY = os.environ.get('OPENAI_API_KEY')
BASE_URL = os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1')
EVAL_MODEL = os.environ.get('EVAL_MODEL', 'gpt-5.2')

# Tune concurrency to the API provider's rate limit.
MAX_WORKERS = int(os.environ.get('EVAL_MAX_WORKERS', '6'))

# Configure portable input and output roots through environment variables.
input_root_path = os.environ.get('CLEANED_ROOT', './test_results_diagnosis')
output_root_path = os.environ.get('LLM_ROOT', './test_results_LLM')

# Model result directories included in the paper comparison.
file_path_list = [
    'finetuning/deepseek-r1-qwen3-8b',
    'finetuning_sr/deepseek-r1-qwen3-8b',
    # 'finetuning_sr_ICL/deepseek-r1-qwen3-8b',
    # 'finetuning_kg/deepseek-r1-qwen3-8b',
    # 'finetuning_kg_v2/deepseek-r1-qwen3-8b',
    'finetuning_kg_v3/deepseek-r1-qwen3-8b',
    # 'finetuning_kg_v3_v2/deepseek-r1-qwen3-8b',
    # 'finetuning_kg_v3_ICL/deepseek-r1-qwen3-8b',
    'finetuning_DAPT/deepseek-r1-qwen3-8b',
    'finetuning_ICL/deepseek-r1-qwen3-8b',
    # 'finetuning_combined_ICL_woICL/deepseek-r1-qwen3-8b',
    'finetuning_combined_ICL_ICL/deepseek-r1-qwen3-8b',
    'FetalScribe/deepseek-r1-qwen3-8b',
    
    'with_ICL/closed_source/gpt-5.1',
    'with_ICL/closed_source/gpt-5.2',
    'with_ICL/closed_source/gpt-5.4',
    'with_ICL/closed_source/gemini-3.1-flash-lite-preview',
    'with_ICL/closed_source/gemini-3.1-pro-preview',
    'with_ICL/closed_source/claude-sonnet-4-6',
    'with_ICL/closed_source/claude-opus-4-6',
    'with_ICL/closed_source/claude-opus-4-7',
    'with_ICL/closed_source/grok-4.1',
    'with_ICL/closed_source/grok-4.2-nothinking',
    'with_ICL/closed_source/doubao-seed-2-0-lite-260215',
    'with_ICL/closed_source/qwen3-max',
    'with_ICL/open_source/qwen3-235b-a22b',
    'with_ICL/open_source/DeepSeek-V3.2',
    # 'with_ICL/open_source/MiMo-V2-Flash',
    'with_ICL/open_source/kimi-k2.5',
    'with_ICL/open_source/glm-5',
    'with_ICL/AntAngelMed', 
    'with_ICL/HuatuoGPT-Vision-34B', 
    'with_ICL/OpenBioLLM-70B-4bit',

    'without_ICL/closed_source/gpt-5.1',
    'without_ICL/closed_source/gpt-5.2',
    'without_ICL/closed_source/gpt-5.4',
    'without_ICL/closed_source/gemini-3.1-flash-lite-preview',
    'without_ICL/closed_source/gemini-3.1-pro-preview',
    'without_ICL/closed_source/claude-sonnet-4-6',
    'without_ICL/closed_source/claude-opus-4-6',
    'without_ICL/closed_source/claude-opus-4-7',
    'without_ICL/closed_source/grok-4.1',
    'without_ICL/closed_source/grok-4.2-nothinking',
    'without_ICL/closed_source/doubao-seed-2-0-lite-260215',
    'without_ICL/closed_source/qwen3-max',
    'without_ICL/open_source/qwen3-235b-a22b',
    'without_ICL/open_source/DeepSeek-V3.2',
    # 'without_ICL/open_source/MiMo-V2-Flash',
    'without_ICL/open_source/kimi-k2.5',
    'without_ICL/open_source/glm-5',
    'without_ICL/AntAngelMed', 
    'without_ICL/HuatuoGPT-Vision-34B',
    'without_ICL/OpenBioLLM-70B-4bit'
]

# These directories require model-specific labels because sample counts differ.
folder_specific_label_list = [
    'finetuning_sr/deepseek-r1-qwen3-8b',
    'finetuning_sr_ICL/deepseek-r1-qwen3-8b'
]

# Evaluation datasets.
dataset_list = [
    'InDistribution_test.json',
    'OutofDistribution_center1_normal.json', 
    'OutofDistribution_center1_patients.json',
    'OutofDistribution_center2_normal.json', 
    'OutofDistribution_center2_patients.json'
]


class FetalMRIEvaluationbyModel():

    def __init__(self, api_key, base_url):

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

        self.system_prompt = '你是一名对胎儿影像报告进行"关键词提取"的人工智能助手。你将会收到一段文本，你的任务是分析这段文本中提到了哪些异常或疾病，并且进行格式化的整理。你需要遵循以下要求：\n\
            (1) 只整理异常生理现象和疾病，忽视文本中提及的正常生理现象（例如"胼胝体形态正常""胎盘位于后壁/前壁""脑沟回清晰"）；一些较轻微的改变（例如"透明隔间腔略偏窄""宫颈管稍缩短""胎盘组织较厚/稍薄"这类使用了"稍/略"等削弱语气的文本）也需要被整理，但要与严重异常有所区分；母体的改变（例如"母体肾盂积水"）也需要整理。具体的格式要求，请参考第3条。\n\
            (2) 一项异常指的是"1个部位/器官的1种疾病"，不可随意拆分与合并。例如，"颅后窝池明显增宽，双侧小脑半球体积明显小、小脑下蚓部体积小，考虑小脑发育不良可能性大"都是描述"小脑发育不良"相关的异常，算作1项内容。又如，"幕上侧脑室及第三脑室重度扩张，右侧脑室旁及双侧脑室内多发出血灶，并右侧脑室旁软化灶"，由于扩张、出血、软化是不同的疾病，则需要拆分为3项："幕上侧脑室及第三脑室重度扩张""右侧脑室旁及双侧脑室内多发出血灶""右侧脑室旁软化灶"。\n\
            (3) 你的输出中应保留分析过程，结尾必须严格按照以下格式："综上所述：共发现胎儿严重异常N项：1. XXX，2. XXX，...，N. XXX；发现胎儿轻微异常及母体相关异常M项：1. YYY，2. YYY，...，M. YYY。"也就是说，把胎儿严重异常归为一类，胎儿轻微异常和母体异常归为另一类。\n\
            \n\
            ***下面提供5个示例，你的分析与输出需要参考它们：***\n\
            [示例1]\n\
                用户输入：胎儿脑部MR结构未见异常；宫内单胎，臀位，胎盘组织大部分位于子宫前壁；上述改变，建议结合产前诊断。\n\
                期望的输出：报告中提到的"宫内单胎，臀位，胎盘组织大部分位于子宫前壁"等内容都是正常生理现象，没有提到任何疾病与异常。综上所述：共发现胎儿严重异常0项；发现胎儿轻微异常及母体相关异常0项。\n\
            \n\
            [示例2]\n\
                用户输入：胎儿双侧透明隔显示完整，胼胝体形态可，视交叉、垂体柄可见；部分型前置胎盘，胎盘组织大部分位于子宫后壁，胎盘后壁下段局部较厚，胎盘内信号欠均匀；扫及母体左肾盂、肾盏及输尿管腹段轻度扩张、积液；母体腰骶部皮下见少许絮状T2WI稍高信号影。上述改变，建议结合产前诊断。\n\
                期望的输出：报告中提到"胎盘后壁下段局部较厚，胎盘内信号欠均匀"算1项严重异常；此外，"母体左肾盂、肾盏及输尿管腹段轻度扩张、积液""母体腰骶部皮下见少许絮状T2WI稍高信号影"分别是母体的2项异常。综上所述：共发现胎儿严重异常1项：1. 胎盘后壁下段局部较厚，信号欠均匀；发现胎儿轻微异常及母体相关异常2项：1. 母体左肾盂、肾盏及输尿管腹段轻度扩张、积液，2. 母体腰骶部皮下絮状T2WI稍高信号影。\n\
            \n\
            [示例3]\n\
                用户输入："31+1周孕"胎儿针对性颅脑MRI普通扫描： 幕上侧脑室及第三脑室重度扩张，右侧脑室旁及双侧脑室内多发出血灶，并右侧脑室旁软化灶；提示为胎儿颅内出血Ⅳ级；建议结合产前诊断咨询。\n\
                期望的输出：报告中提到"幕上侧脑室及第三脑室重度扩张，右侧脑室旁及双侧脑室内多发出血灶，并右侧脑室旁软化灶"都是严重异常，并且对应不同的疾病，应当拆分为3项。报告中"提示为胎儿颅内出血Ⅳ级；建议结合产前诊断咨询"是总结和建议，不是异常/疾病的关键词。综上所述：共发现胎儿严重异常3项：1.  幕上侧脑室及第三脑室重度扩张，2. 右侧脑室旁及双侧脑室内多发出血，3. 右侧脑室旁软化；发现胎儿轻微异常及母体相关异常0项。\n\
            \n\
            [示例4]\n\
                用户输入：孕龄约31+6周，胎儿头位。双侧枕、顶叶侧脑室旁脑白质广泛异常信号，T1WI及T2WI呈稍高信号，弥散受限；双侧基底节区及小脑亦见弥散受限高信号；双侧侧脑室扩张，左侧后角最宽约1.57cm，右侧约1.2cm，三角区左侧宽约1.3cm，右侧约1.2cm，伴双侧侧脑室后角室管膜下囊肿（左侧0.9×0.7cm，右侧1.0×0.9cm）；双侧枕颞叶局部脑实质变薄（最薄约0.3～0.5cm）；颅后窝池明显增宽（小脑蚓部后缘与枕骨内侧壁间距约1.4cm），双侧小脑半球及小脑蚓部体积明显偏小（小脑半球最大横径约2.8cm）。建议结合产前诊断，必要时隔期复查。\n\
                期望的输出：报告中提到"双侧枕、顶叶侧脑室旁脑白质广泛异常信号，T1WI及T2WI呈稍高信号，弥散受限"需归纳为1项严重异常，不可拆分；"双侧基底节区及小脑亦见弥散受限高信号"是1项严重异常；"双侧侧脑室扩张 (后续具体数字略)"是严重异常；"双侧侧脑室后角室管膜下囊肿 (具体数字略)"是严重异常；"双侧枕颞叶局部脑实质变薄 (具体数字略)"是严重异常；"颅后窝池明显增宽 (具体数字略)，双侧小脑半球及小脑蚓部体积明显偏小 (具体数字略)"都描述小脑发育不良风险，归纳算作1项严重异常。综上所述：共发现胎儿严重异常6项：1. 双侧枕、顶叶侧脑室旁脑白质异常信号，2. 双侧基底节区及小脑弥散异常信号，3. 双侧侧脑室扩张，4. 双侧侧脑室后角室管膜下囊肿，5. 双侧枕颞叶局部脑实质变薄，6. 颅后窝池明显增宽、双侧小脑半球及小脑蚓部体积明显偏小；发现胎儿轻微异常及母体相关异常0项。\n\
            \n\
            [示例5]\n\
                用户输入：胎儿针对性颅脑MRI普通扫描： 颅后窝池明显增宽，双侧小脑半球体积明显小、小脑下蚓部体积小，考虑小脑发育不良可能性大；双侧侧脑室扩张，左侧中度扩张，右侧轻度扩张。双侧侧脑室后角可见室管膜下囊肿； 双侧基底节区及枕、顶叶侧脑室旁脑白质广泛异常信号，双侧枕颞叶局部脑实质变薄。 综上，性质待定，建议结合产前诊断咨询。\n\
                期望的输出：报告中提到"颅后窝池明显增宽，双侧小脑半球体积明显小、小脑下蚓部体积小，考虑小脑发育不良可能性大"都描述了小脑发育的问题，算作1项严重异常；"双侧侧脑室扩张，左侧中度扩张，右侧轻度扩张"共算作1项异常；"双侧侧脑室后角可见室管膜下囊肿"是1项异常；"双侧基底节区及枕、顶叶侧脑室旁脑白质广泛异常信号"需要按照"双侧基底节区"与"枕、顶叶侧脑室"部位，需记录为2项异常；"双侧枕颞叶局部脑实质变薄"算作1项异常。综上所述：共发现胎儿严重异常6项：1. 颅后窝池明显增宽，双侧小脑半球体积明显小、小脑下蚓部体积小，2. 双侧侧脑室扩张，3. 双侧侧脑室后角可见室管膜下囊肿，4. 双侧基底节区白质异常信号，5. 枕、顶叶白质异常信号，6. 双侧枕颞叶局部脑实质变薄；发现胎儿轻微异常及母体相关异常0项。'

    def call_llm(self, model_name, text_content, max_retries=8):

        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {'role': 'system', 'content': self.system_prompt},
                        {'role': 'user', 'content': text_content},
                    ],
                    temperature=0.1,
                    max_tokens=2048,
                )
                return completion.choices[0].message.content

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 * attempt
                    print(f"Request failed, retrying in {wait_time} seconds... Error: {e}")
                    time.sleep(wait_time)
                else:
                    print(f"Max retries reached. Skipping. Error: {e}")
                    return None

    def _call_llm_with_null_retry(self, model_name, text_content, null_retries=10):
        """Retry API responses whose message content is null or empty."""
        for retry in range(null_retries):
            result = self.call_llm(model_name, text_content)
            if result is not None and result.strip() != '':
                return result.strip()
            else:
                print(f'    Empty response on attempt {retry + 1}; retrying in 0.5 s...')
                time.sleep(0.5)
        return None

    def process_single_file_v0(self, input_filename, output_filename, model_name):
        try:
            with open(input_filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("输入JSON必须是列表格式")
            print(f"成功加载输入数据，共 {len(data)} 条记录")
        except Exception as e:
            print(f"读取输入文件失败：{str(e)}")
            raise

        output_list = []
        success_count = 0
        fail_count = 0

        for i, item in enumerate(data):
            # Read the normalized prediction produced by stage 1.
            cleaned_output = item['cleaned_output']
            text_content = '请整理以下case：\n"报告文本"：' + cleaned_output
            model_evaluated_result = self.call_llm(model_name, text_content)
            model_evaluated_result = model_evaluated_result.strip()

            if model_evaluated_result not in [None, '']:
                output_list.append({
                    "output": cleaned_output,
                    "evaluated_result": model_evaluated_result
                })
                success_count += 1
            else:
                print('模型调用出现bug，返回None或者空字符串')
                fail_count += 1

            if (i + 1) % 10 == 0:
                print(f'  已处理 {i + 1}/{len(data)} 条...')

        print(f"处理完成：成功 {success_count} 条，失败 {fail_count} 条")
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(output_list, f, ensure_ascii=False, indent=2)
        print(f"输出文件已保存至：{output_filename}")

    def process_single_file(self, input_filename, output_filename, model_name):
        try:
            with open(input_filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("输入JSON必须是列表格式")
            print(f"成功加载输入数据，共 {len(data)} 条记录")
        except Exception as e:
            print(f"读取输入文件失败：{str(e)}")
            return

        # Process API calls concurrently while preserving record order.
        results_dict = {}
        lock = threading.Lock()
        success_count = 0
        fail_count = 0
        processed_count = 0

        def _process_one(i, item):
            nonlocal success_count, fail_count, processed_count
            cleaned_output = item['cleaned_output']
            text_content = '请整理以下case：\n"报告文本"：' + cleaned_output

            model_evaluated_result = self._call_llm_with_null_retry(model_name, text_content)

            with lock:
                if model_evaluated_result:
                    results_dict[i] = {
                        "output": cleaned_output,
                        "evaluated_result": model_evaluated_result
                    }
                    success_count += 1
                else:
                    print(f'  ✗ 第{i}条：重试仍失败，跳过')
                    fail_count += 1
                processed_count += 1
                if processed_count % 10 == 0:
                    print(f'  已处理 {processed_count}/{len(data)} 条...')

        print(f"  🚀 启动 {MAX_WORKERS} 线程并行处理...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(_process_one, i, item): i for i, item in enumerate(data)}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    idx = futures[future]
                    with lock:
                        fail_count += 1
                        processed_count += 1
                    print(f'  ✗ 第{idx}条异常: {e}')

        # Restore the original record order.
        output_list = [results_dict[i] for i in sorted(results_dict.keys())]

        print(f"处理完成：成功 {success_count} 条，失败 {fail_count} 条")
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(output_list, f, ensure_ascii=False, indent=2)
        print(f"输出文件已保存至：{output_filename}")

    def process_label_v0(self, input_filename, output_filename, model_name):
        try:
            with open(input_filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("输入JSON必须是列表格式")
            print(f"成功加载输入数据，共 {len(data)} 条记录")
        except Exception as e:
            print(f"读取输入文件失败：{str(e)}")
            raise

        output_list = []
        success_count = 0
        fail_count = 0

        for i, item in enumerate(data):
            # Read the normalized reference produced by stage 1.
            cleaned_label = item['cleaned_label']
            text_content = '请整理以下case：\n"报告文本"：' + cleaned_label
            model_evaluated_result = self.call_llm(model_name, text_content)
            model_evaluated_result = model_evaluated_result.strip()

            if model_evaluated_result not in [None, '']:
                output_list.append({
                    "label": cleaned_label,
                    "evaluated_result": model_evaluated_result
                })
                success_count += 1
            else:
                print('模型调用出现bug，返回None或者空字符串')
                fail_count += 1

            if (i + 1) % 10 == 0:
                print(f'  已处理 {i + 1}/{len(data)} 条...')

        print(f"处理完成：成功 {success_count} 条，失败 {fail_count} 条")
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(output_list, f, ensure_ascii=False, indent=2)
        print(f"输出文件已保存至：{output_filename}")

    def process_label(self, input_filename, output_filename, model_name):
        try:
            with open(input_filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("输入JSON必须是列表格式")
            print(f"成功加载输入数据，共 {len(data)} 条记录")
        except Exception as e:
            print(f"读取输入文件失败：{str(e)}")
            return

        # Process API calls concurrently.
        results_dict = {}
        lock = threading.Lock()
        success_count = 0
        fail_count = 0
        processed_count = 0

        def _process_one(i, item):
            nonlocal success_count, fail_count, processed_count
            cleaned_label = item['cleaned_label']
            text_content = '请整理以下case：\n"报告文本"：' + cleaned_label

            model_evaluated_result = self._call_llm_with_null_retry(model_name, text_content)

            with lock:
                if model_evaluated_result:
                    results_dict[i] = {
                        "label": cleaned_label,
                        "evaluated_result": model_evaluated_result
                    }
                    success_count += 1
                else:
                    print(f'  ✗ 第{i}条：重试仍失败，跳过')
                    fail_count += 1
                processed_count += 1
                if processed_count % 10 == 0:
                    print(f'  已处理 {processed_count}/{len(data)} 条...')

        print(f"  🚀 启动 {MAX_WORKERS} 线程并行处理...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(_process_one, i, item): i for i, item in enumerate(data)}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    idx = futures[future]
                    with lock:
                        fail_count += 1
                        processed_count += 1
                    print(f'  ✗ 第{idx}条异常: {e}')

        output_list = [results_dict[i] for i in sorted(results_dict.keys())]

        print(f"处理完成：成功 {success_count} 条，失败 {fail_count} 条")
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(output_list, f, ensure_ascii=False, indent=2)
        print(f"输出文件已保存至：{output_filename}")


if __name__ == '__main__':
    if not API_KEY:
        raise RuntimeError('OPENAI_API_KEY is required')
    evaluator = FetalMRIEvaluationbyModel(API_KEY, BASE_URL)

    # Extract clinical keywords from model predictions.
    for model_name in file_path_list:
        output_dir = os.path.join(output_root_path, model_name)
        os.makedirs(output_dir, exist_ok=True)

        for dataset_name in dataset_list:
            # Stage 1 prefixes normalized files with [Cleaned]_.
            cleaned_dataset_name = '[Cleaned]_' + dataset_name
            file_full_name = os.path.join(input_root_path, model_name, cleaned_dataset_name)
            evaluated_full_name = os.path.join(output_dir, dataset_name)

            # Skip completed files to avoid duplicate API charges.
            if os.path.exists(evaluated_full_name): 
                print(f'  已存在，跳过: {evaluated_full_name}')
                continue
            
            # Missing model outputs are allowed in a multi-model comparison.
            if not os.path.exists(file_full_name):
                print(f'  缺失输入文件，跳过: {file_full_name}')
                continue

            print(f'\n处理: {model_name} | {cleaned_dataset_name}')
            evaluator.process_single_file(file_full_name, evaluated_full_name, EVAL_MODEL)

    # Extract reference keywords once per dataset.
    print('\n处理 label...')
    label_output_dir = os.path.join(output_root_path, 'label')
    os.makedirs(label_output_dir, exist_ok=True)

    for dataset_name in dataset_list:
        cleaned_dataset_name = '[Cleaned]_' + dataset_name
        # References are model-independent; use the first model's cleaned labels.
        file_full_name = os.path.join(input_root_path, file_path_list[0], cleaned_dataset_name)
        label_full_name = os.path.join(label_output_dir, dataset_name)

        # Each reference file is generated only once.
        if os.path.exists(label_full_name):
            print(f'  label 已存在，跳过: {label_full_name}')
            continue
        
        if not os.path.exists(file_full_name):
            print(f'  label 源文件缺失，跳过: {file_full_name}')
            continue

        evaluator.process_label(file_full_name, label_full_name, EVAL_MODEL)

    # Generate model-specific references when sample counts differ.
    print('\n处理文件夹专属 label...')
    for model_name in folder_specific_label_list:
        label_output_dir = os.path.join(output_root_path, model_name, 'label')
        os.makedirs(label_output_dir, exist_ok=True)

        for dataset_name in dataset_list:
            cleaned_dataset_name = '[Cleaned]_' + dataset_name
            file_full_name = os.path.join(input_root_path, model_name, cleaned_dataset_name)
            label_full_name = os.path.join(label_output_dir, dataset_name)

            if os.path.exists(label_full_name):
                print(f'  label 已存在，跳过: {label_full_name}')
                continue

            if not os.path.exists(file_full_name):
                print(f'  label 源文件缺失，跳过: {file_full_name}')
                continue

            print(f'\n处理专属 label: {model_name} | {dataset_name}')
            evaluator.process_label(file_full_name, label_full_name, EVAL_MODEL)

    print('\n所有 LLM 评估 (Part 1) 已完成!')
