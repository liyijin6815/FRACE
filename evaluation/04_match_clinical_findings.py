"""Use an LLM to semantically match extracted findings and count TP/FP/FN."""

import os
import json
import time
import requests
import re


# API credentials must be supplied through the environment.
API_KEY = os.environ.get('OPENAI_API_KEY')
BASE_URL = os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1')
EVAL_MODEL = os.environ.get('EVAL_MODEL', 'gpt-5.2')

# Stage 4 reads and writes within the LLM-evaluation tree.
input_root_path = os.environ.get('LLM_ROOT', './test_results_LLM')
output_root_path = os.environ.get('LLM_ROOT', './test_results_LLM')

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

dataset_list = [
    'InDistribution_test.json', 
    'OutofDistribution_center1_normal.json',
    'OutofDistribution_center1_patients.json',
    'OutofDistribution_center2_normal.json',
    'OutofDistribution_center2_patients.json'
]


def preprocess(text):
    processed_text = text.replace('\n', '').replace('\r', '').replace(' ', '').replace('　', '')
    keyword = '综上所述：'
    keyword_pos = processed_text.rfind(keyword)

    if keyword_pos != -1:
        return processed_text[keyword_pos + len(keyword):]

    else:
        colon_pos = processed_text.rfind('：')
        if colon_pos != -1:
            return processed_text[colon_pos + 1:]

    print('预处理时发生意外，未能找到关键词 "综上所述"')
    return processed_text


class FetalMRIEvaluationbyModel():

    def __init__(self, api_key, base_url):

        self.api_key = api_key
        self.base_url = base_url

        self.system_prompt = '你是一名对胎儿影像报告进行"关键词比对"的人工智能助手。你将会收到一组成对的输入，其中包括Output和Label。你的任务是分析、比对Output和Label中的各项关键词，从而生成True Positive (TP)、False Positive (FP)、False Negative (FN) 的混淆矩阵。以下为本任务的详细说明，你需要遵循其中的要求：\n\
            (1) Output和Label包含"胎儿严重异常"和"胎儿轻微异常及母体异常"两个部分。在评估之前，需要把二者合并到一起评估。\n\
            (2) Output还是Label已经对关键词使用逗号或者分号进行了划分，请以此为标准，不要拆分/合并各项关键词。\n\
            (3) 关键词的含义相近即可，无需追求文本表述的一致性。例如："透明隔腔增宽"和"透明隔腔宽约1.6cm"两者都描述了透明隔腔增宽（正常范围在1.0cm以内），算作比对成功；"双侧基底节区及小脑弥散异常信号"与"双侧基底节区白质异常信号"都提到了双侧基底节区，虽然前者提到了小脑而后者没有，但也算比对成功；"双侧脑室内见异常信号影，以左侧为著，T1WI呈高信号"与"双侧脑室内多发出血灶"，前者提到信号异常，但没有指出出血，算比对失败。\n\
            (4) 同时在Output和Label中出现的关键词是TP；只在Output中出现的关键词属于FP；只在Label中出现的关键词属于FN。\n\
            (5) 你的输出必须按照以下格式："XXX""XXX"为TP，"XXX""XX"为FP，"XXX""XXX"为FN。共发现TP=L项，FP=M项，FN=N项。\n\
            \n\
            ***下面提供4个示例，供你参考***\n\
            [示例1]\n\
                Output："共发现胎儿严重异常4项：1. 胎儿双侧侧脑室轻度增宽，2. 右侧侧脑室体部及后角室管膜下囊肿，3. 左侧侧脑室后角小出血灶，4. 子宫前壁下段偏左胎盘粘连可能性大；发现胎儿轻微异常及母体相关异常1项：1. 母体骶管囊肿。"\n\
                Label："共发现胎儿严重异常4项：1. 胎儿双侧脑室轻度扩张，2. 双侧脑室出血，3. 后颅窝池增宽、考虑蛛网膜囊肿待排，4. 子宫前壁下段偏左胎盘黏连伴植入可能、胎盘厚薄不均；发现胎儿轻微异常及母体相关异常1项：1. 母体骶管囊肿。"\n\
                期望的输出："胎儿双侧脑室轻度扩张""双侧脑室出血""疑似蛛网膜囊肿""子宫前壁下段偏左胎盘黏连可能"母体骶管囊肿"为TP。共发现TP=5项，FP=0项，FN=0项。\n\
            \n\
            [示例2]\n\
                Output："共发现胎儿严重异常5项：1. 胎儿右侧大脑半球严重损毁、脑室内出血、脑实质受压菲薄，2. 左侧脑室重度扩张，3. 左侧额叶及基底节区结构紊乱，4. 左侧外侧裂浅平、额叶岛盖未出现，5. 鼻骨扁平且短小；发现胎儿轻微异常及母体相关异常1项：1. 小脑偏小。"\n\
                Label："共发现胎儿严重异常6项：1. 胎儿右侧大脑半球严重损毁伴片状出血灶，2. 左侧脑室重度扩张，3. 左侧额叶及基底节区结构紊乱，4. 左侧外侧裂变浅、不除外皮层发育畸形，5. 小脑半球偏小、考虑发育不良可能，6. 鼻骨发育异常；发现胎儿轻微异常及母体相关异常0项。"\n\
                期望的输出："胎儿右侧大脑半球严重损毁伴随出血""左侧脑室重度扩张""左侧额叶及基底节区结构紊乱""左侧外侧裂变浅""鼻骨发育异常""小脑半球偏小"为TP。共发现TP=6项，FP=1项，FN=0项。\n\
            \n\
            [示例3]\n\
                Output："共发现胎儿严重异常3项：1. 双侧侧脑室前角及体部前份不规则囊性影，2. 双侧脑室三角区室管膜欠规整伴散在T2WI低信号小结节影，3. 左侧鼻泪管扩张；发现胎儿轻微异常及母体相关异常0项。"\n\
                Label："共发现胎儿严重异常3项：1. 胎儿双侧脑室室管膜下囊肿，2. 室管膜欠规整伴散在低信号小结节，3. 胎儿左侧鼻泪管扩张；发现胎儿轻微异常及母体相关异常1项：1. 胎儿双侧脑室轻度增宽。"\n\
                期望的输出："双侧脑室室管膜下囊性影""室管膜欠规整伴散在低信号小结节影""左侧鼻泪管扩张"为TP，"胎儿双侧脑室轻度增宽"为FN。共发现TP=3项，FP=0项，FN=1项。\n\
            \n\
            [示例4]\n\
                Output："共发现胎儿严重异常4项：1. 双侧侧脑室生发基质区异常信号，提示生发基质出血，2. 冠状位双侧丘脑后方小囊状高信号影，考虑脉络丛囊肿可能，3. 环池、四叠体池、小脑上池及大脑大静脉池略显宽，4. 双侧侧脑室前后角圆钝；发现胎儿轻微异常及母体相关异常2项：1. 胎盘局部增厚，2. 透明隔间腔稍宽。"\n\
                Label："共发现胎儿严重异常1项：1. 胎儿双侧侧脑室生发基质出血样表现；发现胎儿轻微异常及母体相关异常1项：1. 双侧脑室前后角圆钝。"\n\
                期望的输出："胎儿双侧侧脑室生发基质出血样表现""双侧脑室前后角圆钝"为TP，"冠状位双侧丘脑后方小囊状高信号影，考虑脉络丛囊肿可能""环池、四叠体池、小脑上池及大脑大静脉池略显宽""胎盘局部增厚""透明隔间腔稍宽"为FP。共发现TP=2项，FP=4项，FN=0项。'

    def call_llm(self, model_name, text_content, max_retries=8):

        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    f"{self.base_url.rstrip('/')}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": text_content},
                        ],
                        "temperature": 0.1,
                        "max_tokens": 2048
                    },
                    timeout=120
                )

                # Include response context in API error diagnostics.
                if resp.status_code != 200:
                    print(f"  [DEBUG] HTTP {resp.status_code}: {resp.text[:500]}")

                resp.raise_for_status()
                data = resp.json()

                # Print a limited number of raw response diagnostics.
                if attempt == 0:
                    raw_content = data.get("choices", [{}])[0].get("message", {}).get("content", None)
                    if raw_content is None:
                        print(f"  [DEBUG] 返回结构异常: {json.dumps(data, ensure_ascii=False)[:500]}")

                content = data["choices"][0]["message"]["content"]

                if content:
                    content = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL).strip()

                return content

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 * (attempt + 1)
                    print(f"  [RETRY {attempt+1}/{max_retries}] {type(e).__name__}: {e}")
                    time.sleep(wait_time)
                else:
                    print(f"  [FAILED] 重试耗尽: {type(e).__name__}: {e}")
                    return None

    def process_single_file(self, output_data, label_data, output_filename, model_name):

        assert len(output_data) == len(label_data), \
            f'样本数量不匹配: output={len(output_data)}, label={len(label_data)}'

        output_list = []
        success_count = 0
        fail_count = 0

        for idx in range(len(output_data)):
            output_text = output_data[idx]['evaluated_result']
            label_text = label_data[idx]['evaluated_result']

            output_text_preprocessed = preprocess(output_text)
            label_text_preprocessed = preprocess(label_text)

            text_content = '请比对以下case：\n' + 'Output: "' + output_text_preprocessed + '"\n' \
                + 'Label: "' + label_text_preprocessed + '"'

            model_result = self.call_llm(model_name, text_content)

            if model_result is None or model_result.strip() == '':
                print(f'  第 {idx+1} 条：模型返回 None 或空字符串，跳过')
                fail_count += 1
                continue

            model_result = model_result.strip()

            output_list.append({
                "model_result": model_result
            })
            success_count += 1

            if (idx + 1) % 10 == 0:
                print(f'  已处理 {idx + 1}/{len(output_data)} 条...')

        print(f"处理完成：成功 {success_count} 条，失败 {fail_count} 条")
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(output_list, f, ensure_ascii=False, indent=2)
        print(f"输出文件已保存至：{output_filename}")


if __name__ == '__main__':
    if not API_KEY:
        raise RuntimeError('OPENAI_API_KEY is required')
    evaluator = FetalMRIEvaluationbyModel(API_KEY, BASE_URL)

    for model_name in file_path_list:
        output_dir = os.path.join(output_root_path, model_name)
        os.makedirs(output_dir, exist_ok=True)

        for dataset_name in dataset_list:
            output_filename = os.path.join(output_dir, f"[LLM_Eva]_{dataset_name}")

            # Skip completed files to avoid duplicate API charges.
            if os.path.exists(output_filename):
                print(f'  已存在，跳过: {output_filename}')
                continue

            input_filename = os.path.join(input_root_path, model_name, dataset_name)
            # Prefer model-specific references when sample counts differ.
            local_label = os.path.join(input_root_path, model_name, 'label', dataset_name)
            label_filename = local_label if os.path.exists(local_label) else os.path.join(input_root_path, 'label', dataset_name)

            if not os.path.exists(input_filename):
                print(f'  缺失模型输入文件，跳过: {input_filename}')
                continue
            if not os.path.exists(label_filename):
                print(f'  缺失label文件，跳过: {label_filename}')
                continue

            try:
                with open(input_filename, 'r', encoding='utf-8') as f:
                    input_data = json.load(f)
                with open(label_filename, 'r', encoding='utf-8') as f:
                    label_data = json.load(f)
            except Exception as e:
                print(f'  读取输入失败，跳过: {model_name} | {dataset_name} | {e}')
                continue

            print(f'\n处理: {model_name} | {dataset_name}')
            evaluator.process_single_file(input_data, label_data, output_filename, EVAL_MODEL)

    print('\n所有 LLM 评估 (Part 2) 已完成!')
