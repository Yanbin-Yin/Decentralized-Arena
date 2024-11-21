import os
import json

# 文件夹路径
input_folder = "/data/shared/decentralized_arena/math_geometry_v1_selected_responses"
output_folder = "/data/shared/decentralized_arena/math_geometry_final_selected_responses"

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 指定的 question_id 列表，按升序排序
selected_question_ids = sorted([172, 140, 210, 69, 222, 136, 79, 151, 26, 373, 208, 96, 258, 29, 144, 324, 66, 244, 128, 237, 239, 360,
               358, 88, 241, 104, 168, 235, 94, 183, 70, 300, 156, 76, 125, 127, 261, 243, 152, 307, 147, 47, 54, 110,
               339, 80, 181, 395, 196, 95, 179, 116, 264, 12, 178, 223, 218, 377, 41, 101, 114, 219, 174, 327, 30, 213,
               175, 375, 63, 98, 267, 357, 7, 124, 231, 379, 126, 272, 315, 370, 323, 296, 118, 68, 180, 135, 10, 304,
               157, 121, 249, 160, 28, 229, 346, 215, 4, 238, 159, 103])

def process_file(file_name):
    """读取每个 JSONL 文件，筛选出指定的 question_id，并保持原始多行格式，按指定顺序输出"""
    input_file_path = os.path.join(input_folder, file_name)
    output_file_path = os.path.join(output_folder, file_name)

    # 用于存储符合条件的 question_id 数据
    records_by_question_id = {}

    with open(input_file_path, 'r') as input_file:
        buffer = []  # 用于存储一条完整的 JSON 数据
        for line_number, line in enumerate(input_file, start=1):
            stripped_line = line.strip()
            if stripped_line.startswith("{") and buffer:  # 遇到新的记录，处理缓冲区中的完整记录
                try:
                    # 将缓冲区的多行拼接成一个完整的 JSON
                    record = json.loads("".join(buffer))
                    question_id = record.get('question_id')

                    # 筛选出指定的 question_id
                    if question_id in selected_question_ids:
                        records_by_question_id[question_id] = "\n".join(buffer) + '\n'
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file {file_name}, line {line_number}: {e}")
                    print(f"Buffer content: {buffer}")
                buffer = [line]  # 开始新的 JSON 记录
            else:
                buffer.append(line)  # 将当前行添加到缓冲区

        # 处理最后一条记录
        if buffer:
            try:
                record = json.loads("".join(buffer))  # 处理最后的缓冲区数据
                question_id = record.get('question_id')
                if question_id in selected_question_ids:
                    records_by_question_id[question_id] = "\n".join(buffer) + '\n'
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON at end of file {file_name}: {e}")
                print(f"Buffer content: {buffer}")

    # 按指定顺序输出符合条件的记录
    with open(output_file_path, 'w') as output_file:
        for question_id in selected_question_ids:
            if question_id in records_by_question_id:
                output_file.write(records_by_question_id[question_id])

# 遍历输入文件夹中的所有文件
for file_name in os.listdir(input_folder):
    if file_name.endswith(".jsonl"):
        process_file(file_name)

# 第二步：移除生成文件中的空行
def remove_blank_lines(file_path):
    """移除 JSONL 文件中的空行"""
    with open(file_path, 'r') as input_file:
        lines = input_file.readlines()
    with open(file_path, 'w') as output_file:
        for line in lines:
            if line.strip():  # 只写入非空行
                output_file.write(line)

# 遍历输出文件夹并移除空行
for file_name in os.listdir(output_folder):
    if file_name.endswith(".jsonl"):
        remove_blank_lines(os.path.join(output_folder, file_name))

print("处理完成！")
