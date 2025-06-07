from collections import defaultdict


def find_isolated_classes(file_path):
    # 存储每个类别对应的行号
    class_lines = defaultdict(list)

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                # 分割标签和文本
                label, text = line.split('####', 1)
                label = label.strip()
                class_lines[label].append(line_num)
            except ValueError:
                print(f"警告：第 {line_num} 行格式不正确: {line}")
                continue

    # 找出孤立类别（只有1个样本的类别）
    isolated_classes = {label: lines for label, lines in class_lines.items() if len(lines) == 1}

    return isolated_classes, class_lines


# 使用示例
file_path = 'finaldata.txt'
isolated_classes, all_classes = find_isolated_classes(file_path)

print("\n=== 所有类别分布 ===")
for label, lines in all_classes.items():
    print(f"类别 {label}: {len(lines)} 个样本")

print("\n=== 孤立类别（只有1个样本的类别）===")
if isolated_classes:
    for label, lines in isolated_classes.items():
        print(f"类别 {label} 在第 {lines[0]} 行是孤立的")
else:
    print("没有孤立类别 - 所有类别都有至少2个样本")

# 输出统计信息
print(f"\n总类别数: {len(all_classes)}")
print(f"孤立类别数: {len(isolated_classes)}")