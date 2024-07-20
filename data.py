import os
import random
import shutil


def generate_knowledge_graph(num):
    num_timestamps = 200
    num_compute_nodes = num
    num_algorithm_nodes = num
    num_data_nodes = num

    entities = {
        "compute": list(range(num_compute_nodes)),
        "algorithm": list(
            range(num_compute_nodes, num_compute_nodes + num_algorithm_nodes)
        ),
        "data": list(
            range(
                num_compute_nodes + num_algorithm_nodes,
                num_compute_nodes + num_algorithm_nodes + num_data_nodes,
            )
        ),
    }

    print("entities: ", len(entities["algorithm"] + entities["compute"] + entities["data"]))

    dataset = []

    # 1.如果 算力A 在t时刻 -> 数据B，那么必须有一个 算力A 在t时刻 -> 算法C，算法C 在t时刻 -> 数据B，并且这三者的时间戳相同，且随机生成一个[20,80]的时间戳数ct，在后面连续的ct个时间戳内，这三者的关系都是存在的
    # 2.每1个算力指向1个固定的算法，每个算法指向1个固定的数据，所以每个算力指向1个固定的数据

    # 遍历compute_entities，每个compute_entities的key是算力ID
    m = {}
    for compute_id in entities["compute"]:
        # 随机拿出1个算法ID，拿出后要删除
        algorithm_id = random.sample(entities["algorithm"], 1)
        entities["algorithm"].remove(algorithm_id[0])

        # 随机拿出1个数据ID，拿出后要删除
        data_id = random.sample(entities["data"], 1)
        entities["data"].remove(data_id[0])

        # 组成新集合【算力ID，算法ID，数据ID】
        m[compute_id] = [compute_id, algorithm_id[0], data_id[0]]

    # 遍历时间数
    # 确保dataset中每一秒都有333组数据
    for timestamp in range(num_timestamps):
        # 计算dataset中,time=timestamp的数据量
        count = 0
        for i in range(len(dataset)):
            if dataset[i][3] == timestamp:
                count += 1
        need = 333 - count / 3

        for i in range(int(need)):
            if len(m) == 0:
                break

            # 从m中随机取出一个算力ID，选取后要删除
            values_list = list(m.values())
            data = random.sample(values_list, 1)[0]
            m.pop(data[0])
            compute_id = data[0]
            algorithm_id = data[1]
            data_id = data[2]

            # 随机生成一个持续次数
            ct = random.randint(1, 50)

            # 构建数据算力指向算法的关系
            for t in range(ct):
                dataset.append(
                    [compute_id, 1, algorithm_id, timestamp + t]
                )  # 算力 -> 算法
                dataset.append(
                    [algorithm_id, 0, data_id, timestamp + t]
                )  # 算法 -> 数据
                dataset.append([compute_id, 2, data_id, timestamp + t])  # 算力 -> 数据
    # 按照时间戳排序
    dataset = sorted(dataset, key=lambda x: x[3])

    return dataset


def split_dataset(dataset, train_ratio=0.6, test_ratio=0.2):
    total_samples = len(dataset)
    train_size = int(total_samples * train_ratio)
    test_size = int(total_samples * test_ratio)
    val_size = total_samples - train_size - test_size

    train_set = dataset[:train_size]
    val_set = dataset[train_size : train_size + val_size]
    test_set = dataset[train_size + val_size :]

    return train_set, val_set, test_set


def write_to_file(dataset, filename):
    with open(filename, "w") as f:
        for entry in dataset:
            f.write(f"{entry[0]}\t{entry[1]}\t{entry[2]}\t{entry[3]}\t{0}\n")


def main():
    # 算力节点数
    num = 3000

    print("生成知识图谱数据集...")
    kg_dataset = generate_knowledge_graph(num)
    train_set, val_set, test_set = split_dataset(
        kg_dataset, train_ratio=0.8, test_ratio=0.1
    )

    print("\n数据集划分完成，详细信息如下：")
    print(f"训练集大小: {len(train_set)}")
    print(f"验证集大小: {len(val_set)}")
    print(f"测试集大小: {len(test_set)}")

    # 删除现有的data目录（如果存在）
    if os.path.exists("data"):
        shutil.rmtree("data")

    # 创建data目录
    os.makedirs("data")

    # 将数据集写入文件
    write_to_file(train_set, "data/train.txt")
    write_to_file(val_set, "data/valid.txt")
    write_to_file(test_set, "data/test.txt")

    print("数据集写入完成。")


if __name__ == "__main__":
    main()
