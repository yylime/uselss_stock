from collections import Counter

if __name__ == "__main__":
    with open(r"config/all_samples_features.txt", 'r') as f:
        lines = f.readlines()
    counter = Counter(lines)
    res = []
    for f, cnt in counter.items():
        if cnt >= 8:
            res.append(f)
    # with open("selected_features.txt", 'w') as f:
    #     f.writelines(res)
    print(len(res))
    res = []
    for f, cnt in counter.items():
        if cnt >= 10:
            res.append(f)
    with open("config\selected_features.txt", 'w') as f:
        f.writelines(res)

    print(len(res))