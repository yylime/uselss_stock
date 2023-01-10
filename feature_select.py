from collections import Counter

if __name__ == "__main__":
    with open("f.txt", 'r') as f:
        lines = f.readlines()
    counter = Counter(lines)
    res = []
    for f, cnt in counter.items():
        if cnt >= 7:
            res.append(f)
    with open("selected_features.txt", 'w') as f:
        f.writelines(res)
    
    res = []
    for f, cnt in counter.items():
        if cnt >= 9:
            res.append(f)
    with open("selected_features_train.txt", 'w') as f:
        f.writelines(res)
    
    print(len(res))