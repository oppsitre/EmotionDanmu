# Test the danmu when concatenating into ones
from sklearn.ensemble import RandomForestClassifier

videos = []
videos_origin = []
with open('../dataset/texts/video_class_1.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip().split('\t')
        videos.append([int(line[0]), int(line[-1])])
        videos_origin.append([int(line[0]), int(line[-1])])

random.shuffle(videos)
n_data = len(videos)
