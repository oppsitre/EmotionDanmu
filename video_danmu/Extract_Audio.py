# coding: utf-8
#######Extract the audio information###################
# ##  Parse the file doc2id.txt, prepare for the later video frames extraction and audio extraction
# Video_ID, Cluster_1_Time(Begin, End, Center),..., Cluster_9_Time(Begin, End, Center)
# #### NOTE: Turn all the time into integer, cause the ffmpeg can only recognize the integer
# ## Read the file doc2id.txt

import numpy as np
import os
import tqdm
import subprocess
import librosa
import logging
import logging.handlers

def read_file(filename):
    f = open(filename)
    lines = f.readlines()
    f.close()
    extract_time = {}

    for line in lines:
        line = line.split()
        if len(line) == 6:
            # barrage
            if extract_time.get(int(line[0])) == None:
                extract_time[int(line[0])] = [float(line[3]), float(line[4]), float(line[5])]
            else:
                extract_time[int(line[0])] = extract_time[int(line[0])] + [float(line[3]), float(line[4]), float(line[5])]
        elif len(line) == 3:
            # comment
            pass
        else:
            print(line)
            print("Error")
    return extract_time

# def get_duration(file_name):
# def get_duration(input_video):
#     result = subprocess.Popen('ffprobe -i input_video -show_entries format=duration -v quiet -of csv="p=0"', stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
#     output = result.communicate()
#     print(output[0])
#     return output[0]

def get_duration(file_name):
    """get the duration of the time
    """
    # print('get_duration', file_name)
    import re
    video_info = subprocess.Popen(["ffprobe", file_name],stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
    video_info  = video_info.stdout.readlines()
    # print(video_info)
    duration_info = [str(x) for x in video_info if "Duration" in str(x)]
    # print(duration_info)
    # if duration_info == []:
    #     return []
    duration_str = re.findall(r"Duration: (.+?),", duration_info[0])
    #print(duration_str)

    h, m ,s = duration_str[0].split(":")
    duration = int(h)*3600 + int(m)*60 + float(s)

    return duration


def extract_audio(extract_time, data_dir):
    problem_video = []
    logger = logging.getLogger("extract_frames")
    # traverse the extract time
    for key, value in tqdm.tqdm(extract_time.items()):
        video_id = key
        video_name = data_dir + "/" + str(video_id) + "/" + str(video_id) + ".flv"
        if os.path.exists("./"+video_name) == False:
            video_name = data_dir + "/" + str(video_id) + "/" + str(video_id) + ".mp4"
        if os.path.exists("./"+video_name) == False:
            problem_video.append(video_id)
            logger.error("%s not exists" %("./"+video_name))
            continue
        # get the duration of the video
        duration = get_duration(video_name)
        #print("%s duration: %f"%(video_id, duration))
        # sort the time
        sort_time = np.sort(value)
        # construct the shell cmd
        if len(sort_time) != 30:
            logger.error("%s file Only %d frames" %(video_id, len(sort_time)))
            problem_video.append(video_id)
            continue
        if sort_time[1] < 2:
           sort_time[1] = 2
        if sort_time[-1] - sort_time[-2] < 2.5:
          sort_time[-2] = sort_time[-1] - 2.5
        for i in range(len(sort_time)//3):
            t = sort_time[3*i+1]
            if t < 2.3:
                t = 2.3
            elif t > duration + 0.5:
                logger.error("The cluster time %f is exceed the duration of the video %s" %(t, video_id))
                problem_video.append(video_id)
                break
            elif t > duration - 3:
                t = duration - 3
            t = t - 2
            shell_cmd_audio = "ffmpeg -ss "+str(t)+" -i "+video_name+" -t 4  "+"audio/"+str(video_id)+"_"+str(i)+".mp3" +" -v error -y"
            out = subprocess.getstatusoutput(shell_cmd_audio)
            if out[1] != "":
                if "element type mismatch 1 != 0" in out[1]:
                    pass
                else:
                    # print(out[1])
                    logger.error("Error in %d" %(video_id))
                    logger.error(shell_cmd_audio)
                    logger.error(out[1])
                    problem_video.append(video_id)
                    break
            #check if the file extracted success
            ext_file_name = "./audio/" + str(video_id) + "_" + str(i) + ".mp3"
            if os.path.exists(ext_file_name) == False:
                log.error("video %d Extract Failed" % video_id)
                log.error(shell_cmd_audio)
                log.error(out[1])
                problem_video.append(video_id)
                break
            else:
                # check the file length is right
                sound_clip, s = librosa.load(ext_file_name)
                if sound_clip.shape[0] < 66150:
                    logger.error("the audio extract length is less than 66150!")
                    logger.error(shell_cmd_audio)
                    problem_video.append(video_id)
                    break
    for key in list(set(problem_video)):
        extract_time.pop(key)
    # record the problem video id
    f = open("failed_video_id.txt",  "a")
    for x in problem_video:
        f.writelines(str(x) + "\n")
    f.close()


    return list(extract_time.keys())

if __name__ == "__main__":
    extract_time = read_file("doc2id.txt")
    LOG_FILE = 'extract_frames.log'
    handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes = 1024*1024, backupCount = 5)
    fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'
    formatter = logging.Formatter(fmt)
    handler.setFormatter(formatter)
    logger = logging.getLogger('extract_frames')
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    extract_audio(extract_time, "dataset")
