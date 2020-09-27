import os
import vlc
import re
import shlex
import subprocess
import datetime
import csv
from time import sleep
def run_cmd_with_ret(cmd):
    pipe = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell=True)
    stdout, stderr = pipe.communicate()
    out = "".join(map(chr, stderr)) + "".join(map(chr, stdout))
    
    return out

def get_video_info(video_file = 'v0.mkv'):
    cmd = 'ffmpeg -i {}'.format(video_file)
    out = run_cmd_with_ret(cmd)
    #print(out)
    ret = [-1,-1,-1, -1, -1, -1, -1, -1]
    for line in out.split('\n'):
        if 'bitrate' in line:
            print(line)
            words = line.split(',')
            print(words)
            ret[0] = words[0].strip().split(' ')[1] #duration
            ret[1] = words[-1].strip().split(' ')[1] # bitrate
        if 'Stream' in line and 'Video' in line:
            #print(line)
            words = line.split(',')
            #print(words)
            ret[2] = words[-5].strip().split(' ')[0]
            if 'SAR' in  words[-5] or 'kb' in words[-5]:
                ret[2] = words[-6].strip().split(' ')[0]
            ret[3] = words[-4].strip().split(' ')[0]
            ret[4] = words[-3].strip().split(' ')[0]
            ret[5] = words[-2].strip().split(' ')[0]
            ret[6] = words[-1].strip().split(' ')[0]
            
    cmd = 'ls -l --block-size=k {}'.format(video_file)
    out1 = run_cmd_with_ret(cmd)
    ret[7] = out1.split(' ')[4]
    #print(ret)

    return ret, out + 'File Size' +  out1
            
            
            
    
def test_encoder(video_file = 'v0.mkv', encode_file = 'encode.mp4', encoder = 'h264'):
    if encoder == 'av1':
        cmd = "ffmpeg -i {} -vcodec {} -strict -2 {} ".format(video_file, encoder, encode_file)
    else:
        cmd = "ffmpeg -i {} -vcodec {} -crf 22 {} ".format(video_file, encoder, encode_file)
    t0 = datetime.datetime.now()
    os.system(cmd)
    t1 = datetime.datetime.now()
    return t0, t1

def test_play(video_file = 'encode.mp4', loop = 1):
    cmd = 'vlc --intf dummy --play-and-exit  {} '.format(video_file)
    #--input-repeat 3
    t0 = datetime.datetime.now()
    out = run_cmd_with_ret(cmd)
    t1 = datetime.datetime.now()
    #print(out)
    return t0,t1 
    
def finalize(encode_file = 'encode.mp4'):
    os.system('rm {}'.format(encode_file))

def run_test(video_file, encode_file, encoder, plays): 
    finalize(encode_file)
    sleep(5)
    enc_t0, enc_t1 = test_encoder(video_file, encode_file, encoder)
    sleep(5)
    play_t0, play_t1 = test_play(encode_file, plays);
    video_info_0, log0 = get_video_info(video_file)
    video_info_enc, log1 = get_video_info(encode_file)
    ret = [enc_t0, enc_t1, play_t0, play_t1] + video_info_0 + [video_info_enc[-1]]
    finalize(encode_file)
    return ret, log0 + '\n' + log1
    
def run_tests():
    Inpath = '/media/data/hanwu/video_clips1/original_videos/'
    video_types = os.listdir(Inpath)
    #video_types.remove('CoverSong')
    #video_types.remove('Sports')
    #video_types.remove('NewsClip')
    #print(video_types)
    with open('out.csv', 'w', newline='') as fout, open('run_log.txt', 'w') as flog:
        header = ['path','video type', 'file name', 'encoder', 'plays','enc_t0', 'enc_t1', \
                  'play_t0', 'play_t1', 'duration', 'bitrate', \
                  'resolution', 'fps', 'tbr', 'tbn', 'tbc', 'size0', 'size_enc']
        wter = csv.writer(fout)
        wter.writerow(header)
        video_types = ['Animation',   'Gaming',  'HDR',  'HowTo',   'MusicVideo',  'VR']
        for video_type in video_types:
            path = Inpath + video_type + '/'
            pixtypes = os.listdir(path)
            for pix in pixtypes:
                if video_type == 'Animation' and pix == '1080P':
                    continue
                subpath = path + pix + '/'
                files = os.listdir(subpath)
                for file in files:
                    print(subpath+file)
                    for encoder in ['libx264', 'hevc', 'vp8', 'vp9']:
                        for plays in [i for i in range(1, 2)]:
                            ret, log = run_test(subpath + file, 'tmp_encoder.mkv', encoder, plays)
                            ret = [subpath, video_type, file, encoder, plays] + ret
                            print(ret)
                            wter.writerow(ret)
                            flog.write(log)
    
run_tests()
