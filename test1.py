import os
import re
import subprocess
import datetime

# fileName = r'Sports_360P.mkv'
# filePath = r'/Users/hanwu/Downloads/video_clips/Sports/' + fileName

raw_file_list = []
h264_file_list = []
file_failed_list = []

def getFileSize(filePath):
    fsize = os.path.getsize(filePath)#为什么和我直接看文件属性不一样？
    fsize = fsize / float(1024 * 1024)

    return round(fsize, 2)


def getFilesPath(path):
    file_list = os.listdir(path)
    for file_name in file_list:
        new_path = os.path.join(path, file_name)
        if os.path.isdir(new_path):
            getFilesPath(new_path)
        elif os.path.isfile(new_path):
            raw = re.match(r".+_360P\.mkv$", new_path)
            h264 = re.match(r".+_264\.mp4$", new_path)
            if raw:
                raw_file_list.append(new_path)
            elif h264:
                h264_file_list.append(new_path)
        else:
            print("It's not a directory or a file.")


def fileEncoding(file_list):
    print("--- start encoding with H.264 ---")
    codePre = "ffmpeg -threads 1 -i "
    codeMid = " -vcodec libx264 "

    for file_path in file_list:
        subname = file_path.split('.')
        # print(subname)
        output_path = subname[0] + "_h264.mp4"
        command = codePre + file_path + codeMid + output_path

        file_name = os.path.basename(file_path).split('.')
        new_file_name = os.path.basename(output_path).split('.')

        starttime = datetime.datetime.now()

        retcode = subprocess.call(command, shell=True)
        if retcode == 0:
            print("--- " + file_name[0] + " succeeded ---")
        else:
            file_failed_list.append(file_path)
            print("--- " + file_name[0] + " failed ---")

        endtime = datetime.datetime.now()
        print("Runtime of this encoding is %f s." % (endtime - starttime).total_seconds())

        message = "stream=codec_name,codec_type,duration,width,height,r_frame_rate,bit_rate"
        print("--- info of " + file_name[0] + " ---")
        subprocess.call("ffprobe -v error -show_entries " + message + " -of default=noprint_wrappers=1 " + file_path, shell=True)
        size = getFileSize(file_path)
        print("file_size：%.2f MB" % size)
        print("--- info of " + new_file_name[0] + " ---")
        subprocess.call("ffprobe -v error -show_entries " + message + " -of default=noprint_wrappers=1 " + output_path, shell=True)
        size = getFileSize(output_path)
        print("file_size：%.2f MB" % size)

    print("--- end encoding with H.264 ---")
    print("failed:", file_failed_list)

'''

def fileDecoding(file_list):
    # the decoding function doesn't work
    print("--- start decoding with H.264 ---")
    codePre = "ffmpeg -threads 1 -i "
    # codeMid = " -vcodec h264 "
    codeMid = " -vcodec h264 "

    for file_path in file_list:
        subname = file_path.split('.')
        output_path = subname[0] + "_decoded.mkv"
        command = codePre + file_path + codeMid + output_path
        file_name = os.path.basename(file_path).split('.')
        new_file_name = os.path.basename(output_path).split('.')

        retcode = subprocess.call(command, shell=True)
        if retcode == 0:
                print("--- " + file_name[0] + " succeeded ---")
        else:
            file_failed_list.append(file_path)
            print("--- " + file_name[0] + " failed ---")
        message = "stream=codec_name,codec_type,duration,width,height,display_aspect_ratio,r_frame_rate,bit_rate"
        print("--- info of " + file_name[0] + " ---")
        subprocess.call("ffprobe -v error -show_entries " + message + " -of default=noprint_wrappers=1 " + file_path, shell=True)
        size = getFileSize(file_path)
        print("file_size：%.2f MB" % size)
        print("--- info of " + new_file_name[0] + " ---")
        subprocess.call("ffprobe -v error -show_entries " + message + " -of default=noprint_wrappers=1 " + output_path, shell=True)
        size = getFileSize(output_path)
        print(output_path + "：%.2f MB" % size)

    print("--- end decoding with H.264 ---")
    print("failed:", file_failed_list)
'''

if __name__ == '__main__':
    file_path = r'/media/data/hanwu/video_clips/Sports'
    getFilesPath(file_path)
    fileEncoding(raw_file_list)
    getFilesPath(file_path)
    # fileDecoding(h264_file_list)