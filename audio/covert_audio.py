import  os

def convert(infile,outfile,ffmpegPath = 'E://ffmpeg//bin//ffmpeg.exe'):
    cmd = '%s -i %s %s'% (ffmpegPath,infile,outfile)
    execCmd(cmd)

def execCmd(cmd):
    """
    执行计算命令时间
    """
    r = os.popen(cmd)
    text = r.read().strip()
    r.close()
    return text


