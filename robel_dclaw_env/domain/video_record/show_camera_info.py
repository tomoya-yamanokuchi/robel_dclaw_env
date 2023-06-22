import subprocess

def main():
    cmd            = 'sudo v4l2-ctl --device /dev/video2 --list-formats-ext' # videoの数字は適切に設定する
    proc           = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    outs_bytes     = proc.communicate()[0]
    outs_str       = outs_bytes.decode('utf-8')
    outs_str_lists = outs_str.split('\n')

    for line in outs_str_lists:
        line_lists = line.split()
        if len(line_lists) > 0 and line_lists[0] == 'Size:':
            print(line_lists)


if __name__ == '__main__':
    main()
