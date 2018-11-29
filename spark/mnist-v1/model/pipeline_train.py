import subprocess 

if __name__ == '__main__':
    cmd = 'sbt "runMain pipeline_train"'
    subprocess.call(cmd, shell=True)
