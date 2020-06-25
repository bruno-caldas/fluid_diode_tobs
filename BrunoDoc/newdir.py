import os, shutil, datetime

def erase_old_results(output_dir, hash):
    """
    Creating new folder for next results and copping source code
    """
    current_dir = os.path.dirname(os.path.dirname( __file__ ))#os.getcwd()
    print(current_dir)
    now = datetime.datetime.now().strftime("%Y-%m-%d_%Hh%Mmin%Ss")
    new_dir = current_dir + "/" +output_dir + "_"+str(now) + "_"+hash
    source_code_name = current_dir + '/main.py'
    #verify if exist
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
        os.makedirs(new_dir +"/Source")
    if os.path.exists(new_dir + "/Source/"+source_code_name):
        os.remove(new_dir + "/Source/"+source_code_name)
        os.mknod(new_dir + "/Source/"+source_code_name)
    shutil.copy2(source_code_name,  new_dir + "/Source/main.py")

    def copytree(src, dst, symlinks=False, ignore=None):
        """
        Subfunction that ignores the .pyc files
        """
        os.makedirs(dst)
        for item in os.listdir(src):
            source_dir = os.path.join(src, item)
            if os.path.isfile(source_dir) and not item.endswith('.pyc'):
                shutil.copy2(source_dir, dst)
    copytree(current_dir + '/BrunoDoc',  new_dir + "/Source/BrunoDoc" )
    if os.path.exists(new_dir + "/Source/Objetivo.txt"):
        os.remove(new_dir + "/Source/Objetivo.txt")
        os.mknod(new_dir + "/Source/Objetivo.txt")
    return new_dir
