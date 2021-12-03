import os
import shutil
import datetime

def clean(exp_name):
    log_root = './history/{}/{}'.format(get_month_day(), exp_name)
    if os.path.exists(log_root):
        answer = input('Find folder of the same name {}! Continue?[y/n]'.format(log_root))
        if 'y' not in answer:
            exit(0)

        shutil.rmtree(log_root)

    if not os.path.exists('./history/{}/'.format(get_month_day())):
        os.mkdir('./history/{}/'.format(get_month_day()))

    os.mkdir(log_root)


def get_all_py_file(root, py_list):
    files = os.listdir(root)
    for file in files:
        if os.path.isdir(file) and 'history' not in file and 'backup' not in file:
            get_all_py_file(os.path.join(root, file), py_list)
        else:
            if file.endswith('py'):
                py_list.append(os.path.join(root, file))


def backup(exp_name):
    os.mkdir('./history/{}/{}/backup_pyfile'.format(get_month_day(), exp_name))
    py_list = []
    get_all_py_file('.', py_list)
    print('py_list', py_list)
    for file in py_list:
        shutil.copy(file,
            os.path.join('./history/{}/{}/backup_pyfile'.format(get_month_day(), exp_name), file.split('/')[-1]))

month_day = None
def get_month_day():
    global month_day
    if month_day is None:
        month_day = '{}_{}'.format(datetime.datetime.now().month, datetime.datetime.now().day)
        return month_day
    else:
        return month_day
    # return '{}_{}'.format(datetime.datetime.now().month, datetime.datetime.now().day)