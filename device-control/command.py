import paramiko
import shutil
import subprocess
import os


def send_folder(raspberries:list, local_folder:str, remote_folder:str):
    for raspberry in raspberries:
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(raspberry["ip"], username=raspberry["username"], password=raspberry["password"])

            sftp = ssh.open_sftp()

            # 递归地复制本地文件夹到远程文件夹
            shutil.make_archive("temp_archive", 'zip', local_folder)
            sftp.put("temp_archive.zip", remote_folder + "/temp_archive.zip")
            sftp.close()

            # 解压缩文件夹
            commands = [
                f"cd {remote_folder} ; unzip temp_archive.zip",
                f"cd {remote_folder} ; rm temp_archive.zip; ls"
            ]
            for c in commands:
                stdin, stdout, stderr = ssh.exec_command(c)
                print(f"Output from {raspberry['ip']}:\n{stdout.read().decode()}")

            ssh.close()

            print(f"Sent {local_folder} to {raspberry['ip']}")
            print("="*80)

        except Exception as e:
            print(f"Failed to send {local_folder} to {raspberry['ip']}: {str(e)}")


def send_file(raspberries:list, local_file:str, remote_folder:str):
    for raspberry in raspberries:
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(raspberry["ip"], username=raspberry["username"], password=raspberry["password"])

            sftp = ssh.open_sftp()
            sftp.put(local_file, remote_folder + "/" + os.path.split(local_file)[-1])
            sftp.close()

            ssh.close()

            print(f"Sent {local_file} to {raspberry['ip']}")

        except Exception as e:
            print(f"Failed to send {local_file} to {raspberry['ip']}: {str(e)}")


def command(raspberry:dict, commands_to_execute:list):
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(raspberry["ip"], username=raspberry["username"], password=raspberry["password"])

        # 执行指令
        for command in commands_to_execute:
            stdin, stdout, stderr = ssh.exec_command(command)

            # 打印命令输出
            print(f"Output from {raspberry['ip']}:\n{stdout.read().decode()}")
            print("=" * 80)

        # ssh.close()

    except Exception as e:
        print(f"Failed to execute command on {raspberry['ip']}: {str(e)}")


if __name__ == '__main__':
    # 定义树莓派的IP地址、用户名和密码

    raspberries = [
        {"ip": "192.168.3.2", "username": "pi", "password": "raspberry"},
        {"ip": "192.168.3.3", "username": "pi", "password": "raspberry"},
        {"ip": "192.168.3.4", "username": "pi", "password": "raspberry"},
        {"ip": "192.168.3.7", "username": "pi", "password": "raspberry"},
        {"ip": "192.168.3.8", "username": "pi", "password": "raspberry"},
        {"ip": "192.168.3.10", "username": "pi", "password": "raspberry"},
        {"ip": "192.168.3.11", "username": "pi", "password": "raspberry"},
        {"ip": "192.168.3.12", "username": "pi", "password": "raspberry"},
        {"ip": "192.168.3.13", "username": "pi", "password": "raspberry"},
        {"ip": "192.168.3.20", "username": "pi", "password": "raspberry"},
    ]

    jetson_nanos = [
        {"ip": "192.168.3.5", "username": "hao", "password": "929910"},
        {"ip": "192.168.3.6", "username": "hao", "password": "929910"},
        {"ip": "192.168.3.9", "username": "hao", "password": "929910"},
        {"ip": "192.168.3.15", "username": "hao", "password": "929910"},
        {"ip": "192.168.3.16", "username": "hao", "password": "929910"},
        # {"ip": "192.168.3.22", "username": "hao", "password": "929910"},
        # {"ip": "192.168.3.23", "username": "hao", "password": "929910"},
    ]

    ubuntu = [
        {"ip": "192.168.3.17", "username": "hao", "password": "929910"}
    ]

    operation_dict = {
        1: 'send_folder',
        2: 'send_file',
        3: 'command'
    }
    operation = operation_dict[2]

    if operation == 'send_folder':
        # 要发送目标文件夹
        local_folder = r"E:\2024mem\AI-project\Dataset\HAR"
        remote_folder = "/home/hao/work/fl-pj/Dataset/HAR"

        # 发送文件夹
        send_folder(raspberries, local_folder, remote_folder)

    elif operation == 'send_file':
        # 要发送的目标文件
        local_file = r"E:\2024mem\AI-project\Federated-Learning-Platform\my_utils.py"
        remote_folder = "/home/pi/work/fl-pj/federated-learning-platform"

        # 发送文件
        send_file(raspberries, local_file, remote_folder)

    elif operation == 'command':
        # 执行命令
        commands_to_execute = [
            f"mkdir /home/hao/work/fl-pj/federated-learning-platform",
        ]
        for device in jetson_nanos:
            command(device, commands_to_execute)