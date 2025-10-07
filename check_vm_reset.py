import subprocess
import time
import os
import paramiko

# Configuration
VM_HOST = "paffenroth-23.dyn.wpi.edu"
VM_PORT = 22015
ADMIN_USER = "student-admin"
ADMIN_KEY_PATH = "student-admin_key"    # no passphrase key to detect VM reset & append new key
NEW_KEY_PREFIX = "new_key"               # new key pair to access VM post reset
LOCAL_DEPLOY_SCRIPT = "deploy.sh"
REMOTE_DEPLOY_SCRIPT = "deploy.sh"
SLEEP_INTERVAL = 60  # seconds between checks

def vm_reset_detected(host, port, key):
    try:
        result = subprocess.run([
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-i", key,
            "-p", str(port),
            f"{ADMIN_USER}@{host}",
            "echo 'VMRESET'"], capture_output=True, timeout=10)
        return b"VMRESET" in result.stdout
    except Exception:
        return False


def remove_known_host(host, port):
    subprocess.run(["ssh-keygen", "-R", f"[{host}]:{port}"])

def generate_new_keypair(key_prefix):
    subprocess.run([
        "ssh-keygen",
        "-f", key_prefix,
        "-t", "ed25519",
        "-N", ""
    ])

def comment_first_line_and_append_pubkeys(host, port, user, admin_key, new_key_path, extra_public_keys):
    key = paramiko.Ed25519Key(filename=admin_key)
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port=port, username=user, pkey=key)
    sftp = ssh.open_sftp()
    sftp.chdir('.ssh')
    # Download the current authorized_key file
    with sftp.open('authorized_keys', 'r') as remotefile:
        lines = remotefile.readlines()
    # Comment out the first line if not already commented
    if lines and not lines[0].startswith('#'):
        lines[0] = '# ' + lines[0]
    # Prepare new keys to append
    with open(new_key_path, 'r') as pubkey_file:
        new_key = pubkey_file.read().rstrip('\n') + '\n'
    # The three extra keys you provided
    extra_keys = [
        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJjZ1gb8pjP8MhZOZ0yolIv3k15uTyj9C8U+WpXggfnM pradyumn k tendulkar@DESKTOP-GGHK3QD\n",
        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBbPohsgVrp1B1+BH2YGgImCp8wDc2Hfl7s0T2m3Dqqn rutu@ChetanKharkar21\n",
        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOZ13x2Qanfbrc8r3PlVJc6n/0Mwnv3kEGuBQBU+/qfO bathu@NehaBathuri\n"
    ]
    # Append all four keys
    lines += [new_key] + extra_keys
    # Write the updated authorized_key file back
    with sftp.open('authorized_keys', 'w') as remotefile:
        remotefile.writelines(lines)
    sftp.close()
    ssh.close()
    print("Commented first line and appended new and extra public keys to remote authorized_key.")



def upload_and_run_deploy_script_with_new_key(host, port, user, new_key_path, local_script, remote_script):
    key = paramiko.Ed25519Key(filename=new_key_path)
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port=port, username=user, pkey=key)
    sftp = ssh.open_sftp()
    sftp.put(local_script, remote_script)
    sftp.close()
    # Run deploy.sh with sudo and arguments
    command = f"sudo ./{remote_script} /tmp/team_keys \"student-admin default key\""
    stdin, stdout, stderr = ssh.exec_command(command)
    output = stdout.read().decode()
    error = stderr.read().decode()
    ssh.close()
    if output:
        print(f"deploy.sh output:\n{output}")
    if error:
        print(f"deploy.sh error:\n{error}")


def main():
    while True:
        print("Checking VM reset status...")
        if vm_reset_detected(VM_HOST, VM_PORT, ADMIN_KEY_PATH):
            print("VM reset detected! Securing VM and deploying...")
            remove_known_host(VM_HOST, VM_PORT)
            for fname in [NEW_KEY_PREFIX, f"{NEW_KEY_PREFIX}.pub"]:
                try:
                    os.remove(fname)
                except FileNotFoundError:
                    pass
            generate_new_keypair(NEW_KEY_PREFIX)
            comment_first_line_and_append_pubkeys(
                VM_HOST, VM_PORT, ADMIN_USER, ADMIN_KEY_PATH, f"{NEW_KEY_PREFIX}.pub", None)


            # Now upload and run deploy.sh using new key
            upload_and_run_deploy_script_with_new_key(
                VM_HOST, VM_PORT, ADMIN_USER, NEW_KEY_PREFIX, LOCAL_DEPLOY_SCRIPT, REMOTE_DEPLOY_SCRIPT)
            print("Recovery, securing, and deployment completed.")
        else:
            print("VM not reset, no action needed.")
        time.sleep(SLEEP_INTERVAL)

if __name__ == "__main__":
    main()
