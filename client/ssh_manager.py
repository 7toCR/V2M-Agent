"""
SSH隧道管理器 - 自动管理SSH隧道的创建和关闭
"""

import subprocess
import sys
import time
import os
import signal
import platform
import socket
import threading
from contextlib import contextmanager
from typing import Any, Dict, Optional

try:
    from client_config import load_client_server_config
except ImportError:
    from .client_config import load_client_server_config

# 尝试导入paramiko，如果不可用则使用subprocess
try:
    import paramiko
    PARAMIKO_AVAILABLE = True
except ImportError:
    PARAMIKO_AVAILABLE = False

class SSHTunnelManager:
    """SSH隧道管理器类"""
    
    def __init__(
        self,
        ssh_host: Optional[str] = None,
        ssh_port: Optional[int] = None,
        ssh_username: Optional[str] = None,
        ssh_password: Optional[str] = None,
        remote_host: Optional[str] = None,
        remote_port: Optional[int] = None,
        local_port: Optional[int] = None,
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化SSH隧道管理器
        
        Args:
            ssh_host: SSH服务器地址
            ssh_port: SSH服务器端口
            ssh_username: SSH用户名
            ssh_password: SSH密码
            remote_host: 远程服务地址
            remote_port: 远程服务端口
            local_port: 本地映射端口
        """
        self.config = config or load_client_server_config(config_path)
        ssh_config = self.config.get("ssh", {})
        timeout_config = self.config.get("timeouts", {})

        self.ssh_host = ssh_host if ssh_host is not None else str(ssh_config.get("ssh_host", "")).strip()
        self.ssh_port = int(ssh_port if ssh_port is not None else ssh_config.get("ssh_port", 22))
        self.ssh_username = ssh_username if ssh_username is not None else str(ssh_config.get("ssh_username", "")).strip()
        self.ssh_password = ssh_password if ssh_password is not None else str(ssh_config.get("ssh_password", ""))
        self.remote_host = remote_host if remote_host is not None else str(ssh_config.get("remote_host", "")).strip()
        self.remote_port = int(remote_port if remote_port is not None else ssh_config.get("remote_port", 6006))
        self.local_port = int(local_port if local_port is not None else ssh_config.get("local_port", 6006))
        self.connect_timeout = int(timeout_config.get("connect", 20))
        self.process: Optional[subprocess.Popen] = None
        self.ssh_client = None
        self.local_server: Optional[socket.socket] = None
        self.forward_thread: Optional[threading.Thread] = None
        self.is_windows = platform.system() == "Windows"
        self._reconnect_lock = threading.Lock()

    def _is_placeholder(self, value: Optional[str]) -> bool:
        return value is None or str(value).strip() in {"", "CHANGE_ME"}

    def _validate_config(self) -> bool:
        """Validate required SSH tunnel config before opening a connection."""
        missing = []
        if self._is_placeholder(self.ssh_host):
            missing.append("ssh.ssh_host")
        if self._is_placeholder(self.ssh_username):
            missing.append("ssh.ssh_username")
        if self._is_placeholder(self.ssh_password):
            missing.append("ssh.ssh_password")
        if self._is_placeholder(self.remote_host):
            missing.append("ssh.remote_host")

        if missing:
            print(
                f"❌ SSH配置不完整，请在 client/config_client_server.json 中设置: {', '.join(missing)} "
                "(copy from client/config_client_server.example.json)"
            )
            return False
        return True
        
    def _check_port_available(self, port: int) -> bool:
        """检查端口是否可用"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex(('localhost', port))
                return result != 0  # 0表示端口被占用
        except Exception:
            return True
    
    def _start_with_paramiko(self) -> bool:
        """使用paramiko启动隧道（推荐方式）"""
        try:
            # 创建SSH客户端
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            # 连接到SSH服务器
            self.ssh_client.connect(
                hostname=self.ssh_host,
                port=self.ssh_port,
                username=self.ssh_username,
                password=self.ssh_password,
                timeout=self.connect_timeout,
                look_for_keys=False,
                allow_agent=False
            )

            # 设置keepalive以保持连接活跃
            transport = self.ssh_client.get_transport()
            if transport:
                transport.set_keepalive(30)  # 每30秒发送keepalive

            # 在后台线程中启动本地服务器
            # 注意：不在此时创建通道，而是在客户端连接时按需创建
            self._start_local_forward_server()

            # 等待隧道建立
            if self._wait_for_tunnel():
                print(f"✅ SSH隧道已建立 (使用paramiko): localhost:{self.local_port} -> {self.remote_host}:{self.remote_port}")
                return True
            else:
                print("⚠️ SSH隧道启动超时")
                return False
                
        except Exception as e:
            print(f"使用paramiko启动隧道失败: {e}")
            if hasattr(self, 'ssh_client'):
                try:
                    self.ssh_client.close()
                except:
                    pass
                self.ssh_client = None
            return False

    def _reconnect_paramiko_client(self) -> bool:
        """重建paramiko连接，不中断本地转发线程。"""
        if not PARAMIKO_AVAILABLE:
            return False

        with self._reconnect_lock:
            if self.ssh_client is None:
                return False

            try:
                new_client = paramiko.SSHClient()
                new_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                new_client.connect(
                    hostname=self.ssh_host,
                    port=self.ssh_port,
                    username=self.ssh_username,
                    password=self.ssh_password,
                    timeout=self.connect_timeout,
                    look_for_keys=False,
                    allow_agent=False,
                )

                transport = new_client.get_transport()
                if transport:
                    transport.set_keepalive(30)

                old_client = self.ssh_client
                self.ssh_client = new_client
                try:
                    if old_client:
                        old_client.close()
                except Exception:
                    pass

                print("✅ SSH连接已重建（paramiko）")
                return True
            except Exception as e:
                print(f"⚠️ SSH连接重建失败: {e}")
                return False
    
    def _start_local_forward_server(self):
        """启动本地转发服务器（用于paramiko）"""
        def forward_handler():
            channel_error_count = 0
            try:
                # 创建本地服务器socket
                server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server_socket.settimeout(1.0)  # 设置超时以便能够检查停止标志
                server_socket.bind(('localhost', self.local_port))
                server_socket.listen(5)
                self.local_server = server_socket
                
                while self.ssh_client is not None:
                    try:
                        client_socket, addr = server_socket.accept()
                        # 创建到远程的连接
                        transport = self.ssh_client.get_transport()
                        if transport is None or not transport.is_active():
                            client_socket.close()
                            print("⚠️ SSH传输已失活，尝试重建连接...")
                            self._reconnect_paramiko_client()
                            time.sleep(1)
                            continue
                            
                        try:
                            remote_channel = transport.open_channel(
                                'direct-tcpip',
                                (self.remote_host, self.remote_port),
                                (addr[0], addr[1])
                            )
                            channel_error_count = 0
                            
                            # 检查通道是否成功打开
                            if remote_channel is None:
                                raise Exception(f"无法打开到 {self.remote_host}:{self.remote_port} 的通道")
                        except Exception as channel_error:
                            client_socket.close()
                            channel_error_count += 1
                            error_msg = str(channel_error)
                            if "No route to host" in error_msg or "Connect failed" in error_msg:
                                print(f"⚠️ 无法连接到远程服务 {self.remote_host}:{self.remote_port}")
                                print(f"   请检查:")
                                print(f"   1. 远程服务是否正在运行")
                                print(f"   2. 远程主机地址是否正确 (当前: {self.remote_host})")
                                print(f"   3. 远程端口是否正确 (当前: {self.remote_port})")
                            else:
                                print(f"⚠️ 打开SSH通道失败: {channel_error}")
                            if transport is None or not transport.is_active():
                                self._reconnect_paramiko_client()
                            elif channel_error_count >= 3:
                                print("⚠️ SSH channel failed repeatedly, rebuilding transport...")
                                if self._reconnect_paramiko_client():
                                    channel_error_count = 0
                                time.sleep(1)
                            # 不中断整个转发服务器，继续监听其他连接
                            continue
                        
                        # 在后台线程中转发数据
                        def forward_data(source, dest):
                            try:
                                while True:
                                    try:
                                        data = source.recv(4096)
                                        if not data:
                                            break
                                        dest.send(data)
                                    except (socket.error, OSError, ConnectionResetError, ConnectionAbortedError):
                                        # 连接已关闭或出错，正常退出
                                        break
                            except Exception:
                                # 忽略其他异常
                                pass
                            finally:
                                # 安全关闭socket
                                try:
                                    if source:
                                        source.close()
                                except (socket.error, OSError, AttributeError):
                                    pass
                                try:
                                    if dest:
                                        dest.close()
                                except (socket.error, OSError, AttributeError):
                                    pass
                        
                        threading.Thread(
                            target=forward_data,
                            args=(client_socket, remote_channel),
                            daemon=True
                        ).start()
                        threading.Thread(
                            target=forward_data,
                            args=(remote_channel, client_socket),
                            daemon=True
                        ).start()
                    except socket.timeout:
                        # 超时是正常的，继续循环
                        continue
                    except (socket.error, OSError, ConnectionResetError, ConnectionAbortedError) as e:
                        if self.ssh_client is None:
                            break
                        time.sleep(0.2)
                        continue
                    except Exception as e:
                        if self.ssh_client is not None:
                            print(f"转发连接错误: {e}")
                        time.sleep(0.2)
                        continue
            except (socket.error, OSError) as e:
                # Socket错误，可能是正常关闭
                if self.ssh_client is not None:
                    # 只有在连接仍然存在时才记录
                    pass
            except Exception as e:
                if self.ssh_client is not None:
                    print(f"本地转发服务器错误: {e}")
            finally:
                try:
                    if self.local_server:
                        self.local_server.close()
                except (socket.error, OSError, AttributeError):
                    # 忽略关闭时的错误
                    pass
        
        self.forward_thread = threading.Thread(target=forward_handler, daemon=True)
        self.forward_thread.start()
        time.sleep(0.5)  # 等待服务器启动
    
    def _wait_for_tunnel(self, timeout: Optional[int] = None) -> bool:
        """等待隧道建立"""
        import socket
        if timeout is None:
            timeout = self.connect_timeout
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1)
                    result = s.connect_ex(('localhost', self.local_port))
                    if result == 0:
                        return True
            except Exception:
                pass
            time.sleep(0.5)
        return False
    
    def start(self) -> bool:
        """
        启动SSH隧道
        
        Returns:
            bool: 是否成功启动
        """
        if self.process is not None or self.ssh_client is not None:
            # 仅在隧道确实可用时复用
            if self.is_running():
                return True
            # 清理失活实例后重建
            self.stop()
            time.sleep(0.5)

        if not self._validate_config():
            return False
        
        # 检查端口是否可用
        if not self._check_port_available(self.local_port):
            print(f"警告: 本地端口 {self.local_port} 已被占用，尝试使用现有连接")
            # 假设端口被占用是因为隧道已经存在
            return True
        
        try:
            # 优先使用paramiko（如果可用），因为它能更好地处理密码认证
            if PARAMIKO_AVAILABLE:
                return self._start_with_paramiko()
            
            # 回退到subprocess方式
            if self.is_windows:
                # Windows系统：尝试使用plink
                return self._start_with_plink() or self._start_with_ssh()
            else:
                # Linux/Mac系统：使用ssh
                return self._start_with_ssh()
        except Exception as e:
            print(f"启动SSH隧道失败: {e}")
            return False

    def reconnect(self) -> bool:
        """强制重建SSH隧道。"""
        self.stop()
        time.sleep(1)
        return self.start()
    
    def _start_with_plink(self) -> bool:
        """使用plink启动隧道（Windows）"""
        plink_path = "plink.exe"
        
        # 检查plink是否可用
        try:
            subprocess.run(
                [plink_path, "-V"],
                capture_output=True,
                check=True,
                timeout=5
            )
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False
        
        # 构建plink命令
        plink_command = [
            plink_path,
            "-P", str(self.ssh_port),
            "-L", f"{self.local_port}:{self.remote_host}:{self.remote_port}",
            "-N",  # 不执行远程命令
            "-pw", self.ssh_password,  # 密码
            "-batch",  # 非交互模式
            f"{self.ssh_username}@{self.ssh_host}"
        ]
        
        try:
            # 启动plink进程
            self.process = subprocess.Popen(
                plink_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if self.is_windows else 0
            )
            
            # 等待隧道建立
            if self._wait_for_tunnel():
                print(f"✅ SSH隧道已建立: localhost:{self.local_port} -> {self.remote_host}:{self.remote_port}")
                return True
            else:
                print("⚠️ SSH隧道启动超时，但进程已启动")
                return True  # 即使超时也认为成功，因为进程已启动
                
        except Exception as e:
            print(f"使用plink启动隧道失败: {e}")
            if self.process:
                self.process.terminate()
                self.process = None
            return False
    
    def _start_with_ssh(self) -> bool:
        """使用ssh命令启动隧道"""
        # 构建SSH命令
        ssh_command = [
            "ssh",
            "-p", str(self.ssh_port),
            "-L", f"{self.local_port}:{self.remote_host}:{self.remote_port}",
            "-N",  # 不执行远程命令，只用于端口转发
            "-o", "StrictHostKeyChecking=no",  # 跳过主机密钥检查
            "-o", "UserKnownHostsFile=/dev/null" if not self.is_windows else "UserKnownHostsFile=NUL",
            "-o", "BatchMode=yes",  # 非交互模式（但需要密钥或expect）
            f"{self.ssh_username}@{self.ssh_host}"
        ]
        
        try:
            # 对于需要密码的情况，我们需要使用expect或sshpass
            # 但为了简化，我们先尝试使用ssh密钥
            # 如果没有密钥，可以使用expect脚本
            
            # 尝试使用sshpass（如果可用）
            if self._try_sshpass(ssh_command):
                return True
            
            # 如果没有sshpass，尝试使用expect（如果可用）
            if self._try_expect(ssh_command):
                return True
            
            # 最后尝试直接启动（可能需要手动输入密码或使用密钥）
            self.process = subprocess.Popen(
                ssh_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if self.is_windows else 0
            )
            
            # 等待一下看是否成功
            time.sleep(2)
            if self.process.poll() is None:
                # 进程还在运行
                if self._wait_for_tunnel():
                    print(f"✅ SSH隧道已建立: localhost:{self.local_port} -> {self.remote_host}:{self.remote_port}")
                    return True
                else:
                    print("⚠️ SSH隧道启动超时，但进程已启动")
                    return True
            else:
                # 进程已退出，可能失败
                stderr = self.process.stderr.read().decode('utf-8', errors='ignore') if self.process.stderr else ""
                print(f"SSH隧道启动失败: {stderr}")
                self.process = None
                return False
                
        except FileNotFoundError:
            print("错误: 未找到ssh命令。请确保已安装OpenSSH客户端。")
            return False
        except Exception as e:
            print(f"启动SSH隧道失败: {e}")
            if self.process:
                self.process.terminate()
                self.process = None
            return False
    
    def _try_sshpass(self, ssh_command: list) -> bool:
        """尝试使用sshpass启动隧道"""
        try:
            sshpass_command = [
                "sshpass",
                "-p", self.ssh_password
            ] + ssh_command
            
            self.process = subprocess.Popen(
                sshpass_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if self.is_windows else 0
            )
            
            time.sleep(2)
            if self.process.poll() is None:
                if self._wait_for_tunnel():
                    print(f"✅ SSH隧道已建立 (使用sshpass): localhost:{self.local_port} -> {self.remote_host}:{self.remote_port}")
                    return True
            return False
        except FileNotFoundError:
            return False
        except Exception:
            if self.process:
                self.process.terminate()
                self.process = None
            return False
    
    def _try_expect(self, ssh_command: list) -> bool:
        """尝试使用expect启动隧道"""
        try:
            # 创建expect脚本
            expect_script = f"""
spawn {' '.join(ssh_command)}
expect {{
    "*password*" {{
        send "{self.ssh_password}\\r"
        exp_continue
    }}
    "*yes/no*" {{
        send "yes\\r"
        exp_continue
    }}
}}
interact
"""
            # 这里简化处理，实际需要写入临时文件
            return False  # 暂时不实现expect方式
        except Exception:
            return False
    
    def stop(self):
        """停止SSH隧道"""
        # 关闭paramiko连接
        if self.ssh_client is not None:
            try:
                # 先标记为关闭，停止转发线程
                ssh_client_backup = self.ssh_client
                self.ssh_client = None
                
                # 关闭本地服务器
                if self.local_server:
                    try:
                        self.local_server.close()
                    except (socket.error, OSError, AttributeError):
                        # 忽略关闭时的错误
                        pass
                    self.local_server = None
                
                # 然后关闭SSH连接
                if ssh_client_backup:
                    try:
                        transport = ssh_client_backup.get_transport()
                        if transport:
                            try:
                                transport.close()
                            except:
                                pass
                        ssh_client_backup.close()
                    except (socket.error, OSError, AttributeError):
                        # 忽略关闭时的错误
                        pass
                
                print(f"✅ SSH隧道已关闭 (paramiko)")
            except Exception as e:
                # 不打印错误，因为可能是正常的关闭过程
                pass
            finally:
                self.ssh_client = None
                self.local_server = None
                self.forward_thread = None
        
        # 关闭subprocess进程
        if self.process is not None:
            try:
                if self.is_windows:
                    # Windows使用taskkill
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(self.process.pid)],
                        capture_output=True,
                        timeout=5
                    )
                else:
                    # Linux/Mac使用kill
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait()
                
                print(f"✅ SSH隧道已关闭")
            except Exception as e:
                print(f"关闭SSH隧道时出错: {e}")
            finally:
                self.process = None
    
    def is_running(self) -> bool:
        """检查隧道是否正在运行"""
        # 检查paramiko连接
        if self.ssh_client is not None:
            try:
                transport = self.ssh_client.get_transport()
                return transport is not None and transport.is_active()
            except:
                return False
        
        # 检查subprocess进程
        if self.process is None:
            return False
        return self.process.poll() is None
    
    @contextmanager
    def tunnel(self):
        """
        上下文管理器，自动管理SSH隧道的生命周期
        
        使用示例:
            with tunnel_manager.tunnel():
                # 在这里使用隧道
                client = MusicGenerationClient("http://localhost:6006")
                ...
        """
        started = False
        try:
            if self.start():
                started = True
                yield
            else:
                raise Exception("无法启动SSH隧道")
        finally:
            if started:
                self.stop()


# 全局隧道管理器实例
_tunnel_manager: Optional[SSHTunnelManager] = None


def get_tunnel_manager() -> SSHTunnelManager:
    """获取全局隧道管理器实例"""
    global _tunnel_manager
    if _tunnel_manager is None:
        _tunnel_manager = SSHTunnelManager()
    return _tunnel_manager


def create_tunnel_manager(
    ssh_host: Optional[str] = None,
    ssh_port: Optional[int] = None,
    ssh_username: Optional[str] = None,
    ssh_password: Optional[str] = None,
    remote_host: Optional[str] = None,
    remote_port: Optional[int] = None,
    local_port: Optional[int] = None,
    config_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> SSHTunnelManager:
    """
    创建并配置隧道管理器
    
    Returns:
        SSHTunnelManager: 配置好的隧道管理器
    """
    global _tunnel_manager
    _tunnel_manager = SSHTunnelManager(
        ssh_host=ssh_host,
        ssh_port=ssh_port,
        ssh_username=ssh_username,
        ssh_password=ssh_password,
        remote_host=remote_host,
        remote_port=remote_port,
        local_port=local_port,
        config_path=config_path,
        config=config,
    )
    return _tunnel_manager

