"""
音乐生成客户端 - 负责读取文件、建立SSH隧道、发送文件并下载生成的音频
"""

import socketio
import requests
import json
import os
import time
import threading
from typing import Any, Dict, Optional, List

from tools.live_status import LiveStatus

try:
    from ssh_manager import SSHTunnelManager
    from file_handler import read_lyric_file, save_audio_file, get_output_dir
except ImportError:
    from .ssh_manager import SSHTunnelManager
    from .file_handler import read_lyric_file, save_audio_file, get_output_dir

try:
    from client_config import build_server_url, load_client_server_config
except ImportError:
    from .client_config import build_server_url, load_client_server_config

def safe_print(message):
    """安全打印函数，处理Windows GBK编码问题"""
    try:
        LiveStatus.print_line(str(message))
    except UnicodeEncodeError:
        safe_message = str(message).encode('ascii', 'ignore').decode('ascii')
        LiveStatus.print_line(safe_message)


class MusicGenerationClient:
    """音乐生成客户端类"""

    MAX_RECONNECT_ATTEMPTS = 10
    INITIAL_RECONNECT_DELAY = 1.0  # 初始重连延迟(秒)
    MAX_RECONNECT_DELAY = 30.0  # 最大重连延迟(秒)
    HEARTBEAT_INTERVAL = 25.0  # 心跳间隔(秒)

    def __init__(
        self,
        server_url: Optional[str] = None,
        auto_ssh: Optional[bool] = None,
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化音乐生成客户端

        Args:
            server_url: 服务器地址，默认为 http://localhost:6006
                       如果客户端和服务器在同一台机器上，可以直接使用 http://localhost:6006
                       如果客户端在远程机器上，需要通过SSH隧道，使用 http://localhost:6006（隧道会自动映射）
            auto_ssh: 是否自动建立SSH隧道，默认为True
                      - True: 自动建立SSH隧道（适用于远程连接）
                      - False: 不建立SSH隧道（适用于本地连接或手动管理隧道）
        """
        self.config = config or load_client_server_config(config_path)
        ssh_config = self.config.get("ssh", {})
        timeout_config = self.config.get("timeouts", {})
        retry_config = self.config.get("retry", {})
        progress_config = self.config.get("progress", {})

        self.server_url = (server_url or build_server_url(self.config)).rstrip("/")
        self.auto_ssh = bool(ssh_config.get("enabled", True)) if auto_ssh is None else bool(auto_ssh)
        self.connect_timeout = int(timeout_config.get("connect", 20))
        self.request_timeout = int(timeout_config.get("request", 30))
        self.status_timeout = int(timeout_config.get("status", 10))
        self.download_timeout = int(timeout_config.get("download", 300))
        self.task_timeout = int(timeout_config.get("task", 3600))
        self.poll_interval = int(progress_config.get("poll_interval", 5))
        self.heartbeat_interval = float(progress_config.get("heartbeat_interval", self.HEARTBEAT_INTERVAL))
        self.max_status_errors = int(retry_config.get("max_status_errors", 5))
        self.http_max_retries = int(retry_config.get("http_max_retries", 3))
        self.max_reconnect_attempts = int(retry_config.get("max_reconnect_attempts", self.MAX_RECONNECT_ATTEMPTS))
        self.initial_reconnect_delay = float(retry_config.get("initial_delay", self.INITIAL_RECONNECT_DELAY))
        self.max_reconnect_delay = float(retry_config.get("max_delay", self.MAX_RECONNECT_DELAY))
        self.sio = socketio.Client(
            reconnection=True,
            reconnection_attempts=self.max_reconnect_attempts,
            reconnection_delay=self.initial_reconnect_delay,
            reconnection_delay_max=self.max_reconnect_delay,
            logger=False,
            engineio_logger=False
        )
        self.task_id = None
        self.task_status = None
        self.connected = False
        self.client_sid = None  # 初始化client_sid
        self.tunnel_manager: Optional[SSHTunnelManager] = None

        # 重连状态管理
        self._reconnect_count = 0
        self._is_reconnecting = False
        self._should_stop = False

        # 心跳线程
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._heartbeat_stop_event = threading.Event()
        self._latest_status: Optional[dict] = None
        self._status_render_lock = threading.Lock()
        self._last_status_key = None
        self._recovery_lock = threading.Lock()
        self._last_recovery_ts = 0.0

        # 注册SocketIO事件处理器
        self._register_handlers()
    
    def _register_handlers(self):
        """注册SocketIO事件处理器"""

        @self.sio.event
        def connect():
            safe_print(f"✅ 已连接到服务器: {self.server_url}")
            self.connected = True
            self._reconnect_count = 0  # 重置重连计数
            self._is_reconnecting = False
            try:
                self.client_sid = self.sio.get_sid()
                safe_print(f"客户端SID: {self.client_sid}")
            except Exception as e:
                safe_print(f"⚠️ 获取客户端SID失败: {e}")

            # 如果有正在运行的任务，重新订阅
            if self.task_id and self.task_status not in ["completed", "failed"]:
                self.subscribe_task(self.task_id)
                status = self._get_task_status_with_retry(self.task_id, max_retries=1)
                if status:
                    self._apply_task_status(status, source="reconnect")

        @self.sio.event
        def disconnect():
            if not self._should_stop:
                safe_print("⚠️ 与服务器断开连接，将尝试重连...")
            self.connected = False

        @self.sio.event
        def connect_error(data):
            safe_print(f"⚠️ SocketIO连接错误: {data}")
            self._is_reconnecting = True

        @self.sio.event
        def connected(data):
            safe_print(f"📢 服务器消息: {data.get('message', '')}")

        @self.sio.event
        def task_started(data):
            safe_print(f"🚀 任务开始: {data.get('message', '')}")
            safe_print(f"   任务ID: {data.get('task_id', '')}")
            payload = dict(data or {})
            payload.setdefault("status", "running")
            payload.setdefault("progress", 0)
            self._apply_task_status(payload, source="socket")

        @self.sio.event
        def task_completed(data):
            safe_print(f"🎉 任务完成: {data.get('message', '')}")
            safe_print(f"   生成的文件: {data.get('files', [])}")
            payload = dict(data or {})
            payload.setdefault("status", "completed")
            payload.setdefault("progress", 100)
            self._apply_task_status(payload, source="socket", force=True)

        @self.sio.event
        def task_failed(data):
            # 服务端会在 error 中塞入更完整的 stderr/stdout 片段（可能很长）
            err = data.get('error', '未知错误')
            safe_print(f"❌ 任务失败: {err}")
            payload = dict(data or {})
            payload.setdefault("status", "failed")
            payload.setdefault("error_message", err)
            self._apply_task_status(payload, source="socket", force=True)

        @self.sio.event
        def task_status(data):
            self._apply_task_status(data or {}, source="socket")

        @self.sio.event
        def pong(data):
            # 收到服务器的pong响应，连接正常
            pass

    def _format_duration(self, seconds: Optional[float]) -> str:
        """Format seconds as HH:MM:SS."""
        if seconds is None:
            return "--:--"
        try:
            total_seconds = max(0, int(float(seconds)))
        except (TypeError, ValueError):
            return "--:--"
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _apply_task_status(self, data: dict, source: str = "status", force: bool = False):
        """Update local task state and render meaningful status changes."""
        if not isinstance(data, dict):
            return

        task_id = data.get("task_id")
        if task_id:
            self.task_id = task_id

        current_status = data.get("status")
        if current_status:
            self.task_status = current_status

        self._latest_status = data
        self._render_task_status(data, source=source, force=force)

    def _render_task_status(self, data: dict, source: str = "status", force: bool = False):
        """Render deduplicated progress, phase, and timing information."""
        status = data.get("status") or self.task_status or "unknown"
        try:
            progress = int(float(data.get("progress", 0) or 0))
        except (TypeError, ValueError):
            progress = 0
        phase = data.get("phase", 0)
        elapsed = data.get("elapsed_seconds")
        duration = data.get("duration_seconds")

        # If the server has not sent timing fields yet, derive a local elapsed value.
        if elapsed is None and data.get("start_time"):
            try:
                elapsed = time.time() - float(data["start_time"])
            except (TypeError, ValueError):
                elapsed = None

        elapsed_bucket = int(float(elapsed or 0) // 30)
        status_key = (status, progress, phase, elapsed_bucket)
        with self._status_render_lock:
            if not force and status_key == self._last_status_key:
                return
            self._last_status_key = status_key

        message = (
            f"source={source}, status={status}, phase={phase}, "
            f"progress={progress}%, elapsed={self._format_duration(elapsed)}"
        )
        if duration is not None:
            message += f", duration={self._format_duration(duration)}"
        LiveStatus.update("客户端生成", message, force=force)

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        quiet: bool = False,
        **kwargs,
    ) -> Optional[dict]:
        """Perform an HTTP request with bounded retries and JSON parsing."""
        if path.startswith("http://") or path.startswith("https://"):
            url = path
        else:
            url = f"{self.server_url}{path}"

        attempts = max(1, max_retries if max_retries is not None else self.http_max_retries)
        request_timeout = timeout if timeout is not None else self.request_timeout

        for attempt in range(attempts):
            try:
                response = requests.request(method, url, timeout=request_timeout, **kwargs)
                try:
                    result = response.json()
                except ValueError:
                    result = {"error": response.text}

                if 200 <= response.status_code < 300:
                    return result if isinstance(result, dict) else {"data": result}

                error = result.get("error") if isinstance(result, dict) else response.text
                if response.status_code >= 500 and attempt < attempts - 1:
                    time.sleep(min(self.initial_reconnect_delay * (2 ** attempt), self.max_reconnect_delay))
                    continue
                if not quiet:
                    safe_print(f"❌ HTTP错误: {response.status_code} - {error}")
                return None
            except requests.exceptions.Timeout:
                if attempt >= attempts - 1:
                    if not quiet:
                        safe_print(f"❌ 请求超时: {url}")
                    return None
            except requests.exceptions.RequestException as e:
                if attempt >= attempts - 1:
                    if not quiet:
                        safe_print(f"❌ 请求失败: {e}")
                    return None

            time.sleep(min(self.initial_reconnect_delay * (2 ** attempt), self.max_reconnect_delay))

        return None

    def _wait_for_server_health(self, timeout: Optional[int] = None) -> bool:
        """Wait until the HTTP health endpoint is reachable."""
        deadline = time.time() + (timeout if timeout is not None else self.connect_timeout)
        while time.time() < deadline:
            result = self._request_json(
                "GET",
                "/api/health",
                timeout=self.status_timeout,
                max_retries=1,
                quiet=True,
            )
            if result and result.get("status") == "healthy":
                return True
            time.sleep(1)
        return False

    def _recover_connection(self, reason: str) -> bool:
        """尝试恢复SSH隧道与SocketIO连接。"""
        now = time.time()
        if now - self._last_recovery_ts < 10:
            return False

        if not self._recovery_lock.acquire(blocking=False):
            return False

        self._last_recovery_ts = now
        try:
            safe_print(f"⚠️ 检测到连接异常，开始恢复: {reason}")

            if self.auto_ssh:
                if self.tunnel_manager is None:
                    self.tunnel_manager = SSHTunnelManager(config=self.config)
                safe_print("🔄 正在重建SSH隧道...")
                if not self.tunnel_manager.reconnect():
                    safe_print("⚠️ SSH隧道重建失败，尝试直接启动...")
                    if not self.tunnel_manager.start():
                        safe_print("❌ SSH隧道恢复失败")
                        return False
                if not self.tunnel_manager.is_running():
                    safe_print("❌ SSH隧道未处于运行状态")
                    return False

            if not self._wait_for_server_health(timeout=self.connect_timeout):
                safe_print("❌ 服务端健康检查失败，恢复中止")
                return False

            # 先断开旧socket连接，再尝试重连
            try:
                if hasattr(self.sio, "connected") and self.sio.connected:
                    self.sio.disconnect()
            except Exception:
                pass
            self.connected = False

            try:
                self.sio.connect(
                    self.server_url,
                    transports=['websocket', 'polling'],
                    wait_timeout=self.connect_timeout,
                    socketio_path='/socket.io'
                )
            except Exception as e:
                safe_print(f"⚠️ SocketIO恢复连接失败: {e}")

            # 等待连接建立
            max_wait = min(10, self.connect_timeout)
            wait_count = 0.0
            while not self.connected and wait_count < max_wait:
                time.sleep(0.5)
                wait_count += 0.5
                if hasattr(self.sio, 'connected') and self.sio.connected:
                    self.connected = True
                    try:
                        self.client_sid = self.sio.get_sid()
                    except Exception:
                        pass
                    break

            if not self.connected:
                safe_print("❌ 连接恢复失败")
                return False

            if self.task_id and self.task_status not in ["completed", "failed"]:
                self.subscribe_task(self.task_id)
                status = self._get_task_status_with_retry(self.task_id, max_retries=1)
                if status:
                    self._apply_task_status(status, source="recover", force=True)

            self._start_heartbeat()
            safe_print("✅ 连接恢复成功")
            return True
        finally:
            self._recovery_lock.release()

    def _start_heartbeat(self):
        """启动心跳线程"""
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            return

        self._heartbeat_stop_event.clear()

        def heartbeat_worker():
            while not self._heartbeat_stop_event.wait(self.heartbeat_interval):
                if self.connected and not self._should_stop:
                    try:
                        self.sio.emit('ping', {'timestamp': time.time()})
                    except Exception:
                        pass  # 忽略心跳发送错误

        self._heartbeat_thread = threading.Thread(target=heartbeat_worker, daemon=True)
        self._heartbeat_thread.start()

    def _stop_heartbeat(self):
        """停止心跳线程"""
        self._heartbeat_stop_event.set()
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=2)
            self._heartbeat_thread = None
    
    def connect(self):
        """连接到服务器"""
        # 如果已经连接，直接返回
        if self.connected and hasattr(self.sio, 'connected') and self.sio.connected:
            return True
        
        self._should_stop = False

        try:
            # 如果需要自动建立SSH隧道
            if self.auto_ssh:
                if self.tunnel_manager is None:
                    safe_print("🔗 正在建立SSH隧道...")
                    self.tunnel_manager = SSHTunnelManager(config=self.config)
                elif not self.tunnel_manager.is_running():
                    safe_print("⚠️ 检测到SSH隧道失活，正在重建...")
                    self.tunnel_manager.reconnect()

                if not self.tunnel_manager.is_running():
                    if not self.tunnel_manager.start():
                        safe_print("❌ SSH隧道建立失败")
                        return False
                    safe_print("✅ SSH隧道已建立")
                    # 等待隧道稳定
                    time.sleep(1)

            if not self._wait_for_server_health(timeout=self.connect_timeout):
                safe_print(f"❌ 服务器健康检查失败: {self.server_url}/api/health")
                return False

            safe_print(f"🔗 正在连接服务器 {self.server_url}...")


            try:
                self.sio.connect(
                    self.server_url,
                    transports=['websocket', 'polling'],  # 优先websocket，必要时降级
                    wait_timeout=self.connect_timeout,
                    socketio_path='/socket.io'
                )
            except socketio.exceptions.ConnectionError as e:
                safe_print(f"⚠️ SocketIO连接警告: {e}")
                if hasattr(self.sio, 'connected') and self.sio.connected:
                    self.connected = True
                    try:
                        self.client_sid = self.sio.get_sid()
                    except:
                        pass

            # 等待连接建立，最多等待5秒
            max_wait = min(10, self.connect_timeout)
            wait_count = 0
            while not self.connected and wait_count < max_wait:
                time.sleep(0.5)
                wait_count += 0.5
                if hasattr(self.sio, 'connected') and self.sio.connected:
                    self.connected = True
                    try:
                        self.client_sid = self.sio.get_sid()
                    except:
                        pass
                    break

            if self.connected:
                safe_print("✅ SocketIO连接成功")
                # 启动心跳
                self._start_heartbeat()
                return True
            elif hasattr(self.sio, 'connected') and self.sio.connected:
                self.connected = True
                safe_print("✅ SocketIO连接成功（使用polling传输）")
                try:
                    self.client_sid = self.sio.get_sid()
                except:
                    pass
                # 启动心跳
                self._start_heartbeat()
                return True
            else:
                safe_print("❌ SocketIO连接失败")
                return False

        except socketio.exceptions.ConnectionError as e:
            safe_print(f"❌ SocketIO连接错误: {e}")
            return False
        except Exception as e:
            safe_print(f"❌ 连接失败: {e}")
            return False
    
    def disconnect(self):
        """断开连接"""
        self._should_stop = True

        # 停止心跳
        self._stop_heartbeat()

        # 先关闭SocketIO连接
        try:
            if self.connected or (hasattr(self.sio, 'connected') and self.sio.connected):
                self.sio.disconnect()
                safe_print("🔌 已断开服务器连接")
        except Exception as e:
            safe_print(f"⚠️ 断开SocketIO连接时出错: {e}")
        finally:
            self.connected = False

        # 关闭SSH隧道
        if self.tunnel_manager:
            try:
                self.tunnel_manager.stop()
            except Exception as e:
                safe_print(f"⚠️ 关闭SSH隧道时出错: {e}")
            finally:
                self.tunnel_manager = None
    
    def send_lyric_file(self, file_path: str, output_dir: Optional[str] = None) -> Optional[str]:
        """
        读取lyrics.jsonl文件并发送到服务端，服务器端会写入到sample/lyrics.jsonl
        
        Args:
            file_path: lyrics.jsonl文件的路径（本地路径，例如：D:/.../beauty_02/lyrics.jsonl）
            
        Returns:
            任务ID，如果失败则返回None
        """
        if not self.connected:
            safe_print("❌ 请先连接到服务器")
            return None
        
        # 读取文件内容
        content = read_lyric_file(file_path)
        if content is None:
            return None
        
        try:
            # 转换output_dir为服务器相对路径
            server_output_dir = None
            if output_dir:
                server_output_dir = convert_to_server_path(output_dir)
            
            # 发送文件内容到服务端
            filename = os.path.basename(file_path)
            safe_print(f"📤 正在上传 {filename} 到服务端（将写入到 sample/lyrics.jsonl）...")
            result = self._request_json(
                "POST",
                "/api/upload_lyric",
                json={
                    "content": content,
                    "filename": filename,  # 可能是 lyrics.jsonl 或 lyric.jsonl
                    "output_dir": server_output_dir
                },
                headers={"Content-Type": "application/json"},
                timeout=self.request_timeout,
            )
            if result and result.get('success'):
                safe_print(f"✅ 文件上传成功，服务器端将写入到 sample/lyrics.jsonl")
                return result.get('task_id')
            if result:
                safe_print(f"❌ 文件上传失败: {result.get('error', '未知错误')}")
            return None
        except requests.exceptions.RequestException as e:
            safe_print(f"❌ 请求失败: {e}")
            return None
    
    def start_inference(self, task_id: Optional[str] = None, generate_type: str = "both") -> Optional[str]:
        """
        启动推理任务
        
        Args:
            task_id: 任务ID，如果为None则使用上传文件时返回的task_id
        
        Returns:
            任务ID，如果失败则返回None
        """
        if not self.connected:
            safe_print("❌ 请先连接到服务器")
            return None
        
        try:
            safe_print("📤 正在启动推理任务...")
            # 如果client_sid为None，尝试获取
            if self.client_sid is None and hasattr(self.sio, 'get_sid'):
                try:
                    self.client_sid = self.sio.get_sid()
                except:
                    pass
            
            result = self._request_json(
                "POST",
                "/api/start_inference",
                json={
                    "client_sid": self.client_sid,  # 可能为None，服务器会处理
                    "task_id": task_id,
                    "generate_type": generate_type  # "normal", "bgm", "both"
                },
                headers={"Content-Type": "application/json"},
                timeout=self.request_timeout,
            )
            if result and result.get('success'):
                self.task_id = result.get('task_id')
                self.task_status = "running"
                safe_print(f"✅ 推理任务已启动")
                safe_print(f"   任务ID: {self.task_id}")
                
                # 订阅任务更新
                self.subscribe_task(self.task_id)
                return self.task_id
            if result:
                safe_print(f"❌ 启动推理任务失败: {result.get('error', '未知错误')}")
            return None
        except requests.exceptions.RequestException as e:
            safe_print(f"❌ 请求失败: {e}")
            return None
    
    def subscribe_task(self, task_id: str):
        """订阅任务更新"""
        if self.connected:
            self.sio.emit('subscribe_task', {'task_id': task_id})
            safe_print(f"📋 已订阅任务 {task_id} 的更新")
    
    def get_task_status(self, task_id: Optional[str] = None) -> Optional[dict]:
        """获取任务状态"""
        if task_id is None:
            task_id = self.task_id
        
        if task_id is None:
            safe_print("❌ 没有任务ID")
            return None
        
        status = self._request_json(
            "GET",
            f"/api/task/{task_id}/status",
            timeout=self.status_timeout,
            max_retries=1,
            quiet=True,
        )
        if status:
            return status
        safe_print("❌ 获取任务状态失败")
        return None
    
    def wait_for_inference_completion(self, task_id: Optional[str] = None, timeout: Optional[int] = None, poll_interval: Optional[int] = None) -> bool:
        """
        等待推理任务完成（使用HTTP轮询，更可靠）

        Args:
            task_id: 任务ID，如果为None则使用self.task_id
            timeout: 超时时间（秒），默认1小时
            poll_interval: 轮询间隔（秒），默认5秒

        Returns:
            是否成功完成
        """
        if task_id is None:
            task_id = self.task_id

        if task_id is None:
            safe_print("❌ 没有正在执行的任务")
            return False

        if timeout is None:
            timeout = self.task_timeout
        if poll_interval is None:
            poll_interval = self.poll_interval

        LiveStatus.start("客户端生成", f"等待任务完成 timeout={timeout}s")
        start_time = time.time()
        last_progress = -1
        consecutive_errors = 0
        max_consecutive_errors = self.max_status_errors

        try:
            while time.time() - start_time < timeout:
                # 优先检查SocketIO推送的状态
                if self.task_status in ["completed", "failed"]:
                    break

                # 使用HTTP轮询查询状态（更可靠）
                try:
                    status = self._get_task_status_with_retry(task_id)
                    if status:
                        self._apply_task_status(status, source="poll", force=False)
                        consecutive_errors = 0  # 重置错误计数
                        current_status = status.get('status', '')
                        if current_status == "completed":
                            self.task_status = "completed"
                            break
                        elif current_status == "failed":
                            self.task_status = "failed"
                            break

                        progress = status.get('progress', 0)

                        # 只在进度变化时更新
                        if progress != last_progress:
                            last_progress = progress
                            elapsed = status.get("elapsed_seconds", time.time() - start_time)
                            phase = status.get("phase", 0)
                            LiveStatus.update(
                                "客户端生成",
                                f"status={current_status}, phase={phase}, progress={progress}%, elapsed={self._format_duration(elapsed)}",
                            )
                    else:
                        consecutive_errors += 1
                        if consecutive_errors >= max_consecutive_errors:
                            safe_print(f"\n⚠️ 连续{consecutive_errors}次获取状态失败")
                            if self._recover_connection(f"status_poll_errors={consecutive_errors}"):
                                consecutive_errors = 0
                except Exception as e:
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        safe_print(f"\n⚠️ 连续{consecutive_errors}次获取状态失败: {e}")
                        if self._recover_connection(f"status_poll_exception={consecutive_errors}"):
                            consecutive_errors = 0

                time.sleep(poll_interval)

            # 确保进度条显示100%
            if self.task_status == "completed":
                if last_progress < 100:
                    LiveStatus.update("客户端生成", "status=completed, progress=100%", force=True)

        finally:
            if self.task_status not in ["completed", "failed"] and time.time() - start_time >= timeout:
                LiveStatus.update("客户端生成", "等待超时", force=True)

        if time.time() - start_time >= timeout:
            LiveStatus.finish("等待超时")
            safe_print("⏰ 等待超时")
            return False

        if self.task_status == "completed":
            duration = None
            if self._latest_status:
                duration = self._latest_status.get("duration_seconds") or self._latest_status.get("elapsed_seconds")
            LiveStatus.finish(f"任务成功完成 耗时={self._format_duration(duration)}")
            safe_print(f"✅ 任务成功完成! 耗时: {self._format_duration(duration)}")
            return True
        else:
            LiveStatus.finish("任务失败")
            safe_print("❌ 任务失败")
            return False

    def _get_task_status_with_retry(self, task_id: str, max_retries: int = 3) -> Optional[dict]:
        """带重试的获取任务状态"""
        return self._request_json(
            "GET",
            f"/api/task/{task_id}/status",
            timeout=self.status_timeout,
            max_retries=max_retries,
            quiet=True,
        )
    
    def list_audio_files(self, task_id: Optional[str] = None) -> List[str]:
        """
        获取生成的音频文件列表
        
        Args:
            task_id: 任务ID，如果为None则使用self.task_id
            
        Returns:
            音频文件名列表
        """
        return self.list_audio_files_by_phase(task_id=task_id, phase=None)

    def list_audio_files_by_phase(self, task_id: Optional[str] = None, phase: Optional[int] = None) -> List[str]:
        """
        获取生成的音频文件列表（可按phase过滤）

        Args:
            task_id: 任务ID，如果为None则使用self.task_id
            phase: None/1/2

        Returns:
            音频文件名列表
        """
        if task_id is None:
            task_id = self.task_id
        
        if task_id is None:
            safe_print("❌ 没有任务ID")
            return []
        
        params = {}
        if phase in (1, 2):
            params["phase"] = str(phase)
        result = self._request_json(
            "GET",
            f"/api/task/{task_id}/list_files",
            params=params,
            timeout=self.status_timeout,
            max_retries=1,
        )
        if not result:
            return []

        files = result.get('files', [])
        safe_print(f"📂 找到 {len(files)} 个音频文件")
        return files
    
    def download_audio_files(self, task_id: Optional[str] = None, output_dir: Optional[str] = None) -> bool:
        """
        下载所有音频文件到指定目录
        
        Args:
            task_id: 任务ID，如果为None则使用self.task_id
            output_dir: 输出目录，如果为None则从output_prompt文件路径推断
            
        Returns:
            是否下载成功
        """
        return self.download_audio_files_by_phase(task_id=task_id, output_dir=output_dir, phase=None)

    def download_audio_files_by_phase(self, task_id: Optional[str] = None, output_dir: Optional[str] = None, phase: Optional[int] = None) -> bool:
        """
        下载音频文件（可按phase过滤）
        """
        if task_id is None:
            task_id = self.task_id

        if task_id is None:
            safe_print("❌ 没有任务ID")
            return False

        # 获取文件列表
        files = self.list_audio_files_by_phase(task_id, phase=phase)
        if not files:
            safe_print("⚠️ 没有找到音频文件")
            return False

        # 过滤出.flac文件
        flac_files = [f for f in files if f.endswith(".flac")]
        if not flac_files:
            safe_print("⚠️ 没有找到.flac文件")
            return False

        phase_prefix = f"phase{phase} " if phase in (1, 2) else ""
        safe_print(f"📥 开始下载 {phase_prefix}{len(flac_files)} 个音频文件...")

        success_count = 0
        failed_files = []

        # 首轮下载
        for filename in flac_files:
            if self._download_file_with_resume(task_id, filename, output_dir):
                success_count += 1
                safe_print(f"   ✅ {filename}")
            else:
                failed_files.append(filename)
                safe_print(f"   ❌ {filename} - 首轮下载失败")

        # 补偿下载多轮
        max_extra_rounds = 3
        round_idx = 1
        while failed_files and round_idx <= max_extra_rounds:
            safe_print(f"📥 开始补偿下载(第{round_idx}轮): {len(failed_files)} 个文件...")
            remaining = []
            for filename in failed_files:
                if self._download_file_with_resume(task_id, filename, output_dir):
                    success_count += 1
                    safe_print(f"   ✅ {filename}")
                else:
                    remaining.append(filename)
                    safe_print(f"   ❌ {filename} - 补偿下载失败")
            failed_files = remaining
            round_idx += 1

        safe_print(f"📥 下载完成: {success_count}/{len(flac_files)} 个文件")
        if failed_files:
            safe_print(f"⚠️ 仍有 {len(failed_files)} 个文件下载失败: {failed_files}")
            return False
        
        # 验证下载的文件是否真的存在于目标目录
        if output_dir:
            safe_print(f"🔍 验证下载的文件是否存在于目标目录: {output_dir}")
            verified_count = 0
            for filename in flac_files:
                file_path = os.path.join(output_dir, filename)
                if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                    verified_count += 1
                else:
                    safe_print(f"   ⚠️ 文件不存在或为空: {filename}")
            
            if verified_count == len(flac_files):
                safe_print(f"✅ 所有 {verified_count} 个文件验证通过")
            else:
                safe_print(f"⚠️ 只有 {verified_count}/{len(flac_files)} 个文件验证通过")
        
        return True

    def _download_file_with_resume(
        self,
        task_id: str,
        filename: str,
        output_dir: Optional[str] = None,
        max_retries: int = 5,
        base_backoff: float = 1.0,
    ) -> bool:
        """
        单文件下载，支持分块写入、重试和断点续传（Range）。
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(output_dir, filename)
        else:
            file_path = filename

        url = f"{self.server_url}/api/task/{task_id}/download/{filename}"

        for attempt in range(max_retries):
            try:
                # 已有文件大小（用于续传）
                existing_size = 0
                if os.path.exists(file_path):
                    try:
                        existing_size = os.path.getsize(file_path)
                    except OSError:
                        existing_size = 0

                headers = {}

                # 先探测远端大小
                total_size = None
                try:
                    head_resp = requests.head(url, timeout=self.status_timeout)
                    if head_resp.status_code in (200, 206):
                        cl = head_resp.headers.get("Content-Length")
                        if cl is not None:
                            total_size = int(cl)
                except requests.RequestException:
                    # HEAD 失败不致命，可以继续用 GET
                    pass

                # 如果本地已完整，则直接认为成功
                if total_size is not None and existing_size >= total_size > 0:
                    return True

                # 需要续传则加 Range
                if existing_size > 0 and total_size and existing_size < total_size:
                    headers["Range"] = f"bytes={existing_size}-"

                # 真正下载
                resp = requests.get(
                    url,
                    headers=headers,
                    timeout=self.download_timeout,
                    stream=True,
                )

                if resp.status_code not in (200, 206):
                    safe_print(f"   ❌ {filename} - 下载失败: HTTP {resp.status_code}")
                else:
                    # 选择写入模式
                    mode = "ab" if resp.status_code == 206 and "Range" in headers else "wb"

                    with open(file_path, mode) as f:
                        for chunk in resp.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)

                    # 校验大小（若已知）
                    if total_size is not None:
                        try:
                            final_size = os.path.getsize(file_path)
                        except OSError:
                            final_size = -1
                        if final_size < total_size:
                            safe_print(f"   ⚠️ {filename} - 大小不完整({final_size}/{total_size})，将重试")
                        else:
                            return True
                    else:
                        # 没有 Content-Length，只要 HTTP 成功就认为成功
                        return True

            except Exception as e:
                safe_print(f"   ❌ {filename} - 下载异常: {e}")

            # 指数退避
            sleep_time = base_backoff * (2 ** attempt)
            time.sleep(min(sleep_time, 30))

        return False

    def continue_to_bgm(self, task_id: Optional[str] = None) -> bool:
        """通知服务端继续执行BGM阶段"""
        if task_id is None:
            task_id = self.task_id
        if task_id is None:
            safe_print("❌ 没有任务ID")
            return False

        try:
            result = self._request_json(
                "POST",
                f"/api/task/{task_id}/continue",
                json={},
                headers={"Content-Type": "application/json"},
                timeout=self.request_timeout,
            )
            if result and result.get("success"):
                safe_print("✅ 已通知服务端继续生成BGM阶段")
                return True
            if result:
                safe_print(f"❌ 继续失败: {result.get('error', '未知错误')}")
            return False
        except Exception as e:
            safe_print(f"❌ 请求失败: {e}")
            return False
    
    def process_lyric_file(self, file_path: str, output_dir: Optional[str] = None, auto_disconnect: bool = True, generate_type: str = "both") -> bool:
        """
        完整的处理流程：读取文件、发送、等待推理、下载音频
        Args:
            file_path: lyric.jsonl文件的路径
            output_dir: 输出目录，如果为None则从file_path推断
            auto_disconnect: 是否在处理完成后自动断开连接，默认为True
                           设置为False时，可以复用连接处理多个文件
            generate_type: 生成类型，"normal"（只生成无-bgm）、"bgm"（只生成-bgm）、"both"（两阶段都生成）
        Returns:
            是否成功完成所有步骤
        """
        try:
            if not self.connected or not (hasattr(self.sio, 'connected') and self.sio.connected):
                if not self.connect():
                    return False
            

            task_id = self.send_lyric_file(file_path, output_dir=output_dir)
            if task_id is None:
                return False
            

            inference_task_id = self.start_inference(task_id=task_id, generate_type=generate_type)
            if inference_task_id is None:
                return False
            

            if output_dir is None:
                output_dir = get_output_dir(file_path)

            if generate_type == "both":
                # 两阶段生成流程
                safe_print("⏳ 等待第一阶段生成完成（phase=1）...")
                if not self._wait_for_phase_files(inference_task_id, phase=1):
                    return False

                if not self.download_audio_files_by_phase(inference_task_id, output_dir, phase=1):
                    return False
                
                # 验证第一阶段下载的文件
                if not self._verify_downloaded_files(inference_task_id, output_dir, phase=1):
                    safe_print("⚠️ 第一阶段下载的文件验证失败，但下载过程已完成")

                # 4.2 通知服务端继续BGM阶段
                if not self.continue_to_bgm(inference_task_id):
                    return False

                # 4.3 等待整体任务完成（phase=2结束后会completed）
                if not self.wait_for_inference_completion(inference_task_id):
                    return False

                # 4.4 下载第二阶段文件
                if not self.download_audio_files_by_phase(inference_task_id, output_dir, phase=2):
                    return False
                
                # 验证第二阶段下载的文件
                if not self._verify_downloaded_files(inference_task_id, output_dir, phase=2):
                    safe_print("⚠️ 第二阶段下载的文件验证失败，但下载过程已完成")

                safe_print("🎉 两阶段（mixed + bgm）全部完成!")
            elif generate_type == "normal":
                # 只生成无-bgm版本
                safe_print("⏳ 等待生成完成（无-bgm版本）...")
                if not self.wait_for_inference_completion(inference_task_id):
                    return False
                
                if not self.download_audio_files_by_phase(inference_task_id, output_dir, phase=1):
                    return False
                
                # 验证下载的文件是否真的存在于目标目录
                if not self._verify_downloaded_files(inference_task_id, output_dir, phase=1):
                    safe_print("⚠️ 下载的文件验证失败，但下载过程已完成")
                    # 不返回False，因为文件可能已经下载，只是验证有问题
                
                safe_print("🎉 无-bgm版本生成完成!")
            elif generate_type == "bgm":
                # 只生成-bgm版本
                safe_print("⏳ 等待生成完成（-bgm版本）...")
                if not self.wait_for_inference_completion(inference_task_id):
                    return False
                
                if not self.download_audio_files_by_phase(inference_task_id, output_dir, phase=2):
                    return False
                
                # 验证下载的文件是否真的存在于目标目录
                if not self._verify_downloaded_files(inference_task_id, output_dir, phase=2):
                    safe_print("⚠️ 下载的文件验证失败，但下载过程已完成")
                    # 不返回False，因为文件可能已经下载，只是验证有问题
                
                safe_print("🎉 -bgm版本生成完成!")
            
            return True
        except Exception as e:
            safe_print(f"❌ 处理过程中出错: {e}")
            return False
        finally:
            if auto_disconnect:
                self.disconnect()

    def _wait_for_phase_files(self, task_id: str, phase: int, timeout: Optional[int] = None, poll_interval: Optional[int] = None) -> bool:
        """等待某个phase的文件列表出现（用于phase1先下载）"""
        if timeout is None:
            timeout = self.task_timeout
        if poll_interval is None:
            poll_interval = self.poll_interval
        start_time = time.time()
        have_entered_phase = False
        while time.time() - start_time < timeout:
            status = self._get_task_status_with_retry(task_id, max_retries=1)
            if status:
                self._apply_task_status(status, source=f"phase{phase}", force=False)
            if status and status.get("status") == "failed":
                safe_print(f"❌ 任务失败: {status.get('error_message')}")
                return False

            current_phase = status.get("phase", 0) if status else 0
            if current_phase >= phase:
                have_entered_phase = True
                files = self.list_audio_files_by_phase(task_id, phase=phase)
                flac_files = [f for f in files if f.endswith(".flac")]
                if flac_files:
                    safe_print(f"✅ phase{phase} 已生成 {len(flac_files)} 个 .flac 文件")
                    return True

            time.sleep(poll_interval)
            if have_entered_phase and self.task_status not in ["completed", "failed"]:
                if not self.connected and self.auto_ssh:
                    self._recover_connection(f"phase_wait_disconnect_phase={phase}")
        if not have_entered_phase:
            safe_print(f"⏰ 等待进入 phase{phase} 超时")
        else:
            safe_print(f"⏰ 等待 phase{phase} 文件超时")
        return False
    
    def _verify_downloaded_files(self, task_id: str, output_dir: Optional[str], phase: Optional[int] = None) -> bool:
        """
        验证下载的文件是否真的存在于目标目录
        
        Args:
            task_id: 任务ID
            output_dir: 输出目录
            phase: phase编号（1或2）
            
        Returns:
            是否所有文件都存在
        """
        if output_dir is None:
            safe_print("⚠️ 无法验证文件：输出目录未指定")
            return False
        
        # 获取应该下载的文件列表
        files = self.list_audio_files_by_phase(task_id, phase=phase)
        flac_files = [f for f in files if f.endswith(".flac")]
        
        if not flac_files:
            safe_print("⚠️ 没有文件需要验证")
            return False
        
        safe_print(f"🔍 验证 {len(flac_files)} 个下载的文件是否存在于目标目录...")
        
        missing_files = []
        existing_files = []
        
        for filename in flac_files:
            file_path = os.path.join(output_dir, filename)
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                existing_files.append(filename)
            else:
                missing_files.append(filename)
        
        if missing_files:
            safe_print(f"❌ 验证失败: {len(missing_files)} 个文件不存在或为空:")
            for filename in missing_files:
                safe_print(f"   - {filename}")
            safe_print(f"✅ 验证通过: {len(existing_files)} 个文件存在")
            return False
        else:
            safe_print(f"✅ 验证通过: 所有 {len(existing_files)} 个文件都已存在于目标目录")
            return True


def main():
    """
    主函数示例
    使用说明：
    1. 如果客户端和服务器在同一台机器上（都在容器内）：
       client = MusicGenerationClient(server_url="http://localhost:6006", auto_ssh=False)
    2. 如果客户端在远程机器上，需要通过SSH隧道：
       client = MusicGenerationClient(server_url="http://localhost:6006", auto_ssh=True)
       注意：SSH隧道会自动将 localhost:6006 映射到远程服务器的 172.17.0.6:6006
    """


    file_path = r"D:\Python\Project\MCP\Agent\prompt_agent\sample\output\beauty_2\lyric.jsonl"

    client = MusicGenerationClient(server_url="http://localhost:6006", auto_ssh=True)
    
    if client.process_lyric_file(file_path):
        print("处理成功!")
    else:
        print("处理失败!")

def run_main(folder_path):
    """批量处理文件，复用同一个客户端连接"""
    # 创建单个客户端实例，复用连接
    client = MusicGenerationClient(server_url="http://localhost:6006", auto_ssh=True)
    
    try:
        if not client.connect():
            safe_print("❌ 无法连接到服务器")
            return
        
        success_count = 0
        fail_count = 0
        
        for root, dirs, files in os.walk(folder_path):
            for _dir in dirs:
                folder_list=[]
                sub_folder_path = os.path.join(root, _dir)
                for item in os.listdir(sub_folder_path):
                    folder_list.append(item)
                if "audio.jsonl" in folder_list and "lyric.jsonl" in folder_list and "photo.jsonl" in folder_list and "scene.jsonl" in folder_list and "text.jsonl" in folder_list and "video.jsonl" in folder_list:
                    print(f"sub_root:{os.path.join(sub_folder_path, 'lyric.jsonl')}")
                    file_path = os.path.join(sub_folder_path, "lyric.jsonl")
                    
                    # 处理文件，但不自动断开连接（auto_disconnect=False）
                    if client.process_lyric_file(file_path, auto_disconnect=False):
                        success_count += 1
                        print("处理成功!")
                    else:
                        fail_count += 1
                        print("处理失败!")

                    if not client.connected or not (hasattr(client.sio, 'connected') and client.sio.connected):
                        safe_print("⚠️ 连接已断开，尝试重连...")
                        if not client.connect():
                            safe_print("❌ 重连失败，跳过后续文件")
                            break
        safe_print(f"\n📊 处理完成: 成功 {success_count} 个，失败 {fail_count} 个")
    finally:
        client.disconnect()


def convert_to_server_path(windows_path: str) -> str:
    """
    将Windows路径转换为服务器相对路径
    
    Args:
        windows_path: Windows绝对路径，例如 D:/Python/Project/MCP/Agent/prompt_agent/sample/output/scene_17
        
    Returns:
        服务器相对路径，例如 sample/output/scene_17
    """
    # 获取项目根目录（从当前文件位置推断）
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_file_dir)  # client的父目录是项目根目录
    
    # 标准化路径
    windows_path = os.path.normpath(windows_path)
    project_root = os.path.normpath(project_root)
    
    try:
        # 计算相对路径
        rel_path = os.path.relpath(windows_path, project_root)
        # 将Windows路径分隔符转换为Unix风格
        rel_path = rel_path.replace('\\', '/')
        return rel_path
    except ValueError:
        # 如果路径不在项目根目录下，尝试提取sample/output之后的部分
        if 'sample' in windows_path and 'output' in windows_path:
            parts = windows_path.split('sample')
            if len(parts) > 1:
                remaining = parts[1].replace('\\', '/')
                return 'sample' + remaining
        # 如果无法转换，返回原路径（去掉盘符）
        return windows_path.replace('\\', '/').lstrip('/')


def check_missing_audio_files(folder_path: str) -> dict:
    """
    检查缺少的音频文件
    
    文件名格式可能是：
    - 时间戳-idx.flac (例如: 2026-01-14-11-44-SeeYouNextTime.flac)
    - 时间戳-idx-bgm.flac (例如: 2026-01-14-11-44-SeeYouNextTime-bgm.flac)
    - 或者直接是 idx.flac 或 idx-bgm.flac
    
    Args:
        folder_path: 目录路径，例如 D:/Python/Project/MCP/Agent/prompt_agent/sample/output/scene_17
        
    Returns:
        {
            "missing_normal": ["idx1", "idx2", ...],  # 缺少无-bgm版本
            "missing_bgm": ["idx1", "idx2", ...]     # 缺少-bgm版本
        }
    """
    missing_normal = []
    missing_bgm = []
    
    # 读取目录下所有.flac文件，提取idx
    # 文件名格式可能是：时间戳-idx.flac 或 时间戳-idx-bgm.flac
    # 或者直接是：idx.flac 或 idx-bgm.flac
    # 时间戳格式：YYYY-MM-DD-HH-MM (5个部分)
    found_normal_indices = set()
    found_bgm_indices = set()
    
    def extract_idx_from_filename(filename: str, is_bgm: bool = False) -> str:
        """
        从文件名中提取idx
        文件名格式：时间戳-idx.flac 或 时间戳-idx-bgm.flac
        
        注意：如果lyric.jsonl中的idx包含时间戳，那么提取的idx也应该包含时间戳
        例如：文件名 2026-01-14-11-35-BeautyRoutine.flac 应该提取为 2026-01-14-11-35-BeautyRoutine
        """
        # 去掉扩展名
        name = os.path.splitext(filename)[0]
        
        # 如果是-bgm版本，先去掉-bgm后缀
        if is_bgm and name.endswith('-bgm'):
            name = name[:-4]  # 去掉 '-bgm'
        
        # 分割文件名
        parts = name.split('-')
        
        # 检查前5个部分是否是时间戳格式（都是数字）
        # 时间戳格式：YYYY-MM-DD-HH-MM
        if len(parts) >= 6:
            # 检查前5个部分是否都是数字（可能是时间戳）
            is_timestamp = True
            for i in range(min(5, len(parts))):
                if not parts[i].isdigit():
                    is_timestamp = False
                    break
            
            if is_timestamp:
                # 前5部分是时间戳，idx应该包含时间戳（因为lyric.jsonl中的idx也包含时间戳）
                # 例如：2026-01-14-11-35-BeautyRoutine -> 2026-01-14-11-35-BeautyRoutine
                idx = name  # 保留整个name作为idx（包含时间戳）
            else:
                # 不是时间戳格式，整个name就是idx
                idx = name
        else:
            # 少于6个部分，整个name就是idx
            idx = name
        
        return idx
    
    if os.path.exists(folder_path):
        for item in os.listdir(folder_path):
            if item.endswith('.flac'):
                # 检查是否是-bgm版本
                if item.endswith('-bgm.flac'):
                    idx = extract_idx_from_filename(item, is_bgm=True)
                    found_bgm_indices.add(idx)
                else:
                    idx = extract_idx_from_filename(item, is_bgm=False)
                    found_normal_indices.add(idx)
    
    # 读取lyric.jsonl文件获取所有idx
    lyric_file_path = os.path.join(folder_path, 'lyric.jsonl')
    if not os.path.exists(lyric_file_path):
        safe_print(f"❌ lyric.jsonl文件不存在: {lyric_file_path}")
        return {"missing_normal": [], "missing_bgm": []}
    
    idx_list = []
    lyric_data = []
    try:
        with open(lyric_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    idx = str(item.get('idx', ''))
                    if idx:
                        idx_list.append(idx)
                        lyric_data.append(item)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        safe_print(f"❌ 读取lyric.jsonl失败: {e}")
        return {"missing_normal": [], "missing_bgm": []}
    
    # 检查缺少哪些文件
    for idx in idx_list:
        # 检查无-bgm版本
        if idx not in found_normal_indices:
            missing_normal.append(idx)
        
        # 检查-bgm版本
        if idx not in found_bgm_indices:
            missing_bgm.append(idx)
    
    safe_print(f"📊 检查结果:")
    safe_print(f"   总共有 {len(idx_list)} 首歌曲")
    safe_print(f"   已找到无-bgm版本: {len(found_normal_indices)} 首")
    safe_print(f"   已找到-bgm版本: {len(found_bgm_indices)} 首")
    safe_print(f"   缺少无-bgm版本: {len(missing_normal)} 首")
    safe_print(f"   缺少-bgm版本: {len(missing_bgm)} 首")
    
    return {
        "missing_normal": missing_normal,
        "missing_bgm": missing_bgm
    }


def create_lyrics_for_missing(folder_path: str, missing_indices: list, add_bgm_suffix: bool = False) -> str:
    """
    为缺少的音乐创建lyrics.jsonl文件
    
    Args:
        folder_path: 目录路径
        missing_indices: 缺少的idx列表
        add_bgm_suffix: 是否给idx添加-bgm后缀
        
    Returns:
        lyrics.jsonl文件的路径
    """
    lyric_file_path = os.path.join(folder_path, 'lyric.jsonl')
    if not os.path.exists(lyric_file_path):
        safe_print(f"❌ lyric.jsonl文件不存在: {lyric_file_path}")
        return ""
    
    # 读取lyric.jsonl
    lyric_data = []
    try:
        with open(lyric_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    lyric_data.append(item)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        safe_print(f"❌ 读取lyric.jsonl失败: {e}")
        return ""
    
    # 创建lyrics.jsonl（清空原有内容）
    lyrics_path = os.path.join(folder_path, 'lyrics.jsonl')
    try:
        # 如果文件已存在，先删除以确保清空
        if os.path.exists(lyrics_path):
            try:
                os.remove(lyrics_path)
            except Exception:
                pass
        
        # 写入新内容（'w' 模式会自动创建新文件）
        with open(lyrics_path, 'w', encoding='utf-8') as f:
            for item in lyric_data:
                idx = str(item.get('idx', ''))
                if idx in missing_indices:
                    # 如果需要添加-bgm后缀
                    if add_bgm_suffix:
                        item['idx'] = f"{idx}-bgm"
                    # 写入文件
                    data = json.dumps(item, ensure_ascii=False)
                    f.write(data + '\n')
        
        safe_print(f"✅ 已创建lyrics.jsonl: {lyrics_path}（已清空原有内容）")
        safe_print(f"   包含 {len(missing_indices)} 首歌曲")
        if add_bgm_suffix:
            safe_print(f"   已添加-bgm后缀")
        return lyrics_path
    except Exception as e:
        safe_print(f"❌ 创建lyrics.jsonl失败: {e}")
        return ""


def check_and_generate_missing_music(folder_path: str, auto_disconnect: bool = True) -> bool:
    """
    自动检查缺少的音频文件并生成
    Args:
        folder_path: 目录路径，例如 D:/Python/Project/MCP/Agent/prompt_agent/sample/output/scene_17
        auto_disconnect: 是否在处理完成后自动断开连接
    Returns:
        是否成功完成所有步骤
    """
    # 1. 检查缺少的文件
    safe_print(f"🔍 开始检查缺少的音频文件: {folder_path}")
    missing_info = check_missing_audio_files(folder_path)
    
    missing_normal = missing_info.get("missing_normal", [])
    missing_bgm = missing_info.get("missing_bgm", [])
    
    if not missing_normal and not missing_bgm:
        safe_print("✅ 所有音频文件都已存在，无需生成")
        return True
    
    # 2. 创建客户端连接
    client = MusicGenerationClient(server_url="http://localhost:6006", auto_ssh=True)
    
    try:
        if not client.connect():
            safe_print("❌ 无法连接到服务器")
            return False
        
        # 3. 根据缺少情况执行相应策略
        success = True
        
        if missing_normal and missing_bgm:
            # 情况3：两者都缺少，先执行无-bgm，再执行-bgm
            safe_print("📋 检测到缺少无-bgm和-bgm版本，将分别生成")
            
            # 3.1 生成无-bgm版本
            safe_print("\n" + "="*60)
            safe_print("第一步：生成无-bgm版本")
            safe_print("="*60)
            lyrics_path = create_lyrics_for_missing(folder_path, missing_normal, add_bgm_suffix=False)
            if not lyrics_path:
                safe_print("❌ 创建lyrics.jsonl失败（无-bgm）")
                return False
            
            if not client.process_lyric_file(lyrics_path, output_dir=folder_path, auto_disconnect=False, generate_type="normal"):
                safe_print("❌ 生成无-bgm版本失败")
                success = False
            else:
                safe_print("✅ 无-bgm版本生成完成")
                # 重新检查文件夹，验证文件是否已下载
                safe_print("🔍 重新检查文件夹，验证无-bgm版本文件...")
                time.sleep(1)  # 等待文件系统同步
                verify_info = check_missing_audio_files(folder_path)
                remaining_normal = verify_info.get("missing_normal", [])
                if remaining_normal:
                    safe_print(f"⚠️ 仍有 {len(remaining_normal)} 个无-bgm版本文件缺失: {remaining_normal}")
                    success = False
                else:
                    safe_print("✅ 无-bgm版本文件验证通过，所有文件已存在")
            
            # 3.2 生成-bgm版本（只有在前一步成功时才执行）
            if success:
                safe_print("\n" + "="*60)
                safe_print("第二步：生成-bgm版本")
                safe_print("="*60)
                lyrics_path = create_lyrics_for_missing(folder_path, missing_bgm, add_bgm_suffix=True)
                if not lyrics_path:
                    safe_print("❌ 创建lyrics.jsonl失败（-bgm）")
                    success = False
                else:
                    if not client.process_lyric_file(lyrics_path, output_dir=folder_path, auto_disconnect=False, generate_type="bgm"):
                        safe_print("❌ 生成-bgm版本失败")
                        success = False
                    else:
                        safe_print("✅ -bgm版本生成完成")
                        # 重新检查文件夹，验证文件是否已下载
                        safe_print("🔍 重新检查文件夹，验证-bgm版本文件...")
                        time.sleep(1)  # 等待文件系统同步
                        verify_info = check_missing_audio_files(folder_path)
                        remaining_bgm = verify_info.get("missing_bgm", [])
                        if remaining_bgm:
                            safe_print(f"⚠️ 仍有 {len(remaining_bgm)} 个-bgm版本文件缺失: {remaining_bgm}")
                            success = False
                        else:
                            safe_print("✅ -bgm版本文件验证通过，所有文件已存在")
        
        elif missing_normal:
            # 情况1：只缺少无-bgm
            safe_print("📋 检测到只缺少无-bgm版本")
            lyrics_path = create_lyrics_for_missing(folder_path, missing_normal, add_bgm_suffix=False)
            if not lyrics_path:
                safe_print("❌ 创建lyrics.jsonl失败")
                success = False
            else:
                if not client.process_lyric_file(lyrics_path, output_dir=folder_path, auto_disconnect=False, generate_type="normal"):
                    safe_print("❌ 生成无-bgm版本失败")
                    success = False
                else:
                    safe_print("✅ 无-bgm版本生成完成")
                    # 重新检查文件夹，验证文件是否已下载
                    safe_print("🔍 重新检查文件夹，验证无-bgm版本文件...")
                    time.sleep(1)  # 等待文件系统同步
                    verify_info = check_missing_audio_files(folder_path)
                    remaining_normal = verify_info.get("missing_normal", [])
                    if remaining_normal:
                        safe_print(f"⚠️ 仍有 {len(remaining_normal)} 个无-bgm版本文件缺失: {remaining_normal}")
                        success = False
                    else:
                        safe_print("✅ 无-bgm版本文件验证通过，所有文件已存在")
        
        elif missing_bgm:
            # 情况2：只缺少-bgm
            safe_print("📋 检测到只缺少-bgm版本")
            lyrics_path = create_lyrics_for_missing(folder_path, missing_bgm, add_bgm_suffix=True)
            if not lyrics_path:
                safe_print("❌ 创建lyrics.jsonl失败")
                success = False
            else:
                if not client.process_lyric_file(lyrics_path, output_dir=folder_path, auto_disconnect=False, generate_type="bgm"):
                    safe_print("❌ 生成-bgm版本失败")
                    success = False
                else:
                    safe_print("✅ -bgm版本生成完成")
                    # 重新检查文件夹，验证文件是否已下载
                    safe_print("🔍 重新检查文件夹，验证-bgm版本文件...")
                    time.sleep(1)  # 等待文件系统同步
                    verify_info = check_missing_audio_files(folder_path)
                    remaining_bgm = verify_info.get("missing_bgm", [])
                    if remaining_bgm:
                        safe_print(f"⚠️ 仍有 {len(remaining_bgm)} 个-bgm版本文件缺失: {remaining_bgm}")
                        success = False
                    else:
                        safe_print("✅ -bgm版本文件验证通过，所有文件已存在")
        
        # 最终检查所有文件
        if success:
            safe_print("\n" + "="*60)
            safe_print("最终验证：检查所有文件")
            safe_print("="*60)
            final_check = check_missing_audio_files(folder_path)
            final_missing_normal = final_check.get("missing_normal", [])
            final_missing_bgm = final_check.get("missing_bgm", [])
            
            if final_missing_normal or final_missing_bgm:
                safe_print(f"⚠️ 最终检查发现仍有文件缺失:")
                if final_missing_normal:
                    safe_print(f"   缺少无-bgm版本: {len(final_missing_normal)} 首 - {final_missing_normal}")
                if final_missing_bgm:
                    safe_print(f"   缺少-bgm版本: {len(final_missing_bgm)} 首 - {final_missing_bgm}")
                success = False
            else:
                safe_print("✅ 所有音频文件都已存在，验证通过!")
        
        if success:
            safe_print("\n🎉 所有缺少的音频文件已生成完成!")
        
        return success
        
    except Exception as e:
        safe_print(f"❌ 处理过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if auto_disconnect:
            client.disconnect()


def create_lyrics(folder_path)->str:
    """
    用于创造一个没有音乐的乐谱lyrics.jsonl发送至服务器进行歌曲创作
    Args:
        folder_path:待处理的文件 D:/Python/Project/MCP/Agent/prompt_agent/sample/output/beauty_02/lyric.jsonl
    Returns:
        None
    """
    root,file=os.path.split(folder_path)
    #print(f"{root}")
    #print(f"{file}")
    name_list=[]
    for item in os.listdir(root):
        #print(f"{item}")
        name,ext=os.path.splitext(item)
        if ext==".flac":
            name_list.append(name)
    #print(name_list)
    idx_list=[]
    lyric_list=[]
    with open(folder_path,"r",encoding="utf-8") as f:
        for item in f:
            item=json.loads(item)
            #print(f"{item}\n{type(item)}")
            value=item["idx"]
            idx_list.append(str(value))
            lyric_list.append(item)
            #print(f"{item}")
            #print(f"{value}\n{type(item)}")

    lyrics_path = os.path.join(root,"lyrics.jsonl")
    missing_count = 0
    with open(lyrics_path, "w", encoding="utf-8") as f:
        for item in idx_list:
            if item in name_list:
                continue
            else:
                for lyric in lyric_list:
                    if item==lyric["idx"]:
                        data=json.dumps(lyric, ensure_ascii=False)
                        f.write(data+'\n')
                        missing_count += 1
                        break
    
    # 如果所有文件都已存在，返回空字符串
    if missing_count == 0:
        safe_print(f"⚠️ 所有音频文件都已存在，无需创建lyrics.jsonl")
        # 删除空文件
        if os.path.exists(lyrics_path):
            try:
                os.remove(lyrics_path)
            except:
                pass
        return ""
    
    safe_print(f"✅ {lyrics_path} is created! (包含 {missing_count} 首歌曲)")
    return str(lyrics_path)

def run_music():
    """
    主函数示例 - 自动检查并生成缺少的音频文件（支持无-bgm和-bgm版本）
    使用说明：
    1. 如果客户端和服务器在同一台机器上（都在容器内）：
       client = MusicGenerationClient(server_url="http://localhost:6006", auto_ssh=False)
    2. 如果客户端在远程机器上，需要通过SSH隧道：
       client = MusicGenerationClient(server_url="http://localhost:6006", auto_ssh=True)
       注意：SSH隧道会自动将 localhost:6006 映射到远程服务器的 172.17.0.6:6006
    """

    r"""    
    file_path_list=[
        r"D:\Python\Project\MCP\Agent\prompt_agent\sample\output_video_audio\beauty_02\lyric.jsonl",
        r"D:\Python\Project\MCP\Agent\prompt_agent\sample\output_video_audio\food_04\lyric.jsonl",
        r"D:\Python\Project\MCP\Agent\prompt_agent\sample\output_video_audio\live_13\lyric.jsonl",
        r"D:\Python\Project\MCP\Agent\prompt_agent\sample\output_video_audio\pet_10\lyric.jsonl",
        r"D:\Python\Project\MCP\Agent\prompt_agent\sample\output_video_audio\car_02\lyric.jsonl",
        r"D:\Python\Project\MCP\Agent\prompt_agent\sample\output_video_audio\knowledge_11\lyric.jsonl",
        r"D:\Python\Project\MCP\Agent\prompt_agent\sample\output_video_audio\scene_17\lyric.jsonl",
        r"D:\Python\Project\MCP\Agent\prompt_agent\sample\output_video_audio\skill_02\lyric.jsonl",
        r"D:\Python\Project\MCP\Agent\prompt_agent\sample\output_video_audio\sport_02\lyric.jsonl",
        r"D:\Python\Project\MCP\Agent\prompt_agent\sample\output_video_audio\technology_10\lyric.jsonl",

        r"D:\Python\Project\MCP\Agent\prompt_agent\sample\output_video_audio\beauty_06\lyric.jsonl",
        r"D:\Python\Project\MCP\Agent\prompt_agent\sample\output_video_audio\food_02\lyric.jsonl",
        r"D:\Python\Project\MCP\Agent\prompt_agent\sample\output_video_audio\live_02\lyric.jsonl",
        r"D:\Python\Project\MCP\Agent\prompt_agent\sample\output_video_audio\pet_14\lyric.jsonl",
        r"D:\Python\Project\MCP\Agent\prompt_agent\sample\output_video_audio\car_04\lyric.jsonl",
        r"D:\Python\Project\MCP\Agent\prompt_agent\sample\output_video_audio\knowledge_20\lyric.jsonl",
        r"D:\Python\Project\MCP\Agent\prompt_agent\sample\output_video_audio\scene_05\lyric.jsonl",
        r"D:\Python\Project\MCP\Agent\prompt_agent\sample\output_video_audio\skill_14\lyric.jsonl",
        r"D:\Python\Project\MCP\Agent\prompt_agent\sample\output_video_audio\sport_10\lyric.jsonl",
        r"D:\Python\Project\MCP\Agent\prompt_agent\sample\output_video_audio\technology_19\lyric.jsonl",
    ]
    """
    file_path_list=[
        r"D:\Python\Project\MCP\Agent\prompt_agent\sample\output\1\lyric.jsonl",
        r"D:\Python\Project\MCP\Agent\prompt_agent\sample\output\2\lyric.jsonl",
        r"D:\Python\Project\MCP\Agent\prompt_agent\sample\output\3\lyric.jsonl",
        r"D:\Python\Project\MCP\Agent\prompt_agent\sample\output\4\lyric.jsonl",
        r"D:\Python\Project\MCP\Agent\prompt_agent\sample\output\5\lyric.jsonl",
    ]



    for file_path in file_path_list:
        folder_path = os.path.dirname(file_path)
        
        safe_print(f"\n{'='*60}")
        safe_print(f"处理文件夹: {folder_path}")
        safe_print(f"{'='*60}")
        
        # 使用check_and_generate_missing_music自动检查并生成缺少的文件
        # 该函数会自动：
        # 1. 检查缺少无-bgm和-bgm版本的文件
        # 2. 根据缺少情况分别生成：
        #    - 只缺少无-bgm：生成无-bgm版本
        #    - 只缺少-bgm：生成-bgm版本
        #    - 两者都缺少：先生成无-bgm，再生成-bgm
        if check_and_generate_missing_music(folder_path, auto_disconnect=True):
            safe_print(f"✅ {folder_path} 处理成功!")
        else:
            safe_print(f"❌ {folder_path} 处理失败!")



if __name__ == "__main__":
    #main()
    #folder_path=r"D:\Python\Project\MCP\Agent\prompt_agent\sample\output"
    #run_main(folder_path)
    run_music()
